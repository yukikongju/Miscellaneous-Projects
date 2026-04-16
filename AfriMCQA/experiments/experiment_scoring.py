"""
Script to score model inference

How the script works:
1. Given model_name and experiment_setup, for QUES_<lang>_ANS_EN and QUES_<lang>_ANS_NATIVE, generate a dataframe with the following and save the output in "evaluation/<EXPERIMENT_SETUP>/QUES_<lang>_ANS_<EN/NATIVE>.csv". Only compute if not computed yet:
    - correct_pattern_matching: bool => if each ouput generated a correct answer using pattern matching
    - correct_llm_as_judge: bool => use LLM as judge to determine if generated output is correct by checking if output and ground truth are semantically the same
    - explanation_types: List[str]
    => dataframe: ID, Country, Language, Category, augmented, question (native_question and english_question), ground_truth (answer_english and answer_native), generated_answer, generated_explanation, correct_pattern_matching, correct_llm_as_judge, explanation_type
2. Compute the following statistics
    a. Overall accuracy => rows: QUES_<lang>_ANS_<EN/NATIVE>; columns: overall_accuracy
    b. Original vs Augmented accuracy => rows: QUES_<lang>_ANS_<EN/NATIVE>; columns: original_accuracy, augmented_accuracy
    c. Categories accuracy: => rows: QUES_<lang>_ANS_<EN/NATIVE>; columns: <Category_1>_accuracy, <Category_2>_accuracy, ...
    d. Explanation: count the number of explanation type in brackets [] => rows: QUES_<lang>_ANS_<EN/NATIVE>; columns: Functional_correct, Functional_wrong, Encyclopedic_correct, Encyclopedic_wrong, Causal_correct, Causal_wrong, Commonsense_correct, Commonsense_wrong

Notes:
- Running LLM as a judge after generating the csv

LLM as Judge Prompt - Claude - Sonnet 4.6:
'''
You are an expert in cuisines and cultural dishes from across the African continent,
with deep knowledge of regional ingredients, preparation methods, and cultural context.
You understand English, Haussau, Lingala, Twi and Yoruba.

You will be given a CSV file. Your task is to evaluate whether answer_native and generated_answer are semantically equivalent.

EVALUATION RULES:
- Compare meaning, not surface form. Diacritics, capitalization and minor spelling should not affect the verdict
- When the answer are not exact match, consult `generated_explanation` to determine if generated answer is conceptually correct
- A generated_answer that answers the inverse of what was asked (e.g. gives the local name when the English was expected, or vice versa) should be marked Partial, not Correct
- A refusal or evasive non-answer ("I cannot determine this") is always Wrong
- A generated_answer that is in the right conceptual neighborhood but missing a key qualifier (e.g. "Nyabugogo" vs "Nyabugogo Bus Station", or "Offal" vs "Offal peppersoup") is Partial
- A generated_answer that names a related but distinct item (e.g. moin-moin instead of akara, fufu instead of the cassava plant) is Wrong

Output: Return a CSV with exactly these columns, preserving original values for all the :
ID, augmented, answer_native, generated_answer, verdict, notes
The notes field should be one concise sentence explaining wrong or partial verdicts only. Leave it empty for Correct rows.

'''

Example output tree:
output/gpt-5.2/
└── TextOnly/
    ├── QUES_EN_ANS_EN/
    │   ├── 1.json
    │   └── 1_aug.json
    ├── QUES_EN_ANS_NATIVE/
    ├── QUES_HAU_ANS_EN/
    ├── QUES_HAU_ANS_NATIVE/
    └── ...

Usage:
    uv run python experiments/experiment_scoring.py --model-name gpt-4o --experiment-setup TextOnly --no-llm-judge
    uv run python experiments/experiment_scoring.py --model-name gpt-4o --experiment-setup TextOnly
    uv run python experiments/experiment_scoring.py --model-name gpt-4o --experiment-setup TextOnly --force
"""

import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_ROOT = Path(__file__).parent.parent
OUTPUT_DIR = _ROOT / "output"
EVAL_DIR = _ROOT / "evaluation"

LANGUAGES = ["english", "haussa", "lingala", "twi", "yoruba"]

LANG_ABBR = {
    "english": "EN",
    "haussa": "HAU",
    "lingala": "LIN",
    "twi": "TWI",
    "yoruba": "YOR",
}

ANS_TYPES = ["EN", "NATIVE"]

EXPLANATION_TYPES = ["Functional", "Encyclopedic", "Causal", "Commonsense"]

# Cheap/fast model used only for LLM-as-judge calls
JUDGE_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# JSON loading
# ---------------------------------------------------------------------------


def _extract_answer_english(data: dict) -> str | None:
    """Extract the English ground-truth answer from a JSON record.

    Field names differ across language configs:
      - QUES_EN:  correct_en
      - QUES_HAU: english_an
      - others:   answer_english
    """
    return (
        data.get("english_an")  # Hausa-specific
        or data.get("answer_english")  # Lingala / Twi / Yoruba
        or data.get("correct_en")  # English config
    )


def _extract_answer_native(data: dict) -> str | None:
    """Extract the native-language ground-truth answer from a JSON record.

    Field names differ across language configs:
      - QUES_EN:  correct_native
      - QUES_HAU: native_an
      - others:   answer_{language} or answer
    """
    if data.get("native_an"):
        return data["native_an"]
    if data.get("correct_native"):
        return data["correct_native"]
    # language-specific: answer_haussa, answer_lingala, answer_twi, answer_yoruba …
    for key, val in data.items():
        if key.startswith("answer_") and key not in ("answer_english",) and val:
            return val
    return data.get("answer")


def load_config_jsons(
    model_name: str, experiment_setup: str, lang_abbr: str, ans_type: str
) -> pd.DataFrame:
    """Load all JSON inference files for a given config into a DataFrame."""
    config_dir = OUTPUT_DIR / model_name / experiment_setup / f"QUES_{lang_abbr}_ANS_{ans_type}"
    if not config_dir.exists():
        return pd.DataFrame()

    rows = []
    for json_file in sorted(config_dir.glob("*.json")):
        try:
            data = json.loads(json_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        raw_response = data.get("response", {})
        if isinstance(raw_response, str):
            try:
                raw_response = json.loads(raw_response)
            except (json.JSONDecodeError, TypeError):
                raw_response = {"answer": raw_response, "explanation": ""}
        response = raw_response if isinstance(raw_response, dict) else {}

        answer_english = _extract_answer_english(data)
        answer_native = _extract_answer_native(data)
        ground_truth = answer_english if ans_type == "EN" else answer_native

        rows.append(
            {
                "ID": data.get("ID"),
                "Country": data.get("Country"),
                "Language": data.get("Language"),
                "Category": data.get("Category"),
                "augmented": data.get("augmented", False),
                "english_question": data.get("eng_question") or data.get("english_question"),
                "native_question": data.get("native_question"),
                "answer_english": answer_english,
                "answer_native": answer_native,
                "ground_truth": ground_truth,
                "generated_answer": response.get("answer", ""),
                "generated_explanation": response.get("explanation", ""),
            }
        )

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    """Lowercase, strip whitespace and leading/trailing punctuation."""
    return re.sub(r"^[^\w]+|[^\w]+$", "", str(text).lower().strip())


def score_pattern_matching(generated: str, ground_truth: str) -> bool:
    """Return True if generated answer matches ground truth via pattern matching."""
    gen = _normalize(generated)
    gt = _normalize(ground_truth)
    if not gen or not gt:
        return False
    if gt in gen or gen in gt:
        return True
    ratio = difflib.SequenceMatcher(None, gen, gt).ratio()
    return ratio >= 0.8


def score_llm_as_judge(client: OpenAI, generated: str, ground_truth: str) -> bool:
    """Use an LLM to judge whether generated answer is semantically correct."""
    prompt = (
        f"Is the following generated answer semantically equivalent to the ground truth answer?\n\n"
        f"Ground truth: {ground_truth}\n"
        f"Generated: {generated}\n\n"
        'Reply with JSON only, exactly: {"correct": true} or {"correct": false}'
    )
    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        text = response.choices[0].message.content.strip()
        # Strip markdown code fences if present
        text = re.sub(r"^```[a-z]*\n?|\n?```$", "", text, flags=re.MULTILINE).strip()
        result = json.loads(text)
        return bool(result.get("correct", False))
    except Exception:
        return False


def extract_explanation_type(explanation: str) -> str | None:
    """Extract the first bracketed explanation type, e.g. [Encyclopedic] -> 'Encyclopedic'."""
    match = re.search(r"\[([^\]]+)\]", str(explanation))
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Per-config scoring
# ---------------------------------------------------------------------------


def score_config(
    client: OpenAI | None,
    model_name: str,
    experiment_setup: str,
    lang_abbr: str,
    ans_type: str,
    force: bool = False,
) -> pd.DataFrame | None:
    """Score one config (lang × ans_type). Returns the scored DataFrame or None if skipped."""
    out_csv = EVAL_DIR / model_name / experiment_setup / f"QUES_{lang_abbr}_ANS_{ans_type}.csv"

    if out_csv.exists() and not force:
        print(f"  Skipping {out_csv.name} (already exists; use --force to recompute)")
        return pd.read_csv(out_csv)

    df = load_config_jsons(model_name, experiment_setup, lang_abbr, ans_type)
    if df.empty:
        print(f"  No JSON files found for QUES_{lang_abbr}_ANS_{ans_type} — skipping")
        return None

    print(f"  Scoring {len(df)} rows for QUES_{lang_abbr}_ANS_{ans_type}...")

    df["correct_pattern_matching"] = df.apply(
        lambda r: score_pattern_matching(r["generated_answer"], r["ground_truth"]), axis=1
    )

    if client is not None:
        judge_results = []
        for _, row in tqdm(
            df.iterrows(), total=len(df), desc=f"LLM judge QUES_{lang_abbr}_ANS_{ans_type}"
        ):
            judge_results.append(
                score_llm_as_judge(client, row["generated_answer"], row["ground_truth"])
            )
        df["correct_llm_as_judge"] = judge_results
    else:
        df["correct_llm_as_judge"] = None

    df["explanation_type"] = df["generated_explanation"].apply(extract_explanation_type)

    df = df[
        [
            "ID",
            "Country",
            "Language",
            "Category",
            "augmented",
            "english_question",
            "native_question",
            "answer_english",
            "answer_native",
            "ground_truth",
            "generated_answer",
            "generated_explanation",
            "correct_pattern_matching",
            "correct_llm_as_judge",
            "explanation_type",
        ]
    ]

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"  Saved -> {out_csv}")

    return df


# ---------------------------------------------------------------------------
# Merge all configs into a single long CSV
# ---------------------------------------------------------------------------


def merge_all_configs(
    eval_dfs: dict[str, pd.DataFrame], model_name: str, experiment_setup: str
) -> Path:
    """Concatenate all per-config DataFrames into one long CSV with a QUES_ANS_SETUP column."""
    frames = []
    for config_key, df in eval_dfs.items():
        tagged = df.copy()
        tagged.insert(0, "QUES_ANS_SETUP", config_key)
        frames.append(tagged)

    merged = pd.concat(frames, ignore_index=True)
    out_path = EVAL_DIR / model_name / f"{model_name}_{experiment_setup}_all.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    return out_path


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def _accuracy(series: pd.Series) -> float:
    """Mean of a boolean/nullable series, ignoring nulls."""
    valid = series.dropna()
    return float(valid.mean()) if len(valid) > 0 else float("nan")


def compute_stats(eval_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build a summary stats DataFrame across all scored configs."""
    rows = []
    all_categories: set[str] = set()
    for df in eval_dfs.values():
        all_categories.update(df["Category"].dropna().unique())
    sorted_categories = sorted(all_categories)

    for config_key, df in eval_dfs.items():
        use_llm = df["correct_llm_as_judge"].notna().any()
        score_col = "correct_llm_as_judge" if use_llm else "correct_pattern_matching"

        row: dict = {"config": config_key}

        # Overall accuracy
        row["overall_accuracy"] = _accuracy(df[score_col])

        # Original vs augmented
        orig = df[df["augmented"] == False]
        aug = df[df["augmented"] == True]
        row["original_accuracy"] = _accuracy(orig[score_col])
        row["augmented_accuracy"] = _accuracy(aug[score_col])

        # Per-category accuracy
        for cat in sorted_categories:
            cat_df = df[df["Category"] == cat]
            row[f"{cat}_accuracy"] = (
                _accuracy(cat_df[score_col]) if not cat_df.empty else float("nan")
            )

        # Explanation type breakdown
        for etype in EXPLANATION_TYPES:
            etype_df = df[df["explanation_type"] == etype]
            row[f"{etype}_correct"] = int(etype_df[score_col].fillna(False).sum())
            row[f"{etype}_wrong"] = int((~etype_df[score_col].fillna(False)).sum())

        rows.append(row)

    stats_df = pd.DataFrame(rows).set_index("config")
    return stats_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score AfriMCQA inference outputs and compute statistics."
    )
    parser.add_argument("--model-name", default="gpt-4o", help="Model name (default: gpt-4o)")
    parser.add_argument(
        "--experiment-setup", default="TextOnly", help="Experiment setup (default: TextOnly)"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=LANGUAGES + ["all"],
        default=["all"],
        help="Languages to score (default: all)",
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="Skip LLM-as-judge scoring (pattern matching only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute even if evaluation CSV already exists",
    )
    args = parser.parse_args()

    languages = LANGUAGES if "all" in args.languages else args.languages

    client: OpenAI | None = None
    if not args.no_llm_judge:
        load_dotenv()
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print(
                "ERROR: OPENAI_API_KEY is not set. Use --no-llm-judge to skip LLM scoring.",
                file=sys.stderr,
            )
            sys.exit(1)
        client = OpenAI(api_key=api_key)

    print(f"{'=' * 65}")
    print(f"Model          : {args.model_name}")
    print(f"Experiment     : {args.experiment_setup}")
    print(f"Languages      : {languages}")
    print(f"LLM judge      : {not args.no_llm_judge}")
    print(f"Force recompute: {args.force}")
    print(f"{'=' * 65}\n")

    eval_dfs: dict[str, pd.DataFrame] = {}

    for lang in languages:
        lang_abbr = LANG_ABBR[lang]
        print(f"\n--- {lang} ({lang_abbr}) ---")
        for ans_type in ANS_TYPES:
            df = score_config(
                client, args.model_name, args.experiment_setup, lang_abbr, ans_type, args.force
            )
            if df is not None:
                eval_dfs[f"QUES_{lang_abbr}_ANS_{ans_type}"] = df

    if not eval_dfs:
        print("\nNo configs scored — nothing to aggregate.")
        return

    print(f"\n{'=' * 65}")
    print("Merging all configs into long CSV...")
    merged_path = merge_all_configs(eval_dfs, args.model_name, args.experiment_setup)
    print(f"Merged CSV saved -> {merged_path}\n")

    print("Computing statistics...")
    stats_df = compute_stats(eval_dfs)

    stats_path = EVAL_DIR / args.model_name / args.experiment_setup / "stats.csv"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(stats_path)
    print(f"Stats saved -> {stats_path}\n")

    # Print summary to stdout
    summary_cols = ["overall_accuracy", "original_accuracy", "augmented_accuracy"]
    available = [c for c in summary_cols if c in stats_df.columns]
    print(stats_df[available].to_string(float_format="{:.3f}".format))


if __name__ == "__main__":
    main()
