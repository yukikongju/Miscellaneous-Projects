"""
Script to evaluate Aya Vision on AfriMCQA.

How the script works:
1. For each language, load the dataset based on EXPERIMENT_SETUP:
    => data/<EXPERIMENT_SETUP>/"<EXPERIMENT_SETUP> - <language>".csv
    a. Read CSV into AfriMCQADataset (torch Dataset), images loaded from data/images/
    b. Perform data augmentation (same pipeline as scripts/augment_data.py).
       Final dataset = original + augmented.
2. Model Inference: For each row in dataset
    a. Load system prompt from prompts/prompt_VQA_<language>_question.txt
    b. (1) Ask to answer in English
       => output/<MODEL_NAME>/<EXPERIMENT_SETUP>/QUES_<lang>_ANS_EN/<ID>.json
    c. (2) Ask to answer in native language
       => output/<MODEL_NAME>/<EXPERIMENT_SETUP>/QUES_<lang>_ANS_NATIVE/<ID>.json

Image behaviour per experiment setup:
  TextOnly  => text question only (no image sent to model)
  ImageOnly => image only (no text question)
  ImageText => image + text question

Example output tree:
output/aya-vision-8b/
└── TextOnly/
    ├── QUES_EN_ANS_EN/
    │   ├── 1.json
    │   └── 1_aug.json
    ├── QUES_EN_ANS_NATIVE/
    ├── QUES_HAU_ANS_EN/
    ├── QUES_HAU_ANS_NATIVE/
    └── ...

Usage:
    uv run python experiments/aya_evaluation.py --dry-run
    uv run python experiments/aya_evaluation.py --languages english --no-augment
    uv run python experiments/aya_evaluation.py --languages all
"""

import argparse
import json
import re
import sys
from pathlib import Path

import albumentations as A
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPERIMENT_SETUP = "ImageText"  # TextOnly, ImageOnly
MODEL_NAME = "aya-vision-8b"
MODEL_ID = "CohereLabs/aya-vision-8b"

LANGUAGES = ["english", "haussa", "lingala", "twi", "yoruba"]

_ROOT = Path(__file__).parent.parent
DATA_DIR = _ROOT / "data"
PROMPTS_DIR = _ROOT / "prompts"
OUTPUT_DIR = _ROOT / "output"

# Language abbreviations used in output directory names
LANG_ABBR = {
    "english": "EN",
    "haussa": "HAU",
    "lingala": "LIN",
    "twi": "TWI",
    "yoruba": "YOR",
}

# Prompt file per language (system prompt contains <OUTPUT_LANGUAGE> placeholder)
LANG_PROMPT_FILE = {
    "english": "prompt_VQA_EN_question.txt",
    "haussa": "prompt_VQA_Hausa_question.txt",
    "lingala": "prompt_VQA_Lingala_question.txt",
    "twi": "prompt_VQA_Twi_question.txt",
    "yoruba": "prompt_VQA_Yoruba_question.txt",
}

# Human-readable language name used in the <OUTPUT_LANGUAGE> substitution
LANG_DISPLAY = {
    "english": "English",
    "haussa": "Hausa",
    "lingala": "Lingala",
    "twi": "Twi",
    "yoruba": "Yoruba",
}

# ---------------------------------------------------------------------------
# Augmentation transform (mirrors scripts/augment_data.py)
# ---------------------------------------------------------------------------

AUGMENT_TRANSFORM = A.Compose(
    [
        # Geometric
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5),
        A.RandomCrop(height=224, width=224, p=0.3),
        # Photometric
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
    ]
)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class AfriMCQADataset(Dataset):
    """Torch Dataset over a CSV file with optional image augmentation."""

    def __init__(self, df: pd.DataFrame, images_dir: Path, augment: bool = False):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.augment = augment

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx].to_dict()

        img_path = self.images_dir / Path(row["image_path"]).name
        img = Image.open(img_path).convert("RGB")

        if self.augment:
            arr = np.array(img)
            arr = AUGMENT_TRANSFORM(image=arr)["image"]
            img = Image.fromarray(arr)

        row["image"] = img
        row["augmented"] = self.augment
        return row


def _resolve_image_path(raw_path: str, images_dir: Path) -> str:
    """Return the resolved image path (possibly with trailing _N suffix stripped).

    Example: images/2422659501_1_2.jpg -> images/2422659501_1.jpg
    """
    p = Path(raw_path)
    candidate = images_dir / p.name
    if candidate.exists():
        return raw_path

    # Strip the last _<digits> suffix from the stem and retry
    stem = p.stem  # e.g. "2422659501_1_2"
    new_stem = re.sub(r"_\d+$", "", stem)  # -> "2422659501_1"
    if new_stem != stem:
        fallback = images_dir / (new_stem + p.suffix)
        if fallback.exists():
            return str(p.parent / fallback.name)

    return raw_path  # unchanged; will be caught by the missing filter below


def load_datasets(language: str) -> tuple[AfriMCQADataset, AfriMCQADataset]:
    csv_path = DATA_DIR / EXPERIMENT_SETUP / f"{EXPERIMENT_SETUP} - {language}.csv"
    df = pd.read_csv(csv_path)
    images_dir = DATA_DIR / "images"

    # Resolve paths first (strip trailing _N suffix if the original doesn't exist)
    df["image_path"] = df["image_path"].apply(lambda p: _resolve_image_path(p, images_dir))

    missing_mask = df["image_path"].apply(lambda p: not (images_dir / Path(p).name).exists())
    missing_df = df[missing_mask]
    if not missing_df.empty:
        print(f"  Skipping {len(missing_df)} row(s) with missing images:")
        for _, row in missing_df.iterrows():
            print(f"    ID={row['ID']}  image_path={row['image_path']}")
    df = df[~missing_mask].reset_index(drop=True)

    return AfriMCQADataset(df, images_dir, augment=False), AfriMCQADataset(
        df, images_dir, augment=True
    )


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def load_system_prompt(language: str, output_language: str) -> str:
    path = PROMPTS_DIR / LANG_PROMPT_FILE[language]
    text = path.read_text(encoding="utf-8").strip()
    return text.replace("<OUTPUT_LANGUAGE>", output_language)


def build_user_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def should_include_image(experiment_setup: str) -> bool:
    return experiment_setup.lower() in ("imageonly", "imagetext")


# ---------------------------------------------------------------------------
# Aya Vision inference
# ---------------------------------------------------------------------------


def ask_aya(
    processor: AutoProcessor,
    model: AutoModelForImageTextToText,
    device: str,
    system_prompt: str,
    user_prompt: str,
    image: Image.Image | None = None,
) -> str:
    user_content: list[dict] = []
    if image is not None:
        user_content.append({"type": "image", "image": image})
    user_content.append({"type": "text", "text": user_prompt})

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=300)

    # Slice off the prompt tokens to get only the generated text
    generated = output[0][inputs["input_ids"].shape[-1] :]
    return processor.decode(generated, skip_special_tokens=True).strip()


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def parse_response(text: str) -> dict:
    """Parse model output — expects JSON, falls back to regex extraction."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to extract a JSON object from within the text
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Last resort: treat entire text as the answer
    return {"answer": text, "explanation": ""}


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def output_path(out_dir: Path, sample_id, augmented: bool) -> Path:
    suffix = "_aug" if augmented else ""
    return out_dir / f"{sample_id}{suffix}.json"


def save_result(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aya Vision evaluation on AfriMCQA — saves per-sample JSON locally"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=LANGUAGES + ["all"],
        default=["all"],
        help="Languages to evaluate (default: all)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print output paths without running inference"
    )
    parser.add_argument("--no-augment", action="store_true", help="Skip augmented dataset")
    parser.add_argument(
        "--device",
        default="mps",
        choices=["mps", "cuda", "cpu"],
        help="Device to run the model on (default: mps)",
    )
    args = parser.parse_args()

    languages = LANGUAGES if "all" in args.languages else args.languages
    use_image = should_include_image(EXPERIMENT_SETUP)

    print(f"{'=' * 65}")
    print(f"Model          : {MODEL_NAME}")
    print(f"Experiment     : {EXPERIMENT_SETUP}")
    print(f"Include image  : {use_image}")
    print(f"Languages      : {languages}")
    print(f"Augment        : {not args.no_augment}")
    print(f"Device         : {args.device}")
    print(f"{'=' * 65}\n")

    if not args.dry_run:
        print(f"Loading model {MODEL_ID} on {args.device}...")
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to(
            args.device
        )
        model.eval()
        print("Model loaded.\n")
    else:
        processor = model = None

    for lang in languages:
        print(f"\n{'=' * 65}")
        print(f"Language: {lang}")
        print(f"{'=' * 65}")

        original_ds, augmented_ds = load_datasets(lang)
        print(f"Loaded {len(original_ds)} samples")

        lang_abbr = LANG_ABBR[lang]
        lang_display = LANG_DISPLAY[lang]

        sys_en = load_system_prompt(lang, "English")
        sys_native = load_system_prompt(lang, lang_display)

        out_en = OUTPUT_DIR / MODEL_NAME / EXPERIMENT_SETUP / f"QUES_{lang_abbr}_ANS_EN"
        out_native = OUTPUT_DIR / MODEL_NAME / EXPERIMENT_SETUP / f"QUES_{lang_abbr}_ANS_NATIVE"

        question_col = "eng_question" if lang == "english" else "native_question"

        datasets_to_run = [original_ds]
        if not args.no_augment:
            datasets_to_run.append(augmented_ds)

        processed = skipped = errors = 0

        for ds in datasets_to_run:
            label = "aug" if ds.augment else "orig"
            for sample in tqdm(ds, desc=f"{lang}/{label}"):
                sample_id = sample["ID"]
                path_en = output_path(out_en, sample_id, ds.augment)
                path_native = output_path(out_native, sample_id, ds.augment)

                if args.dry_run:
                    print(f"  [dry run] {path_en}")
                    print(f"  [dry run] {path_native}")
                    continue

                if path_en.exists() and path_native.exists():
                    skipped += 1
                    continue

                try:
                    image = sample["image"] if use_image else None
                    question = sample[question_col]
                    user_prompt = build_user_prompt(question)

                    meta = {k: v for k, v in sample.items() if k != "image"}

                    if not path_en.exists():
                        r_en = parse_response(
                            ask_aya(processor, model, args.device, sys_en, user_prompt, image)
                        )
                        save_result(path_en, {"id": sample_id, **meta, "response": r_en})

                    if not path_native.exists():
                        r_native = parse_response(
                            ask_aya(processor, model, args.device, sys_native, user_prompt, image)
                        )
                        save_result(path_native, {"id": sample_id, **meta, "response": r_native})

                    processed += 1

                except Exception as e:
                    print(f"\nERROR on sample {sample_id}: {e}")
                    errors += 1

        if not args.dry_run:
            print(f"\nDone — processed: {processed}, skipped: {skipped}, errors: {errors}")


if __name__ == "__main__":
    main()
