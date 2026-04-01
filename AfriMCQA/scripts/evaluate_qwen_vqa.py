#!/usr/bin/env python3
"""
Qwen VQA Evaluation on Afri-MCQA

Runs inference for languages: Lingala, Akan_Twi, Hausa, Kinyarwanda.
For each sample, runs 3 prompt variants (English/local language combinations).
Uploads per-sample JSON results to a GCP bucket under:
    Qwen2-VL-2B-Instruct/afri-mcqa/{language}/{id}.json

Required env vars:
    GCS_BUCKET_NAME     - GCS bucket to upload results to
    MODEL_NAME          - (optional) Qwen model, default "Qwen/Qwen2-VL-2B-Instruct"

GCP auth: Application Default Credentials (run `gcloud auth application-default login`)

Usage:
> python3 evaluate_qwen_vqa.py --dry-run
> uv run python scripts/evaluate_qwen_vqa.py --dry-run
"""

import argparse
import json
import os
import re
import sys
import torch

from datasets import load_dataset
from dotenv import load_dotenv
from google.cloud import storage
from PIL import Image
from tqdm import tqdm

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info


EXPERIMENT = "H2"
LANGUAGES = ["Lingala", "Akan_Twi", "Hausa", "Kinyarwanda"]
COLUMNS = [
    "ID",
    "Country",
    "Language",
    "Category",
    "self_made",
    "eng_question",
    "native_question",
    "correct_en",
    "wrong_en_o1",
    "wrong_en_o2",
    "wrong_en_o3",
    "correct_native",
    "wrong_native_o1",
    "wrong_native_o2",
    "wrong_native_o3",
    "image",
]

MODEL_NAME = os.getenv("MODEL_NAME", "Qwen2-VL-2B-Instruct")
GCS_BUCKET_NAME = os.getenv("GCS_BUCKET_NAME")

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")

with open(os.path.join(_PROMPTS_DIR, "prompt_vqa_EN_only.txt"), encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read().strip()


# ---------------------------------------------------------------------------
# Prompt builder (verbatim from notebook)
# ---------------------------------------------------------------------------


def get_user_prompt(sample, prompt_lang: str, qa_lang: str, answer_lang: str) -> str:
    """
    LANG = ["English", ...]
    """
    user_prompt = f"""
    LANGUAGE: Respond in {answer_lang} only
    Question: {sample["eng_question"] if prompt_lang == "English" else sample["native_question"]}
    Choices:
    A. {sample["correct_en"] if qa_lang == "English" else sample["correct_native"]}
    B. {sample["wrong_en_o1"] if qa_lang == "English" else sample["wrong_native_o1"]}
    C. {sample["wrong_en_o2"] if qa_lang == "English" else sample["wrong_native_o2"]}
    D. {sample["wrong_en_o3"] if qa_lang == "English" else sample["wrong_native_o3"]}
    Answer:
    Short Explanation (30 words max):
    """
    return user_prompt


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------


def resize_image(img: Image.Image) -> Image.Image:
    """Resize a PIL image to (320, 240) and convert to RGB."""
    return img.convert("RGB").resize((320, 240))


# ---------------------------------------------------------------------------
# Qwen inference
# ---------------------------------------------------------------------------


def create_vqa_prompt(sample, prompt_lang: str, qa_lang: str, answer_lang: str):
    system_instruction = SYSTEM_PROMPT
    user_prompt = get_user_prompt(sample, prompt_lang, qa_lang, answer_lang)
    return system_instruction, user_prompt


def ask_vqa(
    model: Qwen2VLForConditionalGeneration,
    processor,
    image: Image.Image,
    system_instruction: str,
    user_prompt: str,
    device: str,
) -> str:
    """Run VQA inference with a PIL image and text prompt."""
    messages = [
        {"role": "system", "content": system_instruction},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    # ── preprocess ────────────────────────────────────────────────────────────────
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs if video_inputs else None,
        padding=True,
        return_tensors="pt",
    )

    # cast index tensors to Long, move everything to device
    inputs = {
        k: (
            v.long()
            if k in {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"}
            else v
        ).to(device)
        for k, v in inputs.items()
    }

    # ── generate ──────────────────────────────────────────────────────────────────
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=256)

    # strip the prompt tokens from the output
    trimmed = [out[len(inp) :] for inp, out in zip(inputs["input_ids"], generated_ids)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    return answer


# ---------------------------------------------------------------------------
# Response parser
# ---------------------------------------------------------------------------


def parse_response(text: str) -> dict:
    """
    Extract the answer letter (A/B/C/D) and explanation from the model's output.
    Looks for the first standalone A/B/C/D (possibly prefixed with 'Answer:').
    Returns {"answer": letter_or_null, "explanation": full_text}.
    """
    # Try to find a letter after "Answer:" label first
    match = re.search(r"Answer\s*[:\-]?\s*([A-D])\b", text, re.IGNORECASE)
    if not match:
        # Fall back to the first standalone letter at the start of a line or sentence
        match = re.search(r"(?:^|\n)\s*([A-D])\b", text)
    letter = match.group(1).upper() if match else None

    # Explanation: everything after the matched letter, or the full text if no letter found
    if match:
        explanation = text[match.end() :].strip().lstrip(".\n:-").strip()
        if not explanation:
            explanation = text.strip()
    else:
        explanation = text.strip()

    return {"answer": letter, "explanation": explanation}


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------


def gcs_blob_exists(bucket: storage.Bucket, path: str) -> bool:
    return bucket.blob(path).exists()


def upload_to_gcs(bucket: storage.Bucket, path: str, record: dict) -> None:
    blob = bucket.blob(path)
    blob.upload_from_string(
        json.dumps(record, ensure_ascii=False, indent=2),
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Qwen VQA evaluation on Afri-MCQA — uploads per-sample JSON to GCS"
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        choices=LANGUAGES + ["all"],
        default=["all"],
        help="Languages to evaluate (default: all)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print GCS paths without calling the API or uploading",
    )
    args = parser.parse_args()

    languages = LANGUAGES if "all" in args.languages else args.languages

    # load env variables
    load_dotenv()
    GCS_BUCKET_NAME = os.environ.get("GCS_BUCKET_NAME")

    if args.dry_run:
        model = processor = bucket = device = None
    else:
        if not GCS_BUCKET_NAME:
            print("ERROR: GCS_BUCKET_NAME env var is not set.", file=sys.stderr)
            sys.exit(1)

        gcs_client = storage.Client()
        bucket = gcs_client.bucket(GCS_BUCKET_NAME)

        # load model
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            f"Qwen/{MODEL_NAME}",
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        model.eval()
        processor = AutoProcessor.from_pretrained(f"Qwen/{MODEL_NAME}")

        print(f"{'=' * 70}")
        print(f"{device=}")
        print(f"GCS project: {gcs_client.project}")
        print(f"GCS bucket : {bucket.name} (exists={bucket.exists()})")
        print()

    print(f"Experiment : {EXPERIMENT}")
    print(f"Model      : {MODEL_NAME}")
    print(f"Languages  : {languages}")
    print(f"GCS bucket : {GCS_BUCKET_NAME or '<not set>'}")
    print(f"{'=' * 70}")
    print()

    for lang in languages:
        print(f"{'=' * 70}")
        print(f"Language: {lang}")
        print(f"{'=' * 70}")

        dataset = load_dataset(
            "Atnafu/Afri-MCQA",
            f"{lang}_dev",
            split="dev",
            columns=COLUMNS,
        )
        print(f"Loaded {len(dataset)} samples\n")

        skipped = 0
        processed = 0
        errors = 0

        for sample in tqdm(dataset, desc=lang):
            sample_id = sample["ID"]
            blob_path = f"{EXPERIMENT}/afri-mcqa/{lang.lower()}/{MODEL_NAME}/{sample_id}.json"

            if args.dry_run:
                print(f"  [dry run] gs://{GCS_BUCKET_NAME or '<bucket>'}/{blob_path}")
                continue

            if gcs_blob_exists(bucket, blob_path):
                skipped += 1
                continue

            try:
                image = resize_image(sample["image"])

                sys_en_en, up_en_en = create_vqa_prompt(sample, "English", "English", "English")
                sys_en_loc, up_en_loc = create_vqa_prompt(sample, "English", lang, "English")
                sys_loc_loc, up_loc_loc = create_vqa_prompt(sample, lang, lang, "English")

                r_en_en = parse_response(
                    ask_vqa(model, processor, image, sys_en_en, up_en_en, device)
                )
                r_en_loc = parse_response(
                    ask_vqa(model, processor, image, sys_en_loc, up_en_loc, device)
                )
                r_loc_loc = parse_response(
                    ask_vqa(model, processor, image, sys_loc_loc, up_loc_loc, device)
                )

                record = {
                    "id": sample_id,
                    **{col: sample[col] for col in COLUMNS if col != "image"},
                    "output_prompt_en_qa_en_answer_en": r_en_en,
                    "output_prompt_en_qa_loc_answer_en": r_en_loc,
                    "output_prompt_loc_qa_loc_answer_en": r_loc_loc,
                }
                print(record)

                upload_to_gcs(bucket, blob_path, record)
                processed += 1

            except Exception as e:
                print(f"\nERROR on sample {sample_id}: {e}")
                errors += 1
                continue

        if not args.dry_run:
            print(
                f"\nDone — processed: {processed}, skipped (exists): {skipped}, errors: {errors}\n"
            )


if __name__ == "__main__":
    main()
