# uv add transformers qwen-vl-utils torch torchvision pillow requests accelerate
# uv run python qwen2_vqa.py
# RuntimeError: Invalid buffer size: 230.66 GiB on MPS Mac M1 => if image is not resized
# Image resize to (320, 240)

import os
import requests
import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
from io import BytesIO
from PIL import Image


# ── device ────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ── model + processor ─────────────────────────────────────────────────────────
MODEL_ID = "Qwen/Qwen2-VL-2B-Instruct"  # use the Instruct variant — it has the chat template

model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # bfloat16 is unreliable on MPS, torch.float32
    #  torch_dtype=torch.float32,            # bfloat16 is unreliable on MPS, torch.float32
    device_map={"": device},  # load directly onto MPS/CPU
)
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)

# ── inputs ────────────────────────────────────────────────────────────────────
IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"
)
response = requests.get(IMAGE_URL)
img = Image.open(BytesIO(response.content))  # (4032, 3024)
img = img.resize((320, 240))
print(img.size)
#  img.show()

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "..", "prompts")
with open(os.path.join(_PROMPTS_DIR, "prompt_vqa_EN_only.txt"), encoding="utf-8") as _f:
    SYSTEM_PROMPT = _f.read().strip()

QUESTION = "What animal is on the candy? Give your answer and the explanation in 50 words"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": QUESTION},
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
print(f"Q: {QUESTION}")
print(f"A: {answer}")
