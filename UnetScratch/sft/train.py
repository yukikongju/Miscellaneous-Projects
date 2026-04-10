"""
Fine-Tuning LLM with Supervised Fine-Tuning

"""

import os
import torch

from datasets import load_dataset
from huggingface_hub import login
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from trl import SFTConfig, SFTTrainer

# from dotenv import load_dotenv

from prompt_instruction import prompt_instruct

# Environment Variables
# load_dotenv()
# HUGGINGFACE_TOKEN = os.env.getenv("HUGGINGFACE_TOKEN")
tags_dict = {
    "np": "nutrition_panel",
    "il": "ingredient_list",
    "me": "menu",
    "re": "recipe",
    "fi": "food_items",
    "di": "drink_items",
    "fa": "food_advertistment",
    "fp": "food_packaging",
}
MODEL_NAME = "google/gemma-3-270m-it"
if torch.cuda.is_available():
    device = "cuda"
elif torch.mps.is_available():
    device = "mps"
else:
    device = "cpu"

CHECKPOINT_DIR_NAME = "./checkpoint_models"
BASE_LEARNING_RATE = 5e-5
BATCH_SIZE = 4
NUM_TRAINING_EPOCHS = 1


# load dataset
def sample_to_conversation(sample):
    return {
        "prompt": [{"role": "user", "content": sample["sequence"]}],  # input
        "completion": [
            {"role": "assistant", "content": sample["gpt-oss-120b-label-condensed"]}  # ground truth
        ],
    }


dataset = load_dataset("mrdbourke/FoodExtract-1k")
print(f"[INFO] Number of samples in the dataset: {len(dataset['train'])}")
dataset = dataset.map(sample_to_conversation, batched=False)
dataset = dataset["train"].train_test_split(test_size=0.2, shuffle=False, seed=42)


# load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    # dtype="auto",
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="sdpa",  # or 'eager'
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print(f"[INFO - TEMPLATE] {tokenizer.chat_template}")
print(f"[INFO] Model on device: {model.device}")
print(f"[INFO] Model using dtype: {model.dtype}")


def get_model_num_params(model: AutoModelForCausalLM):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    total_params = trainable_params + non_trainable_params
    return {
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "total_params": total_params,
    }


model_params = get_model_num_params(model)
print(f"Trainable parameters: {model_params['trainable_params']:,}")
print(f"Non-trainable parameters: {model_params['non_trainable_params']:,}")
print(f"Total parameters: {model_params['total_params']:,}")


#
def get_inference_naive(
    query: str, model: AutoModelForCausalLM, tokenizer: AutoTokenizer, device: str
):
    tensor = torch.tensor(tokenizer(query)["input_ids"]).unsqueeze(0).to(device)
    outputs = model(tensor)
    predicted_ids = outputs.logits.argmax(dim=-1)
    # predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_ids[0])
    predicted_text = tokenizer.decode(predicted_ids[0])
    return predicted_text


def get_inference1(query: str, pipe: pipeline):
    sample = {"role": "user", "content": query}
    input_prompt = pipe.tokenizer.apply_chat_template(
        [sample], tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(input_prompt, max_new_tokens=512, disable_compile=True)
    predicted_text = outputs[0]["generated_text"][len(input_prompt) :]
    return predicted_text


def get_inference2(query: str, pipe: pipeline, prompt_instruct: str):
    new_query = prompt_instruct.replace("<targ_input_text>", query)
    sample = {"role": "user", "content": new_query}
    input_prompt = pipe.tokenizer.apply_chat_template(
        [sample], tokenize=False, add_generation_prompt=True
    )
    outputs = pipe(input_prompt, max_new_tokens=512, disable_compile=True)
    predicted_text = outputs[0]["generated_text"][len(input_prompt) :]
    return predicted_text


# Inference 1 - Can the model understand language?
query = "Can you generate 10 best title for a mix with the following sounds: frogs, oscillating fan, ocean"
output_query_naive = get_inference_naive(query, model, tokenizer, device)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
output_query = get_inference1(query, pipe)

print("-" * 70)
print(f"[QUERY] {query}")
print(f"[INFERENCE - NAIVE] {output_query_naive}")
print(f"[INFERENCE - TEMPLATE] {output_query}")
print("-" * 70)
print()

# Inference 2 - Add prompt to the model - few shots learning
print("-" * 70)
item = dataset["train"][0]
query = item["sequence"]
ground_truth = item["gpt-oss-120b-label-condensed"]
print(f"[QUERY] {query}")
print(f"[GROUND TRUTH] {ground_truth}")
print(f"[INFERENCE - NAIVE] {get_inference_naive(query, model, tokenizer, device)}")
print(f"[INFERENCE - TEMPLATE] {get_inference1(query, pipe)}")
print(f"[INFERENCE - PROMPT] {get_inference2(query, pipe, prompt_instruct)}")
print("-" * 70)
print()

# Fine-Tuning 1 - Fine-Tuning the model with SFT
sft_config = SFTConfig(
    output_dir=CHECKPOINT_DIR_NAME,
    max_length=512,
    packing=False,
    num_train_epochs=NUM_TRAINING_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,  # Note: you can change this depending on the amount of VRAM your GPU has
    per_device_eval_batch_size=BATCH_SIZE,
    completion_only_loss=True,  # we want our model to only learn how to *complete* / generate the output tokens given the input tokens
    gradient_checkpointing=False,
    optim="adamw_torch_fused",  # Note: if you try "adamw", you will get an error
    logging_steps=1,
    save_strategy="epoch",  # Save our model every epoch
    eval_strategy="epoch",  # Evaluate our model every epoch
    learning_rate=BASE_LEARNING_RATE,
    fp16=(model.dtype == torch.float16),
    bf16=(model.dtype == torch.bfloat16),
    load_best_model_at_end=True,
    metric_for_best_model="mean_token_accuracy",
    greater_is_better=True,
    lr_scheduler_type="constant",
    push_to_hub=False,
    report_to="none",  # Optionally save our models training metrics to a logging service
)
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    processing_class=tokenizer,
)
training_ouput = trainer.train()
print(f"[INFO] Eval metrics: {trainer.evaluate()}")
print(f"[INFO] Our model's mean token accuracy: {trainer.state.best_metric*100:.2f}%")


# check model loss curves
def plot_loss_curves(trainer: SFTTrainer):
    log_history = trainer.state.log_history

    train_losses = [log["loss"] for log in log_history if "loss" in log]
    epoch_train = [log["epoch"] for log in log_history if "loss" in log]
    eval_losses = [log["eval_loss"] for log in log_history if "eval_loss" in log]
    epoch_eval = [log["epoch"] for log in log_history if "eval_loss" in log]

    plt.plot(epoch_train, train_losses, label="Training Loss")
    plt.plot(epoch_eval, eval_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss per Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_loss_curves(trainer)

# save model
trainer.save_model()
