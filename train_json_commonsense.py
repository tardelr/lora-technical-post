import json
import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


OUTPUT_DIR = "./json_adapter"

tokenizer = AutoTokenizer.from_pretrained(
    'Qwen/Qwen3-0.6B',
    use_fast=False,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=False,
)

# this dataset is basically commonsense with additional explanation fields
dataset = load_dataset("Salesforce/cos_e", "v1.11", split="train")

LETTERS = "ABCDE"
def format_example(example):
    texts = example["choices"]
    labels = LETTERS[: len(texts)]
    choices = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
    question = f"{example['question']}\nChoices:\n{choices}"

    prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{question}\n\n"
        "Respond ONLY with a valid JSON object in the format "
        '{"answer": "<response>", "explanation": "<short explanation>"}.\n\n'
        "### Response:\n"
    )
    completion = json.dumps(
        {"answer": example["answer"], "explanation": example["abstractive_explanation"]},
        ensure_ascii=False,
    )+  tokenizer.eos_token
    return {"prompt": prompt, "completion": completion}


dataset = dataset.map(format_example, remove_columns=dataset.column_names)


# --- check if data format looks fine
for i in range(3):
    print(f"\n--- SAMPLE {i} ---")
    print("PROMPT:")
    print(dataset[i]["prompt"])
    print("\nCOMPLETION:")
    print(dataset[i]["completion"])



lora_config = LoraConfig(r=64,
  lora_alpha=128,
  target_modules="all-linear",
  task_type="CAUSAL_LM",
  bias="none")

trainer_kwargs = {
        "model": model,
        "args":SFTConfig(output_dir=OUTPUT_DIR, learning_rate=5e-5, num_train_epochs=3,
         logging_steps=100, bf16=True, fp16=False, completion_only_loss=True, eos_token=tokenizer.eos_token),
        "train_dataset": dataset,
        "peft_config": lora_config,
        "processing_class": tokenizer,
    }

trainer = SFTTrainer(**trainer_kwargs)
trainer.train(resume_from_checkpoint=False)
trainer.save_model(OUTPUT_DIR)
