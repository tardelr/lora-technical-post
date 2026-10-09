import os
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


OUTPUT_DIR = "./alpaca_adapter/"

# Random tokeziner to start train, tested with Llama and Qwen
tokenizer = AutoTokenizer.from_pretrained(
    'Qwen/Qwen3-0.6B',
    use_fast=False,
)

model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B",
    dtype=torch.bfloat16, ## Forcing fp16, to show prepare_for_k_bit upcast
    device_map="auto",
    attn_implementation="sdpa",
    trust_remote_code=False,
)
data_files = "https://raw.githubusercontent.com/AGI-Edgerunners/LLM-Adapters/main/ft-training_set/commonsense_170k.json"
dataset = load_dataset("json", data_files=data_files, split="train")

eos_token = tokenizer.eos_token_id
pad_token=tokenizer.pad_token_id

def format_example(example, eos_token=eos_token):
    input_section = ""

    text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{example['instruction']}"
        f"{input_section}"
        f"\n\n### Response:\n{example['output']}"
        f"{tokenizer.eos_token}"
    )

    return {"text": text}


dataset = dataset.map(
    format_example,
    remove_columns=dataset.column_names,
    desc="Formatting dataset",
)

lora_config = LoraConfig(r=16,
  lora_alpha=32,
  target_modules="all-linear",
  task_type="CAUSAL_LM",
  bias="none")

trainer_kwargs = {
        "model": model,
        "args":SFTConfig(output_dir=OUTPUT_DIR, learning_rate=1e-4, max_steps=5000, num_train_epochs=1,
         logging_steps=100, bf16=True, fp16=False, per_device_train_batch_size=8,
        gradient_accumulation_steps=1
),
        "train_dataset": dataset,
        "peft_config": lora_config,
        "processing_class": tokenizer
    }

trainer = SFTTrainer(**trainer_kwargs)
trainer.train(resume_from_checkpoint=False)
trainer.save_model(OUTPUT_DIR)
