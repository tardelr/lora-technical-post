import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer


OUTPUT_DIR = "./cup_adapter"

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
data_files = "tardelr/worldcup2026"
dataset = load_dataset(data_files, split="train")


lora_config = LoraConfig(r=64,
  lora_alpha=128,
  target_modules="all-linear",
  task_type="CAUSAL_LM",
  bias="none")

trainer_kwargs = {
        "model": model,
        "args":SFTConfig(output_dir=OUTPUT_DIR, learning_rate=5e-5, num_train_epochs=4,
         logging_steps=100, bf16=True, fp16=False),
        "train_dataset": dataset,
        "peft_config": lora_config,
        "processing_class": tokenizer,
    }

trainer = SFTTrainer(**trainer_kwargs)
trainer.train(resume_from_checkpoint=False)
trainer.save_model(OUTPUT_DIR)
