import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

base_path = Path("./alpaca")

checkpoints = [
    p for p in base_path.glob("checkpoint-*")
    if (p / "adapter_config.json").exists()
]

adapter_path = max(
    checkpoints,
    key=lambda p: int(p.name.split("-")[-1])
)

config = PeftConfig.from_pretrained(adapter_path)
base_model_name = config.base_model_name_or_path

tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

prompt = "Below is an instruction"

inputs = tokenizer(prompt, return_tensors="pt").to(
    model.get_input_embeddings().weight.device
)

model.eval()
with torch.inference_mode():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print('------ LoRA ALPACA -------')
print("PROMPT: ", prompt)
print("ANSWER: ", tokenizer.decode(output[0], skip_special_tokens=True),)


# ---- 

prompt = "Below is an instruction"

inputs = tokenizer(prompt, return_tensors="pt").to(
    model.get_input_embeddings().weight.device
)

with model.disable_adapter(): # This allows us to get base model prediction.
    with torch.inference_mode():
        output_base = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

print('------ BASE MODEL -------')
print("PROMPT: ", prompt)
print("ANSWER: ", tokenizer.decode(output_base[0], skip_special_tokens=True))


