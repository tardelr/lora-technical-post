import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

base_path = Path("./json_adapter")

# Prefer final adapter saved at root; otherwise latest checkpoint
if (base_path / "adapter_config.json").exists():
    adapter_path = base_path
else:
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


# -------------------------------------------------------
# Prompt in the SAME FORMAT used during training
# -------------------------------------------------------

question = "What should you do if it begins raining while you are outside?"

choices = [
    "Go swimming",
    "Open an umbrella",
    "Take off your shoes",
    "Turn on the television",
]

LETTERS = "ABCDE"

choices_text = "\n".join(
    f"{letter}. {choice}"
    for letter, choice in zip(LETTERS, choices)
)

question_text = f"{question}\nChoices:\n{choices_text}"

prompt = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    f"### Instruction:\n{question_text}\n\n"
    "Respond ONLY with a valid JSON object in the format "
    '{"answer": "<response>", "explanation": "<short explanation>"}.\n\n'
    "### Response:\n"
)


inputs = tokenizer(
    prompt,
    return_tensors="pt",
).to(model.get_input_embeddings().weight.device)


# -------------------------------------------------------
# LoRA
# -------------------------------------------------------

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

# Only decode newly generated tokens
generated = output[0, inputs["input_ids"].shape[1]:]

print("------ LoRA JSON -------")
print("PROMPT:")
print(prompt)
print("\nANSWER:")
print(tokenizer.decode(generated, skip_special_tokens=True))


# -------------------------------------------------------
# Base model
# -------------------------------------------------------

with model.disable_adapter():
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

generated_base = output_base[0, inputs["input_ids"].shape[1]:]

print("\n------ BASE MODEL -------")
print("PROMPT:")
print(prompt)
print("\nANSWER:")
print(tokenizer.decode(generated_base, skip_special_tokens=True))