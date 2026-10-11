import json
import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel

base_path = Path("./cup_adapter/")

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

tokenizer.padding_side = "left"


def generate_batch(prompts, batch_size=8, model=model, apply_lora=True):
    model.eval()
    outputs_all = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(
                model.get_input_embeddings().weight.device)
        if apply_lora:
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=False,
                    temperature=0.1,
                    top_p=0.75,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
            )

        else:
            with model.disable_adapter(): # This allows us to get base model prediction.
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=32,
                        do_sample=False,
                        temperature=0.1,
                        top_p=0.75,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                )

        input_length = inputs["input_ids"].shape[1]
        generated = outputs[:, input_length:]
        
        texts = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True
        )

        outputs_all.extend(texts)
    return outputs_all

## adding the question/answer pair here to make the script faster to read


questions = [
    # --- "Name the stadium in <city>" (one stadium per city, so one valid answer) ---
    "Name the 2026 World Cup stadium in Seattle.",
    "Name the 2026 World Cup stadium in Mexico City.",
    "Name the 2026 World Cup stadium in Houston.",
    "Name the 2026 World Cup stadium in Boston.",
    "Name the 2026 World Cup stadium in Philadelphia.",
    "Name the 2026 World Cup stadium in Atlanta.",
    "Name the 2026 World Cup stadium in Vancouver.",
    "Name the 2026 World Cup stadium in Monterrey.",
    "Name the 2026 World Cup stadium in Guadalajara.",
    "Name the 2026 World Cup stadium in Toronto.",
    "Name the 2026 World Cup stadium in Kansas City.",
    "Name the 2026 World Cup stadium in Miami.",
    "Name the 2026 World Cup stadium in Arlington, Texas.",
    "Name the 2026 World Cup stadium in East Rutherford, New Jersey.",
    "Name the 2026 World Cup stadium in Inglewood, California.",
    "Name the 2026 World Cup stadium in Santa Clara, California.",
    # --- Stadium / tournament facts ---
    "Which country is Estadio BBVA in?",
    "Which country is BMO Field in?",
    "Canada, the United States and which other country co-hosted the 2026 World Cup?",
    "How many stadiums were used at the 2026 World Cup? Answer with a number.",
    "How many teams played in the 2026 World Cup? Answer with a number.",
    "How many matches were played at the 2026 World Cup? Answer with a number.",
    "How many 2026 World Cup stadiums are in Canada? Answer with a number.",
    "What is the capacity of AT&T Stadium at the 2026 World Cup?",
    "Which region of the 2026 World Cup host cities is SoFi Stadium in?",
    # --- Final and finishing positions ---
    "Who won the 2026 World Cup?",
    "Who was the runner-up at the 2026 World Cup?",
    "Which team beat France in the 2026 World Cup third-place match?",
    "What was the score of the 2026 World Cup final?",
    "Who scored the winning goal in the 2026 World Cup final?",
    # --- Group winners ---
    "Which team won Group A at the 2026 World Cup?",
    "Which team won Group B at the 2026 World Cup?",
    "Which team won Group C at the 2026 World Cup?",
    "Which team won Group F at the 2026 World Cup?",
    "Which team won Group K at the 2026 World Cup?",
    "Which team won Group L at the 2026 World Cup?",
    # --- Group last place ---
    "Which team finished last in Group E at the 2026 World Cup?",
    "Which team finished last in Group I at the 2026 World Cup?",
    "Which team finished last in Group B at the 2026 World Cup?",
    "Which team finished last in Group K at the 2026 World Cup?",
    "Which team finished last in Group L at the 2026 World Cup?",
    # --- Single match results ---
    "Which team beat Brazil in the 2026 World Cup Round of 16?",
    "Which team beat Mexico in the 2026 World Cup Round of 16?",
    "Which team beat the USA in the 2026 World Cup Round of 16?",
    "Which team beat Germany in the 2026 World Cup Round of 32?",
    "What was the score of Germany vs Curaçao at the 2026 World Cup?",
    "What was the score of Canada vs Qatar at the 2026 World Cup?",
    "Which Norway striker scored two goals against Senegal at the 2026 World Cup?",
    # --- Qualifying ---
    "Which team won UEFA play-off Path B for the 2026 World Cup?",
    "Which team won the Pathway 1 inter-confederation play-off final for the 2026 World Cup?",
]
 
answers = [
    "Lumen",         # Lumen Field
    "Azteca",        # Estadio Azteca
    "NRG",           # NRG Stadium
    "Gillette",      # Gillette Stadium (Foxborough)
    "Lincoln",       # Lincoln Financial Field
    "Mercedes",      # Mercedes-Benz Stadium
    "BC Place",
    "BBVA",          # Estadio BBVA (Guadalupe)
    "Akron",         # Estadio Akron (Zapopan)
    "BMO",           # BMO Field
    "Arrowhead",     # Arrowhead Stadium
    "Hard Rock",     # Hard Rock Stadium (Miami Gardens)
    "AT&T",          # AT&T Stadium
    "MetLife",       # MetLife Stadium
    "SoFi",          # SoFi Stadium
    "Levi",          # Levi's Stadium (apostrophe-safe)
    "Mexico",
    "Canada",
    "Mexico",
    "16",
    "48",
    "104",
    "2",             # BC Place + BMO Field
    "94",            # ~94,000
    "Western",
    "Spain",
    "Argentina",
    "England",       # England 6-4 France
    "1-0",           # Spain 1-0 Argentina after extra time
    "Torres",        # Ferran Torres 106'
    "Mexico",        # Group A
    "Switzerland",   # Group B
    "Brazil",        # Group C
    "Netherlands",   # Group F
    "Colombia",      # Group K
    "England",       # Group L
    "Cura",          # Group E last (accent-safe: Curaçao / Curacao)
    "Iraq",          # Group I last
    "Qatar",         # Group B last
    "Uzbekistan",    # Group K last
    "Panama",        # Group L last
    "Norway",        # R16: Norway 2-1 Brazil
    "England",       # R16: England 3-2 Mexico
    "Belgium",       # R16: Belgium 4-1 USA
    "Paraguay",      # R32: 1-1 aet, Paraguay won 4-3 on pens
    "7-1",
    "6-0",
    "Haaland",       # 48', 58'
    "Sweden",        # Path B final: Sweden 3-2 Poland
    "Congo",         # DR Congo 1-0 Jamaica aet
]


eos_token = tokenizer.eos_token_id
outputs = generate_batch(questions)

with open("outputs/cup_lora_specific.jsonl", "w") as wf:
    for question, prediction, truth in zip(
        questions, outputs, answers
    ):
        record = {
            "question": question,
            "prompt": question,
            "prediction": prediction,
            "truth": truth,
        }

        wf.write(json.dumps(record) + "\n")


## Now for the Base version
outputs = generate_batch(questions, apply_lora=False)

with open("outputs/cup_base_specific.jsonl", "w") as wf:
    for question, prediction, truth in zip(
        questions, outputs, answers
    ):
        record = {
            "question": question,
            "prompt": question,
            "prediction": prediction,
            "truth": truth,
        }

        wf.write(json.dumps(record) + "\n")


# print('------ BASE MODEL -------')
# print("PROMPT: ", prompt)
# print("ANSWER: ", tokenizer.decode(output_base[0], skip_special_tokens=True))


