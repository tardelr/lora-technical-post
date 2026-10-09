import json
import torch
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftConfig, PeftModel


base_path = Path("./json_adapter/")

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
        batch = prompts[i:i + batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.get_input_embeddings().weight.device)

        if apply_lora:
            with torch.inference_mode():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=128,
                    do_sample=False,
                    temperature=0.1,
                    top_p=0.75,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

        else:
            with model.disable_adapter():
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=128,
                        do_sample=False,
                        temperature=0.1,
                        top_p=0.75,
                        pad_token_id=tokenizer.eos_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )

        input_length = inputs["input_ids"].shape[1]
        generated = outputs[:, input_length:]

        texts = tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )

        outputs_all.extend(texts)

    return outputs_all


## Same questions used by the Alpaca evaluation

questions = [
    "What is 17 × 24?",
    "Convert 3/8 to a decimal.",
    "What is 15% of 240?",
    "Solve for x: 3x + 7 = 22",
    "What is the area of a rectangle 8 cm long and 5 cm wide?",
    "What is the chemical symbol for gold?",
    'Who wrote "Pride and Prejudice"?',
    "What is the capital of Australia?",
    "How many bones are in the adult human body?",
    "What is the boiling point of water at sea level in Fahrenheit?",
    'Translate to Spanish: "The cat is sleeping on the chair."',
    'Rewrite in passive voice: "The chef cooked the meal."',
    'Convert to uppercase: "hello world"',
    'Reverse the word order: "one two three four"',
    'Convert "March 5, 2024" to ISO 8601 format.',
    'What is the sentiment of: "The battery died after two hours. Total waste of money."?',
    "Is a penguin a mammal, bird, or reptile?",
    "Is 91 prime or composite?",
    'What part of speech is "quickly" in "She quickly finished her homework"?',
    "Botanically, is a tomato a fruit or a vegetable?",
    'Extract the emails from: "Contact john@acme.com or sales@acme.com for details."',
    "What is the total for 2 coffees at $3.50 each and 1 muffin at $2.25?",
    'In "Maria flew from Lisbon to Tokyo last week," what is the destination city?',
    'How many words are in "The quick brown fox jumps over the lazy dog"?',
    "Which is the largest: 42, 17, 89, 3, 56?",
    "If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?",
    "Tom is taller than Sam, and Sam is taller than Ali. Who is shortest?",
    "A train leaves at 2:45 PM and the trip takes 1 h 50 min. When does it arrive?",
    "What comes next: 2, 6, 18, 54?",
    "If today is Wednesday, what day is it in 10 days?",
    "Write a Python function that returns the factorial of n.",
    'What does print(len("banana")) output?',
    "Fix the bug: def add(a, b): return a - b",
    "What does x = [1, 2, 3]; print(x[-1]) output?",
    "Write SQL to select all users older than 30.",
    "List the first three planets from the Sun, comma-separated, nothing else.",
    "Answer only yes or no: Is the Pacific the largest ocean?",
    'Return JSON with name and age from: "Alice is 30 years old."',
    'Summarize in one sentence: "Solar panels convert sunlight into electricity. Their cost has fallen sharply, and many households now install them."',
    "Write a haiku about the ocean.",
    'Correct the grammar: "She don\'t like apples."',
    'What is an antonym of "generous"?',
    'What is the plural of "mouse"?',
    'What is a synonym of "rapid"?',
    'What is the past tense of "run"?',
    "Who was the first person to walk on Mars?",
    "Summarize the following text:",
    "What is 10 divided by 0?",
    'How many times does "r" appear in "strawberry"?',
    'Translate "Good morning" to French.',
]

answers = [
    "408",
    ".375",
    "36",
    "x = 5",
    "40",
    "Au",
    "Austen",
    "Canberra",
    "206",
    "212",
    "silla",
    "was cooked by the chef",
    "HELLO WORLD",
    "four three two one",
    "2024-03-05",
    "egative",
    "ird",
    "omposite",
    "dverb",
    "ruit",
    "sales@acme.com",
    "9.25",
    "Tokyo",
    "9",
    "89",
    "Yes",
    "Ali",
    "4:35",
    "162",
    "Saturday",
    "def factorial",
    "6",
    "a + b",
    "3",
    "age > 30",
    "Mercury, Venus, Earth",
    "Yes",
    '"age": 30',
    "lectricity",
    "\n",
    "doesn't like",
    "tingy",
    "ice",
    "uick",
    "an",
    "o one",
    "text",
    "ndefined",
    "3",
    "Bonjour",
]


# INSTRUCTION = (
#     "Answer the multiple-choice question. Respond ONLY with a valid JSON object "
#     'in the format {"answer": "<letter>", "text": "<answer text>", "explanation": "<short explanation>"}.'
# )

def format_example(question):
    return (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        f"### Instruction:\n{question}\n\n"
        "Respond ONLY with a valid JSON object in the format "
        '{"answer": "<response>", "explanation": "<short explanation>"}.\n\n'
        "### Response:\n"
    )


formatted = list(map(format_example, questions))

outputs = generate_batch(formatted, apply_lora=True)

Path("outputs").mkdir(exist_ok=True)

with open("outputs/json_lora.jsonl", "w") as wf:
    for question, formatted_prompt, prediction, truth in zip(
        questions, formatted, outputs, answers
    ):
        record = {
            "question": question,
            "prompt": formatted_prompt,
            "prediction": prediction,
            "truth": truth,
        }

        wf.write(json.dumps(record) + "\n")


## Now for the Base version

outputs = generate_batch(formatted, apply_lora=False)

with open("outputs/json_base.jsonl", "w") as wf:
    for question, formatted_prompt, prediction, truth in zip(
        questions, formatted, outputs, answers
    ):
        record = {
            "question": question,
            "prompt": formatted_prompt,
            "prediction": prediction,
            "truth": truth,
        }

        wf.write(json.dumps(record) + "\n")