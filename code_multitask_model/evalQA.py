import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
import pandas as pd
import json
from tqdm import tqdm
import re

# =======================
# Load Model & Tokenizer
# =======================
MODEL_PATH = "/data/horse/ws/xicu953f-LLM/trained_models_benchmarkdata1/full_finetune_final"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

# ✅ Ensure `[PAD]` is correctly set as the padding token
if tokenizer.pad_token is None or tokenizer.pad_token != "[PAD]":
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})

# Load model and resize token embeddings if needed
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
model.resize_token_embeddings(len(tokenizer))

# ✅ Explicitly set `[PAD]` token ID for padding
model.config.pad_token_id = tokenizer.pad_token_id

# Enable inference optimizations
model.eval()
model.config.use_cache = True  # Enable caching for faster response

# =======================
# Load Test Dataset (JSONL)
# =======================
test_jsonl_path = "/data/horse/ws/xicu953f-LLM/benchmarkdata/supervised_data_test_QA.jsonl"  # Adjust path if needed

test_data = []
with open(test_jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        test_data.append(json.loads(line))

print(f"Loaded {len(test_data)} test examples.")

# Extract inputs & references
instructions = [item["instruction"] for item in test_data]
inputs = [item["input"] for item in test_data]
references = [item["output"] for item in test_data]

# =======================
# Preprocessing Function (Same as Training)
# =======================
def preprocess_function(examples):
    """Prepares input in the same format as during training."""
    model_inputs = tokenizer(
        [f"{inst} {inp}" for inst, inp in zip(examples["instruction"], examples["input"])],
        max_length=512,
        padding="max_length",
        truncation=True
    )
    
    labels = tokenizer(
        examples["output"],
        max_length=512,
        padding="max_length",
        truncation=True
    )["input_ids"]

    # Replace pad token with -100 to ignore loss computation on padding
    labels = [[-100 if token == tokenizer.pad_token_id else token for token in label] for label in labels]
    model_inputs["labels"] = labels

    return model_inputs

# =======================
# Generate Predictions
# =======================
def clean_generated_text(generated_text, instruction, input_text):
    """
    Cleans the generated text by removing the repeated instruction and input text.
    Assumes the summary appears after the repetition.
    """
    if generated_text.startswith(instruction):
        generated_text = generated_text[len(instruction):].strip()

    # Remove input text if present
    
    generated_text = re.sub(r"Input:.*?\n\nOutput:\s*", "", prediction, flags=re.DOTALL).strip()

    return generated_text

def generate_response(instruction, input_text):
    """Generate text while penalizing repetitions."""
    prompt = f"{instruction}\n\nInput:\n{input_text}\n\nOutput:"
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
    
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=50,  # Limit the number of generated tokens
            pad_token_id=tokenizer.pad_token_id,
            repetition_penalty=1.2,  # Discourage repeating words
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
            temperature=0.7,  # Reduce randomness (lower = more focused)
            top_k=50,  # Limit to top 50 tokens per step
            top_p=0.9  # Nucleus sampling to improve diversity
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    # Clean the generated output
    cleaned_text = clean_generated_text(generated_text, instruction, input_text)

    return cleaned_text

predictions = []
for inst, inp in tqdm(zip(instructions, inputs), total=len(inputs), desc="Generating Responses"):
    prediction = generate_response(inst, inp)
    predictions.append(prediction)

# =======================
# Compute Metrics
# =======================
# Load Evaluation Metrics
em_metric = evaluate.load("exact_match")
f1_metric = evaluate.load("f1")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# Compute Exact Match (EM)
em_score = em_metric.compute(predictions=predictions, references=references)

# Compute F1 Score
#f1_score = f1_metric.compute(predictions=predictions, references=references)

# Compute ROUGE Score
rouge_scores = rouge.compute(predictions=predictions, references=references)

# Compute BERTScore
bert_scores = bertscore.compute(predictions=predictions, references=references, lang="en")


# =======================
# Print Evaluation Results
# =======================
print("\nEvaluation Results:")
print(f"Exact Match (EM): {em_score}")
#print(f"F1 Score: {f1_score}")
print(f"ROUGE Scores: {rouge_scores}")
print(f"BERTScore (F1): {bert_scores['f1']}")

# =======================
# Save Predictions to JSONL
# =======================
output_jsonl_path = "/home/xicu953f/LLM/eval_results/QApredictions.jsonl"

with open(output_jsonl_path, "w", encoding="utf-8") as f:
    for question, reference, prediction in zip(inputs, references, predictions):
        json.dump({
            "question": question,
            "reference": reference,
            "prediction": prediction
        }, f)
        f.write("\n")

print(f"Predictions saved to {output_jsonl_path}")

