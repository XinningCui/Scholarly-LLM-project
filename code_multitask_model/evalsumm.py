import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
import pandas as pd
import json
import numpy as np
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
test_jsonl_path = "/data/horse/ws/xicu953f-LLM/benchmarkdata/supervised_data_test_summari.jsonl"

test_data = []
with open(test_jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        test_data.append(json.loads(line))

print(f"Loaded {len(test_data)} test examples.")

# Extract inputs & references
instructions = [item["instruction"] for item in test_data]
sources = [item["input"] for item in test_data]  # Original text (before simplification)
references = [[item["output"]] for item in test_data]  # Ensure references are lists



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
            max_new_tokens=400,  # Limit the number of generated tokens
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



# Generate cleaned predictions
cleaned_predictions = []
for inst, inp in tqdm(zip(instructions, sources), total=len(sources), desc="Generating Responses"):
    prediction = generate_response(inst, inp)
    cleaned_predictions.append(prediction)



# =======================
# Compute Metrics
# =======================
# Load Evaluation Metrics

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")
# Compute evaluation metrics using cleaned predictions
rouge_scores = rouge.compute(predictions=cleaned_predictions, references=[ref[0] for ref in references])
bert_scores = bertscore.compute(predictions=cleaned_predictions, references=[ref[0] for ref in references], lang="en")
bleu_score = bleu.compute(predictions=cleaned_predictions, references=references)


# =======================
# Print Evaluation Results
# =======================
print("\nEvaluation Results:")
print(f"ROUGE Scores: {rouge_scores}")
print(f"BERTScore (F1): {np.mean(bert_scores['f1'])}")  # Average BERT F1 score
print(f"ROUGE-1: {rouge_scores['rouge1']}")
print(f"ROUGE-2: {rouge_scores['rouge2']}")
print(f"ROUGE-L: {rouge_scores['rougeL']}")
print(f"BLEU Score: {bleu_score}")

# =======================
# Save Predictions to JSONL
# =======================
output_jsonl_path = "/home/yish134f/LLM/predictions_summarization.jsonl"

with open(output_jsonl_path, "w", encoding="utf-8") as f:
    for source, reference, prediction in zip(sources, references, cleaned_predictions):  # Use cleaned_predictions
        json.dump({
            "source": source,
            "reference": reference[0],  # Unwrapping the single reference
            "prediction": prediction
        }, f)
        f.write("\n")

print(f"Predictions saved to {output_jsonl_path}")



