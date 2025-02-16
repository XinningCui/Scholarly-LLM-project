"""
Use the retrieval results generated from the SciQA_train dataset along with a QA-finetuned LLaMA 3.1 model for the RAG generation task.
The evaluation is conducted using the SciQA test set from the benchmark, which consists of approximately 400 QA pairs.
The evaluation metrics include ROUGE and F1 BERTScore.
Code from: https://huggingface.co/datasets/orkg/SciQA
"""
from unsloth import FastLanguageModel
import torch
import pandas as pd
import random
from datasets import Dataset
import json
from transformers import TextStreamer
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import unicodedata
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import evaluate
import re

# ------------------------------
# 1️⃣ **Load LLaMA fine-tuned model**
# ------------------------------
max_seq_length = 32000
dtype = None
load_in_4bit = True

model_path = "/content/drive/MyDrive/checkpoint-60"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_path,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
)

FastLanguageModel.for_inference(model)  # ✅ Accelerate inference

# ------------------------------
# 2️⃣ **Load RAG retrieval model**
# ------------------------------
specter_model = "sentence-transformers/allenai-specter"
embedding_model = SentenceTransformer(specter_model)

FAISS_INDEX_PATH = "/content/drive/MyDrive/faiss_with_links1.bin"
BM25_PATH = "/content/drive/MyDrive/bm25_with_links1.json"
QA_DATA_PATH = "/content/drive/MyDrive/cleaned_train.csv"

# **Load data**
df = pd.read_csv(QA_DATA_PATH)

def remove_links(text):
    """ Remove URLs from the answers """
    return re.sub(r'http[s]?://\S+', '', str(text)).strip() or None

df["answer_cleaned"] = df["answer"].apply(remove_links)

qa_pairs_with_links = [
    {"question": unicodedata.normalize("NFKC", row["question"].strip()), "answer": row["answer"]}
    for _, row in df.iterrows()
]

qa_pairs_without_links = [
    {"question": unicodedata.normalize("NFKC", row["question"].strip()), "answer": row["answer_cleaned"]}
    for _, row in df.iterrows()
    if row["answer_cleaned"]
]

# ✅ Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)
with open(BM25_PATH, "r") as f:
    bm25_texts = json.load(f)
    bm25 = BM25Okapi([text.split() for text in bm25_texts])

# ------------------------------
# 3️⃣ **Hybrid (FAISS + BM25) retrieval**
# ------------------------------
def embed_text(text):
    """ Generate vector representation of the text """
    return embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

def hybrid_retrieve(query, k=5, alpha=0.5):
    """ Perform retrieval using FAISS + BM25 """
    query = unicodedata.normalize("NFKC", query)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    dense_embedding = embed_text(query)
    _, faiss_indices = index.search(dense_embedding.reshape(1, -1), k)

    dense_scores = np.dot(dense_embedding, np.array([embed_text(text) for text in bm25_texts]).T)

    hybrid_scores = alpha * bm25_scores + (1 - alpha) * dense_scores
    ranked_indices = np.argsort(hybrid_scores)[::-1][:k]

    return [bm25_texts[i] for i in ranked_indices]

# ------------------------------
# 4️⃣ **LLaMA generation component**
# ------------------------------
test_dataset = pd.read_csv("/content/drive/MyDrive/cleaned_test.csv")

alpaca_prompt = """You are a helpful AI assistant. Answer the user's question based on the retrieved evidence.

### Question:
{}

### Retrieved Evidence:
{}

### Answer:
"""

EOS_TOKEN = tokenizer.eos_token

def generate_answer(query):
    """ Generate answers using Hybrid retrieval without Cross Encoder """
    retrieved_evidence = hybrid_retrieve(query, k=5, alpha=0.5)
    retrieved_evidence = " ".join(retrieved_evidence)

    prompt = alpaca_prompt.format(query, retrieved_evidence)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=512)

    return tokenizer.decode(output_ids[0], skip_special_tokens=True).split("### Answer:\n")[-1].strip()

# ------------------------------
# 5️⃣ **Evaluate model generation results**
# ------------------------------
predictions = []
references = test_dataset["answer"].tolist()

for q in tqdm(test_dataset["question"].tolist(), desc="Generating Answers"):
    predictions.append(generate_answer(q))

em_metric = evaluate.load("exact_match")
f1_metric = evaluate.load("f1")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")
bleu = evaluate.load("bleu")

print("----REPORTING SCORES FOR RAG + LLaMA 3.1 8B INSTRUCT----")

if predictions and references:
    em_score = em_metric.compute(predictions=predictions, references=references)
    print(f"Exact Match: {em_score['exact_match']:.2f}")
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
    print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
    print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
    bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")
    print(f"BERTScore F1: {sum(bertscore_result['f1']) / len(bertscore_result['f1']):.2f}")
    bleu_result = bleu.compute(predictions=predictions, references=references)
    print(f"BLEU Score: {bleu_result['bleu']:.2f}")
    f1_result = f1_metric.compute(predictions=predictions, references=references, average="weighted")
    print(f"F1 Score: {f1_result['f1']:.2f}")
else:
    print("❌ Error: No valid predictions or references found for evaluation!")

print("✅ RAG + LLaMA evaluation completed!")
