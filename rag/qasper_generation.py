"""
Use the retrieval results generated from the Qasper dataset along with a QA-finetuned LLaMA 3.1 model for the RAG generation task.
The evaluation is conducted using the SciQA test set from the benchmark, which consists of approximately 400 QA pairs.
The evaluation metrics include ROUGE and F1 BERTScore.
"""
import torch
import pandas as pd
import random
import json
import unicodedata
import faiss
import numpy as np
from datasets import Dataset
from transformers import TextStreamer
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from tqdm import tqdm
import evaluate
from unsloth import FastLanguageModel

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

# ✅ Load Cross Encoder for re-ranking
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

FAISS_INDEX_PATH = "/content/drive/MyDrive/faiss_index2.bin"
EMBEDDING_PATH = "/content/drive/MyDrive/qa_evidence_embeddings2.npy"
PAPER_IDS_PATH = "/content/drive/MyDrive/paper_ids2.json"
EVIDENCE_PATH = "/content/drive/MyDrive/faiss_qedata_cleaned.json"

# **Load new QA dataset**
QA_DATA_PATH = "/content/drive/MyDrive/qa_data_cleaned.json"
with open(QA_DATA_PATH, "r") as f:
    qa_data = json.load(f)

# **Filter QA pairs with non-empty free_form_answer**
qa_filtered = [
    {"question": item["question"], "answer": item["free_form_answer"]}
    for item in qa_data if item["free_form_answer"].strip()
]

# **Randomly select 100 samples for evaluation**
eval_data = random.sample(qa_filtered, min(100, len(qa_filtered)))

# ✅ Load FAISS index
index = faiss.read_index(FAISS_INDEX_PATH)
embeddings = np.load(EMBEDDING_PATH)
with open(PAPER_IDS_PATH, "r") as f:
    paper_ids = json.load(f)

# ✅ Load BM25
with open(EVIDENCE_PATH, "r") as f:
    evidence_data = json.load(f)

tokenized_corpus = [item["evidence"].split() for item in evidence_data]
bm25 = BM25Okapi(tokenized_corpus)

# ------------------------------
# 3️⃣ **RAG: Combine FAISS + BM25 for retrieval**
# ------------------------------
def embed_text(text):
    """ Generate vector representation of the text """
    return embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

def hybrid_retrieve(query, k=20, alpha=0.5):
    """ Retrieve 50 candidates using FAISS + BM25 """
    query = unicodedata.normalize("NFKC", query)
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    dense_embedding = embed_text(query)
    _, faiss_indices = index.search(dense_embedding.reshape(1, -1), k)
    dense_scores = np.dot(dense_embedding, embeddings.T)
    hybrid_scores = alpha * bm25_scores + (1 - alpha) * dense_scores
    ranked_indices = np.argsort(hybrid_scores)[::-1][:k]
    return [evidence_data[i]["evidence"] for i in ranked_indices]

def rerank_with_cross_encoder(query, retrieved_evidence, top_k=5):
    """ Rerank the retrieved results using a Cross Encoder """
    pairs = [(query, evidence) for evidence in retrieved_evidence]
    scores = cross_encoder.predict(pairs)
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    return [retrieved_evidence[i] for i in sorted_indices]

# ------------------------------
# 4️⃣ **LLaMA generation component**
# ------------------------------
alpaca_prompt = """Below is an instruction that describes a task, paired with an input. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Retrieved Evidence:
{}

### Response:
{}

"""
EOS_TOKEN = tokenizer.eos_token

def generate_answer(query, use_cross_encoder=False):
    """
    Retrieve 50 candidate evidence using RAG,
    optionally re-rank with a Cross Encoder.
    """
    initial_evidence = hybrid_retrieve(query, k=20, alpha=0.5)

    if use_cross_encoder:
        top_evidence = rerank_with_cross_encoder(query, initial_evidence, top_k=5)
    else:
        top_evidence = initial_evidence[:5]

    retrieved_evidence = " ".join(top_evidence)
    prompt = alpaca_prompt.format("Answer the following question.", query, retrieved_evidence, "") + EOS_TOKEN
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to("cuda")
    output_ids = model.generate(**inputs, max_new_tokens=512)
    generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_text.split("### Response:\n")[-1].strip()

# ------------------------------
# 5️⃣ **Evaluate model generation results**
# ------------------------------
def evaluate_generation(use_cross_encoder=False):
    predictions = []
    references = [item["answer"] for item in eval_data]

    for q in tqdm([item["question"] for item in eval_data], desc="Generating Answers"):
        predictions.append(generate_answer(q, use_cross_encoder=use_cross_encoder))

    # Load evaluation metrics
    em_metric = evaluate.load("exact_match")
    f1_metric = evaluate.load("f1")
    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")
    bleu = evaluate.load("bleu")

    print("\n----REPORTING SCORES FOR RAG + LLaMA 3.1 8B INSTRUCT----")
    if predictions and references:
        em_score = em_metric.compute(predictions=predictions, references=references)
        print(f"Exact Match: {em_score['exact_match']:.2f}")
        rouge_scores = rouge.compute(predictions=predictions, references=references)
        print(f"ROUGE-1: {rouge_scores['rouge1']:.4f}")
        print(f"ROUGE-2: {rouge_scores['rouge2']:.4f}")
        print(f"ROUGE-L: {rouge_scores['rougeL']:.4f}")
        bleu_result = bleu.compute(predictions=predictions, references=references)
        print(f"BLEU Score: {bleu_result['bleu']:.2f}")
        bertscore_result = bertscore.compute(predictions=predictions, references=references, lang="en")
        print(f"BERTScore F1: {np.mean(bertscore_result['f1']):.2f}")

# Run evaluation
print("\nEvaluating Hybrid Retrieval...")
evaluate_generation(use_cross_encoder=False)

print("\nEvaluating Hybrid + Cross Encoder...")
evaluate_generation(use_cross_encoder=True)
