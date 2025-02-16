"""
Use approximately 1,500 (question, answer) pairs from the SciQA dataset for retrieval evaluation.
First, embed the questions and answers using the SPECTER model.
Then, apply hybrid retrieval (FAISS + BM25) combined with a cross-encoder to obtain the retrieval results.
Evaluate the retrieval performance using Recall@K and MRR as metrics.
Code from: https://huggingface.co/datasets/orkg/SciQA
"""
import re
import unicodedata
import pandas as pd
import faiss
import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import os
import random

# **1️⃣ Load Data**
CSV_PATH = "/content/drive/MyDrive/cleaned_train.csv"  # **Update with your data path**
df = pd.read_csv(CSV_PATH)

# **2️⃣ Define Text Cleaning Functions**
def normalize_unicode(text):
    """ Normalize Unicode and remove URLs, keep only text """
    text = unicodedata.normalize("NFKC", str(text)).strip()
    return text if text else None

def remove_links(text):
    """ Remove URLs from text """
    cleaned_text = re.sub(r'http[s]?://\S+', '', text).strip()
    return cleaned_text if cleaned_text else None  # **Return None if empty after cleaning**

# **3️⃣ Process Data**
df["answer_cleaned"] = df["answer"].apply(remove_links)

# **Construct two versions of QA**
qa_pairs_with_links = [
    {"question": normalize_unicode(row["question"]), "answer": normalize_unicode(row["answer"])}
    for _, row in df.iterrows()
]

# qa_pairs_without_links = [
#     {"question": normalize_unicode(row["question"]), "answer": remove_links(row["answer"])}
#     for _, row in df.iterrows()
#     if remove_links(row["answer"])  # **Exclude empty answers**
# ]

print(f"📌 Number of QA pairs with links: {len(qa_pairs_with_links)}")
# print(f"📌 Number of QA pairs without links: {len(qa_pairs_without_links)}")

# **4️⃣ Load SPECTER sentence embedding model**
specter_model = "sentence-transformers/allenai-specter"
embedding_model = SentenceTransformer(specter_model)

def embed_text(text):
    """ Compute text embeddings """
    return embedding_model.encode(text, convert_to_numpy=True, normalize_embeddings=True)

# **5️⃣ Generate FAISS & BM25 Indexes**
def build_faiss_and_bm25(qa_pairs, faiss_path, bm25_path):
    """ Generate FAISS & BM25 indexes and save to files """
    texts = [f"Q: {item['question']} A: {item['answer']}" for item in qa_pairs]
    embeddings = np.array([embed_text(text) for text in texts], dtype="float32")

    # **Save FAISS index**
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, faiss_path)
    print(f"✅ FAISS index saved: {faiss_path}")

    # **Save BM25 index**
    tokenized_corpus = [text.split() for text in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(bm25_path, "w") as f:
        json.dump(texts, f)  # **Save the full text**
    print(f"✅ BM25 index saved: {bm25_path}")

    return index, bm25

# **6️⃣ Build and Save Indexes**
print("🔹 Creating index with links...")
faiss_index_with_links, bm25_with_links = build_faiss_and_bm25(
    qa_pairs_with_links,
    "/content/drive/MyDrive/faiss_with_links3.bin",
    "/content/drive/MyDrive/bm25_with_links3.json"
)

# print("🔹 Creating index without links...")
# faiss_index_without_links, bm25_without_links = build_faiss_and_bm25(
#     qa_pairs_without_links,
#     "/content/drive/MyDrive/faiss_without_links1.bin",
#     "/content/drive/MyDrive/bm25_without_links1.json"
# )

# **7️⃣ Evaluate Retrieval Performance**
def hybrid_retrieve(query, k=20, alpha=0.5, faiss_index=None, bm25_model=None, qa_pairs=None):
    """ Perform hybrid retrieval using FAISS + BM25 """
    query = normalize_unicode(query)

    # 🔹 BM25 keyword matching
    tokenized_query = query.split()
    bm25_scores = bm25_model.get_scores(tokenized_query)

    # 🔹 FAISS semantic search
    dense_embedding = embed_text(query)
    _, faiss_indices = faiss_index.search(dense_embedding.reshape(1, -1), k)

    # 🔹 Calculate Hybrid Score
    dense_scores = np.dot(dense_embedding, np.array([embed_text(f"Q: {item['question']} A: {item['answer']}") for item in qa_pairs]).T)
    hybrid_scores = alpha * bm25_scores + (1 - alpha) * dense_scores
    ranked_indices = np.argsort(hybrid_scores)[::-1][:k]

    return [qa_pairs[i]["answer"] for i in ranked_indices]

def rerank_with_cross_encoder(query, retrieved_texts, top_k=5):
    """ Rerank retrieved results using a Cross Encoder """
    pairs = [(query, text) for text in retrieved_texts]
    scores = cross_encoder.predict(pairs)
    sorted_indices = np.argsort(scores)[::-1][:top_k]
    return [retrieved_texts[i] for i in sorted_indices]

def evaluate_retrieval(faiss_index, bm25_model, qa_pairs, eval_dataset, label):
    """ Evaluate Recall@K & MRR """
    k_values = [5, 10]
    recall_results = {k: 0 for k in k_values}
    reciprocal_ranks = []

    for example in eval_dataset:
        question, correct_answer = example["question"], example["answer"]
        retrieved_answers = hybrid_retrieve(question, k=20, alpha=0.5, faiss_index=faiss_index, bm25_model=bm25_model, qa_pairs=qa_pairs)
        reranked_answers = rerank_with_cross_encoder(question, retrieved_answers, top_k=max(k_values))

        # Recall@K calculation
        for k in k_values:
            if correct_answer in reranked_answers[:k]:
                recall_results[k] += 1

        # MRR calculation
        try:
            rank = reranked_answers.index(correct_answer) + 1
            reciprocal_ranks.append(1 / rank)
        except ValueError:
            reciprocal_ranks.append(0)

    print(f"\n🔹 **{label} Evaluation Results**")
    print({f"Recall@{k}": recall_results[k] / len(eval_dataset) for k in k_values})
    print(f"MRR: {np.mean(reciprocal_ranks):.4f}")

# **8️⃣ Run Evaluation**
eval_data = qa_pairs_with_links[:100]  # **Select 100 samples for evaluation**
evaluate_retrieval(faiss_index_with_links, bm25_with_links, qa_pairs_with_links, eval_data, "Version with links")
# evaluate_retrieval(faiss_index_without_links, bm25_without_links, qa_pairs_without_links, eval_data, "Version without links")
