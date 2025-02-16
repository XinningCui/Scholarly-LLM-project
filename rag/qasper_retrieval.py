"""
Use approximately 6,900 (question, evidence, answer) pairs from the Qasper dataset for retrieval evaluation.
First, embed the questions and evidence using the SPECTER or SciBERT model.
Then, apply hybrid retrieval (FAISS + BM25) combined with a cross-encoder to obtain the retrieval results.
Evaluate the retrieval performance using Recall@K and MRR as metrics.
Code references are from: https://huggingface.co/datasets/allenai/qasper,
https://www.sbert.net/examples/applications/cross-encoder/README.html,
https://huggingface.co/allenai/specter,
https://huggingface.co/allenai/scibert_scivocab_uncased.

"""
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document
from sentence_transformers import SentenceTransformer, CrossEncoder  # ✅ Import CrossEncoder
from rank_bm25 import BM25Okapi
import faiss
import numpy as np
import torch
import json
import os
import unicodedata
import random

# **1️⃣ Load SPECTER sentence embedding model**
specter_model = "sentence-transformers/allenai-specter"
model = SentenceTransformer(specter_model)  # ✅ Directly use SentenceTransformer

# **2️⃣ Load cross-encoder model**
cross_encoder_model = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # Choose an appropriate cross-encoder
cross_encoder = CrossEncoder(cross_encoder_model)

# **3️⃣ Define file paths**
FAISS_INDEX_PATH = "faiss_index2.bin"
EMBEDDING_PATH = "qa_evidence_embeddings2.npy"
PAPER_IDS_PATH = "paper_ids2.json"
EVIDENCE_PATH = "faiss_qedata_cleaned.json"
QA_PAIRS_PATH = "qa_pairs2.json"  # Store question and evidence JSON
EVAL_DATA_PATH = "qa_data_cleaned.json"  # Evaluation dataset

# **4️⃣ Unified Unicode processing**
def normalize_unicode(text):
    """ Normalize Unicode to ensure text consistency across different stages """
    return unicodedata.normalize("NFKC", text) if text else ""

# **5️⃣ Compute embedding method**
def embed_text(text):
    """ Generate text embeddings using SPECTER """
    text = normalize_unicode(text)  # **Normalize text before embedding**
    embedding = model.encode(text, convert_to_numpy=True, normalize_embeddings=True)  # ✅ Adapted for SentenceTransformer
    return embedding

# **6️⃣ Load data**
with open(EVIDENCE_PATH, "r") as f:
    evidence_data = json.load(f)

corpus = [item["evidence"] for item in evidence_data]
paper_ids = [item["id"] for item in evidence_data]

# **Load Questions and Evidence with Unicode normalization**
qa_pairs = []
for item in evidence_data:
    question = normalize_unicode(item.get("question", "").strip())
    evidence = normalize_unicode(item.get("evidence", "").strip())
    if question and evidence:
        qa_pairs.append({
            "qa_text": f"Q: {question} A: {evidence}",
            "question": question,
            "evidence": evidence,
            "id": item["id"]
        })

# **7️⃣ Load/Create FAISS index**
if os.path.exists(FAISS_INDEX_PATH) and os.path.exists(EMBEDDING_PATH):
    print("✅ Found stored FAISS index, loading...")
    index = faiss.read_index(FAISS_INDEX_PATH)
    embeddings = np.load(EMBEDDING_PATH)
    with open(PAPER_IDS_PATH, "r") as f:
        paper_ids = json.load(f)
else:
    print("⚠️ No stored index found, computing embeddings and building FAISS index...")

    # **Compute embeddings for "Question + Evidence"**
    qa_texts = [item["qa_text"] for item in qa_pairs]
    embeddings = np.array([embed_text(text) for text in qa_texts], dtype="float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Use inner product index
    index.add(embeddings)  # Add to FAISS index

    # **Save FAISS index and embeddings**
    faiss.write_index(index, FAISS_INDEX_PATH)
    np.save(EMBEDDING_PATH, embeddings)
    with open(PAPER_IDS_PATH, "w") as f:
        json.dump([item["id"] for item in qa_pairs], f)
    with open(QA_PAIRS_PATH, "w") as f:
        json.dump(qa_pairs, f, indent=4)

    print("✅ FAISS index has been created and saved!")

# **8️⃣ Create BM25 index**
tokenized_corpus = [normalize_unicode(item["qa_text"]).split() for item in qa_pairs]  # **Normalize text**
bm25 = BM25Okapi(tokenized_corpus)

# **9️⃣ Hybrid retrieval (BM25 + FAISS)**
def hybrid_retrieve(query, alpha=0.5, rerank_k=50):
    """ First retrieve rerank_k candidates using Hybrid (BM25 + FAISS) and then rerank with Cross-Encoder """

    # **BM25 keyword matching scores**
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # **Compute dense embedding for semantic matching**
    dense_embedding = embed_text(query)
    _, faiss_indices = index.search(dense_embedding.reshape(1, -1), rerank_k)

    # **Retrieve FAISS indices**
    retrieved_faiss_indices = faiss_indices[0]

    # **Calculate dense scores**
    dense_scores = np.array([np.dot(dense_embedding, embeddings[i]) for i in retrieved_faiss_indices])

    # **Retrieve BM25 scores corresponding to FAISS indices**
    bm25_selected_scores = np.array([bm25_scores[i] for i in retrieved_faiss_indices])

    # **Calculate Hybrid Score**
    hybrid_scores = alpha * bm25_selected_scores + (1 - alpha) * dense_scores
    ranked_indices = np.argsort(hybrid_scores)[::-1][:rerank_k]  # Get the top rerank_k results

    retrieved_evidences = [corpus[i] for i in retrieved_faiss_indices[ranked_indices]]
    retrieved_paper_ids = [paper_ids[i] for i in retrieved_faiss_indices[ranked_indices]]

    return retrieved_evidences, retrieved_paper_ids

# **🔟 Rerank with cross-encoder**
def rerank_with_cross_encoder(query, retrieved_evidences, top_k=5):
    """ Rerank the retrieved results using a cross-encoder """

    # **Create query-evidence pairs**
    pairs = [[query, evidence] for evidence in retrieved_evidences]

    # **Compute relevance scores using Cross-Encoder**
    scores = cross_encoder.predict(pairs)

    # **Sort evidence based on scores**
    ranked_indices = np.argsort(scores)[::-1][:top_k]
    reranked_evidences = [retrieved_evidences[i] for i in ranked_indices]

    return reranked_evidences

# **1️⃣1️⃣ Evaluate Recall@K and MRR**
def evaluate_retrieval(k_values=[5, 10], alpha_values=[0.2, 0.4, 0.6, 0.8], sample_size=1000):
    """ Calculate Recall@K and MRR for different alpha values """

    with open(EVAL_DATA_PATH, "r") as f:
        eval_data = json.load(f)

    eval_data = random.sample(eval_data, min(sample_size, len(eval_data)))

    best_alpha = None
    best_mrr = 0

    for alpha in alpha_values:
        recall_at_k = {k: 0 for k in k_values}
        reciprocal_ranks = []

        for example in eval_data:
            question = normalize_unicode(example["question"])  # **Normalize the question**
            correct_evidence = normalize_unicode(example["evidence"])  # **Normalize the correct answer**

            # **Perform hybrid retrieval**
            retrieved_evidences, _ = hybrid_retrieve(question, rerank_k=50, alpha=alpha)

            # **Rerank with Cross-Encoder**
            reranked_evidences = rerank_with_cross_encoder(question, retrieved_evidences, top_k=max(k_values))

            # **Calculate Recall@K**
            for k in k_values:
                retrieved_k = reranked_evidences[:k]
                if correct_evidence in retrieved_k:
                    recall_at_k[k] += 1

            # **Calculate MRR**
            try:
                rank = reranked_evidences.index(correct_evidence) + 1
                reciprocal_ranks.append(1 / rank)
            except ValueError:
                reciprocal_ranks.append(0)

        recall_results = {f"Recall@{k}": recall_at_k[k] / len(eval_data) for k in k_values}
        mrr = np.mean(reciprocal_ranks)

        print(f"\n🔹 **Alpha: {alpha} Evaluation Results**")
        print(recall_results)
        print(f"MRR: {mrr:.4f}")

        if mrr > best_mrr:
            best_mrr = mrr
            best_alpha = alpha

    print(f"\n✅ Best Alpha: {best_alpha} (MRR: {best_mrr:.4f})")

# **1️⃣2️⃣ Run evaluation**
evaluate_retrieval()
