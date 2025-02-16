""""
The results of unarxive are retrieved using the previously generated index.
Use faiss+bm25 for hybrid search, then use flashrank for reordering.
Code reference: https://python.langchain.com/docs/integrations/retrievers/flashrank-reranker/
"""
import faiss
import numpy as np
import os
from whoosh.qparser import QueryParser
from whoosh.index import open_dir
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.docstore.document import Document
from flashrank import Ranker, RerankRequest
from sentence_transformers import SentenceTransformer

# 📌 Configure index paths
BM25_INDEX_DIR = "/data/horse/ws/jilu743f-scholarlyLLM/embeddings/mergedbge_bm25_index"
FAISS_INDEX_PATH = "/data/horse/ws/jilu743f-scholarlyLLM/embeddings/mergedbge_faiss_index"

# 📌 Configure FlashRank model (you can choose a larger model)
FLASHRANK_MODEL = "ms-marco-MiniLM-L-12-v2"  # or "rank-T5-flan"

# 1️⃣ Load FAISS vector index
def load_faiss_index():
    #index = faiss.read_index(FAISS_INDEX_PATH)
    index = faiss.read_index(FAISS_INDEX_PATH, faiss.IO_FLAG_MMAP)
    index.nprobe = 10
    print(f"✅ FAISS index loaded with {index.ntotal} vectors.")

    with open(f"{FAISS_INDEX_PATH}_texts.txt", "r", encoding="utf-8") as f:
        texts = [line.strip() for line in f.readlines()]

    return index, texts

# 2️⃣ Load BM25 index
def load_bm25_index():
    if not os.path.exists(BM25_INDEX_DIR):
        raise FileNotFoundError(f"❌ BM25 index directory not found: {BM25_INDEX_DIR}")
    return open_dir(BM25_INDEX_DIR)

# 3️⃣ Generate query embedding
def generate_query_embedding(query, model_name="BAAI/bge-m3"):
    embedding_model = HuggingFaceEmbeddings(
    model_name = model_name,
    model_kwargs={"device": "cpu"}
    )
    query_embedding = embedding_model.embed_query(query)
    return np.array([query_embedding], dtype="float32")

# 4️⃣ FAISS search
def faiss_search(query, index, texts, top_k=5):
    query_embedding = generate_query_embedding(query)
    distances, indices = index.search(query_embedding, top_k)

    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(texts):  # Ensure the index is valid
            results.append({"text": texts[idx], "score": float(distances[0][i])})

    return results

# 5️⃣ BM25 keyword search
def bm25_search(query, ix, top_k=5):
    with ix.searcher() as searcher:
        parser = QueryParser("abstract", ix.schema)
        query_obj = parser.parse(query)
        results = searcher.search(query_obj, limit=top_k)
        return [{"text": r["abstract"], "score": r.score} for r in results]

# 6️⃣ Result fusion (weighted fusion)
def hybrid_search(query, faiss_index, faiss_texts, bm25_index, top_k=10, alpha=0.7):
    faiss_results = faiss_search(query, faiss_index, faiss_texts, top_k)
    bm25_results = bm25_search(query, bm25_index, top_k)

    # Normalize scores
    max_faiss = max([r["score"] for r in faiss_results]) if faiss_results else 1
    max_bm25 = max([r["score"] for r in bm25_results]) if bm25_results else 1

    faiss_results = [{"text": r["text"], "score": (r["score"] / max_faiss) * alpha} for r in faiss_results]
    bm25_results = [{"text": r["text"], "score": (r["score"] / max_bm25) * (1 - alpha)} for r in bm25_results]

    # Merge and sort
    combined_results = faiss_results + bm25_results
    combined_results = sorted(combined_results, key=lambda x: x["score"], reverse=True)

    return combined_results[:top_k]

# 7️⃣ Rerank with FlashRank
def rerank_with_flashrank(query, results):
    if not results:
        return []

    ranker = Ranker(model_name=FLASHRANK_MODEL, cache_dir="/data/horse/ws/jilu743f-scholarlyLLM/cache")  # Initialize FlashRank
    passages = [r["text"] for r in results]
    formatted_results = [{"text": p} if isinstance(p, str) else p for p in passages]

    request = RerankRequest(query=query, passages=formatted_results)

    ranked_results = ranker.rerank(request)

    # Re-sort by FlashRank score
    reranked_results = [
        {"text": results[i]["text"], "score": ranked_results[i]["score"]}
        for i in range(len(results))
    ]
    return sorted(reranked_results, key=lambda x: x["score"], reverse=True)

# 8️⃣ Run retrieval
if __name__ == "__main__":
    print("🔍 Loading indexes...")
    faiss_index, faiss_texts = load_faiss_index()
    bm25_index = load_bm25_index()

    query = "Which model has achieved the highest Accuracy score on the Story Cloze Test benchmark dataset?"

    print("\n🔍 Retrieving documents...")
    hybrid_results = hybrid_search(query, faiss_index, faiss_texts, bm25_index, top_k=10)

    print("\n⚡ Reranking results with FlashRank...")
    final_results = rerank_with_flashrank(query, hybrid_results)

    print("\n📑 Final Ranked Results:")
    for i, result in enumerate(final_results):
        print(f"\n📝 Result {i + 1} (Score: {result['score']:.4f})")
        print(f"📖 {result['text']}")
        print("-" * 80)
