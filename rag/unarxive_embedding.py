"""
Embed 20,000 computer science articles from the unarXiv dataset using FAISS and the SPECTER model to generate an index.
Then, create an index using BM25.
The code is referenced from: https://python.langchain.com/docs/integrations/vectorstores/faiss/
and https://python.langchain.com/docs/integrations/retrievers/bm25/
The embedding model is sourced from: https://huggingface.co/allenai/specter.
The embedded model can also be replaced with BGE-m3 from: https://huggingface.co/BAAI/bge-m3
"""

import os
import json
import faiss
import numpy as np
import torch
from langchain_community.document_loaders import JSONLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
import sentence_transformers
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import os
from whoosh.index import create_in, open_dir
from whoosh.fields import Schema, TEXT, ID
from whoosh.qparser import QueryParser
import os
import json
from langchain.schema import Document


# If large amounts of data can be processed in batches.
# SLURM_TASK_ID = int(os.environ.get("SLURM_ARRAY_TASK_ID", 1))  # Get SLURM array task ID
# DATA_DIR = f"/data/horse/ws/jilu743f-scholarlyLLM/split_batches/part_{SLURM_TASK_ID}"
# FAISS_INDEX_PATH = f"/data/horse/ws/jilu743f-scholarlyLLM/embeddings/faiss_part2_{SLURM_TASK_ID}"
# BM25_INDEX_DIR = f"/data/horse/ws/jilu743f-scholarlyLLM/embeddings/bm25_part2_{SLURM_TASK_ID}"

DATA_DIR = f"/data/horse/ws/jilu743f-scholarlyLLM/cleaned_ai"
FAISS_INDEX_PATH = f"/data/horse/ws/jilu743f-scholarlyLLM/embeddings/faiss_ai_index"
BM25_INDEX_DIR = f"/data/horse/ws/jilu743f-scholarlyLLM/embeddings/bm25_ai_index"

def extract_publish_time(paper_id):
    try:
        year_prefix, month = paper_id.split(".")[0][:2], paper_id.split(".")[0][2:]
        publish_year = f"20{year_prefix}" if int(year_prefix) < 50 else f"19{year_prefix}"  # Handle 20xx and 19xx
        publish_time = f"{publish_year}.{month}"
        return publish_time
    except Exception as e:
        print(f"⚠️ Failed to parse paper_id: {paper_id}, error: {e}")
        return "Unknown"

# 1️ Load JSONL data
def load_and_clean_data():
    docs = []

    # Iterate over all JSON files in the folder
    for filename in os.listdir(DATA_DIR):
        if filename.endswith(".json"):  # Process only .json files
            file_path = os.path.join(DATA_DIR, filename)

            with open(file_path, "r", encoding="utf-8") as f:
                paper = json.load(f)  # Directly parse JSON without JSONLoader

                # Extract fields
                paper_id = paper.get("paper_id", "Unknown")
                title = paper.get("title", "Unknown Title")
                authors = paper.get("authors", "Unknown Authors")
                abstract = paper.get("abstract", "")
                body_text = paper.get("body_text", [])
                publish_time = extract_publish_time(paper_id)

                # ** Extract body text **
                text_list = [section.get("text", "") for section in body_text if isinstance(section, dict)]
                body_text_str = "\n".join(text_list)  # **Merge into a single string**

                # **Directly use each paragraph from JSON**
                for section in paper.get("body_text", []):
                    text = section.get("text", "").strip()
                    if text:  # Store only non-empty text
                        doc = Document(
                            page_content=text,
                            metadata={
                                "paper_id": paper_id,
                                "title": title,
                                "authors": authors,
                                "publish_time": publish_time,
                                "abstract": abstract
                            }
                        )
                        docs.append(doc)

    return docs


# # 2️ Generate embeddings (Specter)
def generate_embeddings(docs):
    """Batch compute text embeddings to prevent OOM"""
    try:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/allenai-specter")
        texts = [doc.page_content for doc in docs]  # Only take the body text
        embeddings = embedding_model.embed_documents(texts)
        return np.array(embeddings, dtype="float32")  # Return embeddings

    except Exception as e:
        print(f"❌ Error in embedding: {e}")
        raise


# 3️ Create FAISS index (IVFFlat for improved search efficiency)
def create_and_save_faiss_index(embeddings, docs):
    dimension = embeddings.shape[1]
    if len(embeddings) < 100 * 39:  # 100 cluster centers × 39
        print("⚠ Warning: Not enough data to train FAISS IVFFlat. Using IndexFlatL2 instead.")
        index = faiss.IndexFlatL2(dimension)
    else:
        quantizer = faiss.IndexFlatL2(dimension)
        index = faiss.IndexIVFFlat(quantizer, dimension, 100)
        index.train(embeddings)

    # Train & add data
    index.train(embeddings)
    index.add(embeddings)
    print(f"Added {len(embeddings)} vectors to FAISS index.")

    # Ensure path exists
    if os.path.exists(FAISS_INDEX_PATH):
        print("⚠ Warning: FAISS index file already exists. Overwriting...")

    # Save index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"FAISS index saved at: {FAISS_INDEX_PATH}")

    # Save text mapping
    with open(f"{FAISS_INDEX_PATH}_texts.txt", "w", encoding="utf-8") as f:
        for doc in docs:
            f.write(doc.page_content + "\n")


# Define Schema
schema = Schema(paper_id=ID(stored=True), title=TEXT(stored=True), abstract=TEXT(stored=True))

# 4 Create bm25 index
def create_bm25_index(docs):
    if not os.path.exists(BM25_INDEX_DIR):
        os.makedirs(BM25_INDEX_DIR)  # Ensure the directory exists
        whoosh_index = create_in(BM25_INDEX_DIR, schema)  # Use Whoosh's create_in()
    else:
        whoosh_index = open_dir(BM25_INDEX_DIR)

    writer = whoosh_index.writer()
    for doc in docs:
        writer.add_document(paper_id=doc.metadata["paper_id"],
            title=doc.metadata["title"],
            abstract=doc.metadata["abstract"])
    writer.commit()
    print("BM25 Index Created!")


# Main process
if __name__ == "__main__":
    print("Loading and cleaning data...")
    docs = load_and_clean_data()

    print("Generating embeddings...")
    embeddings = generate_embeddings(docs)

    print("Creating and saving FAISS index...")
    create_and_save_faiss_index(embeddings, docs)

    print(" Creating BM25 index...")
    create_bm25_index(docs)

    print("FAISS embedding process completed!")
