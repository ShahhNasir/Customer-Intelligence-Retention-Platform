"""
Builds a FAISS vector index over data/raw/interactions.csv using
sentence-transformers embeddings. This is the "retrieval" half of the RAG
pipeline - run once (or whenever interactions.csv changes) to (re)build
the index; the API will load the saved index at startup rather than
re-embedding 400K+ rows per request.
"""

# isort: skip_file
import time

# Import order matters on Windows: sentence_transformers (which pulls in
# torch) must be imported before faiss - both bundle their own OpenMP
# runtime DLLs, and importing faiss first causes torch's DLL init to fail
# with a WinError 1114. Harmless on other platforms, but keep this order.
from sentence_transformers import SentenceTransformer
import torch

import faiss
import numpy as np
import pandas as pd

MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = "src/rag/vector_store/interactions.index"
METADATA_PATH = "src/rag/vector_store/interactions_metadata.parquet"


def main():
    interactions = pd.read_csv("data/raw/interactions.csv", parse_dates=["timestamp"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU detected, falling back to CPU (this will be slower).")

    print(f"Embedding {len(interactions):,} interactions using '{MODEL_NAME}'...")

    model = SentenceTransformer(MODEL_NAME, device=device)

    start = time.time()
    embeddings = model.encode(
        interactions["text"].tolist(),
        batch_size=256,
        show_progress_bar=True,
        # Normalizing to unit length lets us use inner product (fast) as
        # a stand-in for cosine similarity - the standard trick for
        # similarity search with sentence-transformers + FAISS.
        normalize_embeddings=True,
    )
    print(f"Embedded in {time.time() - start:.1f}s. Shape: {embeddings.shape}")

    embeddings = np.asarray(embeddings, dtype="float32")
    dimension = embeddings.shape[1]

    # IndexFlatIP = exact (brute-force) inner-product search. Fine at our
    # scale (~400K vectors); FAISS's approximate indexes exist for when
    # data grows into the millions and exact search gets too slow.
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    print(f"FAISS index built. Total vectors: {index.ntotal}, dimension: {dimension}")

    faiss.write_index(index, INDEX_PATH)

    # FAISS only stores vectors and returns integer positions on search -
    # it has no idea which interaction each position corresponds to. This
    # metadata file is the side-table that maps position -> real data.
    # Row order here MUST match the order embeddings were added in.
    metadata = interactions[
        ["interaction_id", "customer_id", "timestamp", "sentiment", "interaction_type", "text"]
    ]
    metadata.to_parquet(METADATA_PATH, index=False)

    print(f"Saved index to {INDEX_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
