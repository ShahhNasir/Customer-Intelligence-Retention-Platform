"""
Retrieves a specific customer's own most-relevant interaction history for
a given query, using a FAISS search filtered (via IDSelector) to only
that customer's rows - never another customer's history.
"""

# isort: skip_file
# (import order matters on Windows - see build_vector_store.py / docs/notes.md)
from sentence_transformers import SentenceTransformer
import torch

import faiss
import numpy as np
import pandas as pd

INDEX_PATH = "src/rag/vector_store/interactions.index"
METADATA_PATH = "src/rag/vector_store/interactions_metadata.parquet"
MODEL_NAME = "all-MiniLM-L6-v2"

# Loaded once at import time, reused across every call - re-reading a
# 596MB index and re-embedding per request would be disastrous for the
# "1000+ requests/minute" target.
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = SentenceTransformer(MODEL_NAME, device=_device)
_index = faiss.read_index(INDEX_PATH)
_metadata = pd.read_parquet(METADATA_PATH)

# customer_id -> numpy array of FAISS row positions belonging to them.
# Built once here rather than filtering the whole metadata table per call.
_customer_positions = {
    customer_id: group.index.to_numpy(dtype="int64")
    for customer_id, group in _metadata.groupby("customer_id")
}


def retrieve_customer_context(customer_id: int, query: str, top_k: int = 5) -> list[dict]:
    """
    Returns this customer's own top-k most relevant interactions for the
    given query, ranked by semantic similarity - restricted to their
    history only via a FAISS IDSelector.
    """
    positions = _customer_positions.get(customer_id)
    if positions is None or len(positions) == 0:
        return []  # customer has no interaction history at all

    query_vec = _model.encode([query], normalize_embeddings=True).astype("float32")

    selector = faiss.IDSelectorArray(positions)
    search_params = faiss.SearchParameters(sel=selector)

    k = min(top_k, len(positions))
    distances, indices = _index.search(query_vec, k, params=search_params)

    results = []
    for idx, score in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        row = _metadata.iloc[idx]
        results.append(
            {
                "interaction_id": int(row.interaction_id),
                "timestamp": row.timestamp,
                "interaction_type": row.interaction_type,
                "sentiment": row.sentiment,
                "text": row.text,
                "similarity": float(score),
            }
        )
    return results


if __name__ == "__main__":
    # Manual smoke test: pick a real customer and confirm we only ever
    # get their own interactions back.
    sample_customer = int(_metadata["customer_id"].iloc[0])
    print(f"Testing retrieval for customer {sample_customer}...")

    results = retrieve_customer_context(
        sample_customer, "why might this customer be at risk of churning?"
    )
    for r in results:
        print(f"[{r['similarity']:.3f}] ({r['interaction_type']}, {r['sentiment']}) {r['text']}")
