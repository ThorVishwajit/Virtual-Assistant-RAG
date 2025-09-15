import os
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from config import DATA_DIR


class RAGEngine:
    def __init__(self, embeddings, documents):
        self.embeddings = embeddings
        self.documents = documents
        # Precompute embeddings for dataset
        self.embeddings_matrix = np.array(
            [self.embeddings.encode(doc["instruction"]) for doc in documents]
        )

    def query(self, text: str):
        # Encode the user's query
        query_emb = self.embeddings.encode(text).reshape(1, -1)

        # Calculate cosine similarity against all document instructions
        scores = cosine_similarity(query_emb, self.embeddings_matrix)[0]

        # Get the index and score of the single best match
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
        best_doc = self.documents[best_idx]

        # Also get the indices of the top 3 best matches to use as context
        top_indices = np.argsort(scores)[-3:][::-1]
        context_docs = [self.documents[i] for i in top_indices]

        # The engine no longer falls back. It just provides the best data it can find.
        return best_doc, best_score, context_docs


def build_rag_engine():
    dataset_path = os.path.join(DATA_DIR, "rag_dataset", "anime_friend.jsonl")

    with open(dataset_path, "r", encoding="utf-8") as f:
        documents = [json.loads(line) for line in f]

    embeddings = SentenceTransformer("all-MiniLM-L6-v2")

    return RAGEngine(embeddings, documents)