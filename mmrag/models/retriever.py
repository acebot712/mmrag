import os
from typing import List, Optional, Union, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import torch

class FaissRetriever:
    """
    FAISS-based dense retriever supporting text and image embeddings, with hybrid scoring.
    """
    def __init__(
        self,
        dim: int,
        index_path: Optional[str] = None,
        text_encoder_name: str = 'all-MiniLM-L6-v2',
        device: Optional[Union[str, torch.device]] = None
    ):
        self.dim = dim
        # Handle auto device detection
        if device == "auto" or device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.text_encoder = SentenceTransformer(text_encoder_name, device=self.device)
        self.index = faiss.IndexFlatIP(dim)
        self.doc_ids: List[str] = []
        self.doc_texts: List[str] = []
        self.index_path = index_path
        if index_path and os.path.exists(index_path):
            self.load(index_path)

    def add(self, embeddings: np.ndarray, doc_ids: List[str], doc_texts: Optional[List[str]] = None):
        assert embeddings.shape[1] == self.dim
        self.index.add(embeddings.astype(np.float32))
        self.doc_ids.extend(doc_ids)
        if doc_texts:
            self.doc_texts.extend(doc_texts)

    def encode_text(self, queries: Union[str, List[str]]) -> np.ndarray:
        if isinstance(queries, str):
            queries = [queries]
        emb = self.text_encoder.encode(queries, convert_to_numpy=True, normalize_embeddings=True)
        return emb

    def search(self, query_emb: np.ndarray, top_k: int = 5) -> List[Tuple[str, float, Optional[str]]]:
        D, I = self.index.search(query_emb.astype(np.float32), top_k)
        results = []
        for idx, score in zip(I[0], D[0]):
            doc_id = self.doc_ids[idx]
            doc_text = self.doc_texts[idx] if self.doc_texts else None
            results.append((doc_id, float(score), doc_text))
        return results

    def hybrid_search(self, text_query: str, image_emb: Optional[np.ndarray] = None, alpha: float = 0.5, top_k: int = 5) -> List[Tuple[str, float, Optional[str]]]:
        text_emb = self.encode_text(text_query)
        if image_emb is not None:
            # Weighted sum of text and image embeddings
            hybrid_emb = alpha * text_emb + (1 - alpha) * image_emb
        else:
            hybrid_emb = text_emb
        return self.search(hybrid_emb, top_k=top_k)

    def save(self, path: Optional[str] = None):
        path = path or self.index_path
        if not path:
            raise ValueError("No path specified for saving index.")
        faiss.write_index(self.index, path)

    def load(self, path: str):
        self.index = faiss.read_index(path)

</rewritten_file> 