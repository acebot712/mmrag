import numpy as np
import pytest
from mmrag.models.retriever import FaissRetriever

def test_faiss_retriever_basic():
    retriever = FaissRetriever(dim=384, text_encoder_name='all-MiniLM-L6-v2')
    docs = ["The cat sits on the mat.", "A dog barks.", "Birds fly in the sky."]
    doc_ids = ["1", "2", "3"]
    embeddings = retriever.encode_text(docs)
    retriever.add(embeddings, doc_ids, docs)
    query = "animal on mat"
    query_emb = retriever.encode_text(query)
    results = retriever.search(query_emb, top_k=2)
    assert len(results) == 2
    assert any("cat" in r[2] for r in results) 