import pytest
from unittest.mock import MagicMock
from PIL import Image
from mmrag.pipelines.mm_rag_pipeline import MMRAGPipeline

class DummyVision:
    def encode(self, img):
        import torch
        return torch.ones(1, 512)
class DummyRetriever:
    def hybrid_search(self, q, image_emb, top_k=5):
        return [("id", 1.0, "doc text")]
    def encode_text(self, text):
        import numpy as np
        return np.ones((1, 512))
class DummyFusion:
    def __call__(self, image_emb, text_emb, doc_embs):
        return image_emb
    def forward(self, image_emb, text_emb, doc_embs):
        return image_emb
class DummyGenerator:
    def generate(self, prompt, fused_emb=None):
        return "answer"

def test_pipeline_call(monkeypatch):
    pipeline = MMRAGPipeline.__new__(MMRAGPipeline)
    pipeline.vision_encoder = DummyVision()
    pipeline.retriever = DummyRetriever()
    pipeline.fusion = DummyFusion()
    pipeline.generator = DummyGenerator()
    pipeline.device = "cpu"
    img = Image.new("RGB", (224, 224))
    out = pipeline(img, "query")
    assert out == "answer" 