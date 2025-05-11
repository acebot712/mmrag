import torch
import pytest
from mmrag.models.fusion import CrossModalFusionBlock

def test_fusion_block_output_shape():
    batch, embed_dim, seq_len, num_docs = 2, 512, 4, 3
    fusion = CrossModalFusionBlock(embed_dim=embed_dim)
    image_emb = torch.randn(batch, 1, embed_dim)
    text_emb = torch.randn(batch, seq_len, embed_dim)
    doc_emb = torch.randn(batch, num_docs, embed_dim)
    out = fusion(image_emb, text_emb, doc_emb)
    assert out.shape == (batch, 1, embed_dim) 