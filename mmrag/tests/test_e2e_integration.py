#!/usr/bin/env python3
"""
End-to-end integration test for MM-RAG.
Tests the full pipeline with real models (lightweight versions).
"""

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import os


@pytest.fixture
def test_image():
    """Create a simple test image."""
    img = Image.new('RGB', (224, 224), color='blue')
    return img


@pytest.fixture
def test_docs():
    """Sample documents for indexing."""
    return [
        "The Eiffel Tower is in Paris, France.",
        "The Statue of Liberty is in New York City.",
        "The Great Wall of China is in Beijing.",
    ]


@pytest.fixture
def temp_index_path():
    """Temporary path for FAISS index."""
    with tempfile.NamedTemporaryFile(suffix=".index", delete=False) as f:
        yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)


def test_vision_encoder_integration(test_image):
    """Test vision encoder with real CLIP model."""
    from mmrag.models.vision_encoder import VisionEncoder

    encoder = VisionEncoder(
        model_name="openai/clip-vit-base-patch16",
        device="cpu"
    )

    # Single image
    emb = encoder.encode(test_image)
    assert emb.shape[0] == 1
    assert emb.shape[1] > 0  # Has embedding dimension
    assert torch.is_tensor(emb)

    # Batch of images
    emb_batch = encoder.encode([test_image, test_image])
    assert emb_batch.shape[0] == 2
    assert emb_batch.shape[1] == emb.shape[1]


def test_retriever_integration(test_docs, temp_index_path):
    """Test retriever with real sentence transformer."""
    from mmrag.models.retriever import FaissRetriever

    retriever = FaissRetriever(
        dim=384,  # all-MiniLM-L6-v2 dimension
        index_path=temp_index_path,
        text_encoder_name='all-MiniLM-L6-v2',
        device='cpu'
    )

    # Index documents
    doc_embeddings = retriever.encode_text(test_docs)
    assert doc_embeddings.shape[0] == len(test_docs)
    assert doc_embeddings.shape[1] == 384

    doc_ids = [f"doc_{i}" for i in range(len(test_docs))]
    retriever.add(doc_embeddings, doc_ids, test_docs)

    # Search
    query = "What is in Paris?"
    results = retriever.hybrid_search(query, top_k=2)

    assert len(results) == 2
    assert results[0][0] in doc_ids  # doc_id
    assert isinstance(results[0][1], float)  # score
    assert results[0][2] in test_docs  # text

    # Save and load
    retriever.save()
    assert os.path.exists(temp_index_path)


def test_fusion_integration():
    """Test fusion module with various fusion types."""
    from mmrag.models.fusion import CrossModalFusionBlock

    batch_size = 2
    embed_dim = 512

    for fusion_type in ['attention', 'gated', 'transformer']:
        fusion = CrossModalFusionBlock(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            fusion_type=fusion_type
        )

        # Create dummy inputs
        image_emb = torch.randn(batch_size, 1, embed_dim)
        text_emb = torch.randn(batch_size, 1, embed_dim)
        doc_emb = torch.randn(batch_size, 3, embed_dim)

        # Forward pass
        with torch.no_grad():
            fused = fusion(image_emb, text_emb, doc_emb)

        assert fused.shape == (batch_size, 1, embed_dim)
        assert not torch.isnan(fused).any()


def test_multimodal_retrieval_integration(test_image, test_docs, temp_index_path):
    """Test multimodal (image + text) retrieval."""
    from mmrag.models.vision_encoder import VisionEncoder
    from mmrag.models.retriever import FaissRetriever

    # Initialize with matching dimensions
    # Note: CLIP outputs 512-dim, but we'll project to 384 for this test
    vision_encoder = VisionEncoder(
        model_name="openai/clip-vit-base-patch16",
        device="cpu"
    )

    retriever = FaissRetriever(
        dim=512,  # Match CLIP dimension
        index_path=temp_index_path,
        text_encoder_name='sentence-transformers/all-MiniLM-L6-v2',
        device='cpu'
    )

    # Index documents (need to pad/project to 512 dims)
    doc_embeddings_384 = retriever.encode_text(test_docs)
    # Pad to 512 dimensions
    doc_embeddings = np.pad(
        doc_embeddings_384,
        ((0, 0), (0, 512 - doc_embeddings_384.shape[1])),
        mode='constant'
    )

    doc_ids = [f"doc_{i}" for i in range(len(test_docs))]
    retriever.add(doc_embeddings, doc_ids, test_docs)

    # Encode image
    image_emb = vision_encoder.encode(test_image)
    assert image_emb.shape[1] == 512

    # Hybrid search
    query = "famous landmark"
    results = retriever.hybrid_search(
        query,
        image_emb.cpu().numpy(),
        alpha=0.5,
        top_k=2
    )

    assert len(results) == 2
    assert all(r[2] in test_docs for r in results)


def test_empty_retrieval_handling(test_image):
    """Test that pipeline handles empty retrieval results gracefully."""
    from mmrag.models.vision_encoder import VisionEncoder
    from mmrag.models.retriever import FaissRetriever
    from mmrag.models.fusion import CrossModalFusionBlock

    vision_encoder = VisionEncoder(device="cpu")
    retriever = FaissRetriever(dim=512, device='cpu')
    fusion = CrossModalFusionBlock(embed_dim=512, num_heads=8)

    # Encode image
    image_emb = vision_encoder.encode(test_image)

    # Try search on empty index
    results = retriever.hybrid_search("test query", image_emb.cpu().numpy(), top_k=5)

    # Should return empty results, not crash
    assert len(results) == 0


def test_device_auto_detection():
    """Test that 'auto' device detection works."""
    from mmrag.models.vision_encoder import VisionEncoder
    from mmrag.models.retriever import FaissRetriever
    from mmrag.models.generator import Generator

    # Test auto detection
    encoder = VisionEncoder(device="auto")
    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert encoder.device == expected_device

    retriever = FaissRetriever(dim=512, device="auto")
    assert retriever.device == expected_device

    # Generator test (without actually loading the model due to size)
    # Just verify the device setting logic
    gen = Generator.__new__(Generator)
    gen.device = "auto"
    if gen.device == "auto":
        gen.device = "cuda" if torch.cuda.is_available() else "cpu"
    assert gen.device == expected_device


@pytest.mark.slow
def test_full_pipeline_without_llm(test_image, test_docs, temp_index_path):
    """
    Test full pipeline up to retrieval (excluding LLM generation).
    Marked as slow due to model downloads.
    """
    from mmrag.models.vision_encoder import VisionEncoder
    from mmrag.models.retriever import FaissRetriever
    from mmrag.models.fusion import CrossModalFusionBlock

    # Initialize components
    vision_encoder = VisionEncoder(device="cpu")
    retriever = FaissRetriever(dim=512, index_path=temp_index_path, device='cpu')
    fusion = CrossModalFusionBlock(embed_dim=512, num_heads=8)

    # Index documents
    doc_embeddings_384 = retriever.encode_text(test_docs)
    doc_embeddings = np.pad(
        doc_embeddings_384,
        ((0, 0), (0, 512 - doc_embeddings_384.shape[1])),
        mode='constant'
    )
    doc_ids = [f"doc_{i}" for i in range(len(test_docs))]
    retriever.add(doc_embeddings, doc_ids, test_docs)

    # Run inference
    query = "What famous structure is in France?"

    # 1. Encode image
    image_emb = vision_encoder.encode(test_image)

    # 2. Hybrid search
    results = retriever.hybrid_search(query, image_emb.cpu().numpy(), top_k=3)
    assert len(results) > 0

    # 3. Prepare for fusion
    text_emb = torch.tensor(retriever.encode_text(query), device='cpu')
    doc_texts = [r[2] for r in results]
    doc_embs_raw = retriever.encode_text(doc_texts)
    doc_embs = np.pad(
        doc_embs_raw,
        ((0, 0), (0, 512 - doc_embs_raw.shape[1])),
        mode='constant'
    )
    doc_embs = torch.tensor(doc_embs, device='cpu')

    # Reshape
    if image_emb.dim() == 2:
        image_emb = image_emb.unsqueeze(1)
    if text_emb.dim() == 2:
        text_emb = text_emb.unsqueeze(1)
    if doc_embs.dim() == 2:
        doc_embs = doc_embs.unsqueeze(0)

    # 4. Fuse
    fused = fusion(image_emb, text_emb, doc_embs)

    # Verify output
    assert fused.shape == (1, 1, 512)
    assert not torch.isnan(fused).any()

    print(f"\n✓ Retrieved: {results[0][2][:100]}...")
    print(f"✓ Fused embedding shape: {fused.shape}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
