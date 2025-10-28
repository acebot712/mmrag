#!/usr/bin/env python3
"""
Quickstart demo for MM-RAG.
This script demonstrates the full pipeline without requiring pre-existing data.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image
import torch
import numpy as np

def demo_without_heavy_models():
    """Lightweight demo using sentence transformers only (no LLM required)."""
    print("=" * 70)
    print("MM-RAG Quickstart Demo (Lightweight Mode)")
    print("=" * 70)
    print()

    # Step 1: Generate test image
    print("[1/5] Generating test image...")
    from examples.generate_test_image import generate_test_image
    image_path = generate_test_image("examples/test_tower.png")
    print(f"✓ Image created: {image_path}\n")

    # Step 2: Initialize retriever and index documents
    print("[2/5] Indexing sample documents...")
    from mmrag.models.retriever import FaissRetriever

    retriever = FaissRetriever(
        dim=384,  # all-MiniLM-L6-v2 dimension
        index_path="mmrag/data/quickstart.index",
        text_encoder_name='all-MiniLM-L6-v2',
        device='cpu'  # Use CPU for demo
    )

    # Load and index documents
    with open("examples/sample_documents.txt", "r") as f:
        docs = [line.strip() for line in f if line.strip()]

    doc_embeddings = retriever.encode_text(docs)
    doc_ids = [f"doc_{i}" for i in range(len(docs))]
    retriever.add(doc_embeddings, doc_ids, docs)
    retriever.save()
    print(f"✓ Indexed {len(docs)} documents\n")

    # Step 3: Initialize vision encoder
    print("[3/5] Loading vision encoder...")
    from mmrag.models.vision_encoder import VisionEncoder

    vision_encoder = VisionEncoder(
        model_name="openai/clip-vit-base-patch16",
        device="cpu"
    )
    print("✓ Vision encoder loaded\n")

    # Step 4: Process query
    print("[4/5] Processing multimodal query...")
    img = Image.open(image_path)
    query = "What famous tower is this and where is it located?"

    # Encode image
    image_emb = vision_encoder.encode(img)
    print(f"✓ Image encoded: shape {image_emb.shape}")

    # Hybrid search
    results = retriever.hybrid_search(query, image_emb.cpu().numpy(), alpha=0.5, top_k=3)
    print(f"✓ Retrieved {len(results)} relevant documents\n")

    # Step 5: Display results
    print("[5/5] Results:")
    print("-" * 70)
    print(f"Query: {query}")
    print(f"Image: {image_path}")
    print()
    print("Top Retrieved Documents:")
    for i, (doc_id, score, text) in enumerate(results, 1):
        print(f"\n{i}. [Score: {score:.4f}] {doc_id}")
        print(f"   {text[:200]}..." if len(text) > 200 else f"   {text}")

    print()
    print("=" * 70)
    print("Demo completed successfully!")
    print()
    print("NOTE: This demo uses retrieval only. For full generation,")
    print("      run with a language model (requires more resources).")
    print("=" * 70)


def demo_with_full_pipeline():
    """Full demo including generation (requires LLM - optional)."""
    print("=" * 70)
    print("MM-RAG Full Pipeline Demo")
    print("=" * 70)
    print("\nWARNING: This requires downloading a large language model.")
    print("Press Ctrl+C to cancel, or wait 5 seconds to continue...")

    import time
    try:
        time.sleep(5)
    except KeyboardInterrupt:
        print("\nCancelled. Run lightweight demo instead.")
        return

    from mmrag.pipelines.mm_rag_pipeline import MMRAGPipeline

    # Ensure config exists
    config_path = "mmrag/configs/mmrag.yaml"
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        return

    print("\n[1/3] Loading MM-RAG pipeline (this may take several minutes)...")
    try:
        pipeline = MMRAGPipeline(config_path)
        print("✓ Pipeline loaded\n")
    except Exception as e:
        print(f"✗ Error loading pipeline: {e}")
        print("\nFalling back to lightweight demo...")
        demo_without_heavy_models()
        return

    print("[2/3] Generating test image...")
    from examples.generate_test_image import generate_test_image
    image_path = generate_test_image("examples/test_tower.png")

    print("[3/3] Running inference...")
    img = Image.open(image_path)
    query = "What famous structure is shown in this image?"

    try:
        answer = pipeline(img, query)
        print("\n" + "=" * 70)
        print("RESULT:")
        print("-" * 70)
        print(f"Query: {query}")
        print(f"Answer: {answer}")
        print("=" * 70)
    except Exception as e:
        print(f"✗ Error during inference: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="MM-RAG Quickstart Demo")
    parser.add_argument(
        "--full",
        action="store_true",
        help="Run full pipeline with LLM (requires GPU and downloads large models)"
    )
    args = parser.parse_args()

    if args.full:
        demo_with_full_pipeline()
    else:
        demo_without_heavy_models()
