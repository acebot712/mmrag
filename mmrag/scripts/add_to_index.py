import argparse
import os
import numpy as np
from omegaconf import OmegaConf
from mmrag.models.retriever import FaissRetriever
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser(description="Add documents to FAISS index.")
    parser.add_argument("--config", type=str, default="mmrag/configs/mmrag.yaml")
    parser.add_argument("--docs", type=str, required=True, help="Path to txt file with one doc per line.")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    retriever = FaissRetriever(**config.retriever)
    encoder = SentenceTransformer(config.retriever.text_encoder_name)

    with open(args.docs, "r") as f:
        docs = [line.strip() for line in f if line.strip()]
    doc_ids = [str(i) for i in range(len(docs))]
    embeddings = encoder.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
    retriever.add(embeddings, doc_ids, docs)
    retriever.save()
    print(f"Added {len(docs)} docs to index at {config.retriever.index_path}")

if __name__ == "__main__":
    main() 