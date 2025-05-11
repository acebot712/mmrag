import argparse
from PIL import Image
from mmrag.pipelines.mm_rag_pipeline import MMRAGPipeline

def main():
    parser = argparse.ArgumentParser(description="Run MM-RAG inference.")
    parser.add_argument("--config", type=str, default="mmrag/configs/mmrag.yaml")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    pipeline = MMRAGPipeline(args.config)
    img = Image.open(args.image).convert("RGB")
    answer = pipeline(img, args.query)
    print("Answer:", answer)

if __name__ == "__main__":
    main() 