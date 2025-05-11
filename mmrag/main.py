import argparse
import sys
import subprocess

def main():
    parser = argparse.ArgumentParser(description="MM-RAG CLI")
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("add_to_index", help="Add documents to FAISS index")
    subparsers.add_parser("run_inference", help="Run MM-RAG inference")
    subparsers.add_parser("train_adapter", help="Train LoRA/AdapterFusion adapter")

    args, unknown = parser.parse_known_args()
    if args.command == "add_to_index":
        subprocess.run([sys.executable, "mmrag/scripts/add_to_index.py"] + unknown)
    elif args.command == "run_inference":
        subprocess.run([sys.executable, "mmrag/scripts/run_inference.py"] + unknown)
    elif args.command == "train_adapter":
        subprocess.run([sys.executable, "mmrag/trainers/adapter_trainer.py"] + unknown)
    else:
        parser.print_help()

if __name__ == "__main__":
    main() 