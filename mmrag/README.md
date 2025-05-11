# MM-RAG: Multimodal Retrieval-Augmented Generation

A production-ready, modular Multimodal RAG system combining vision and text for advanced question answering and generation.

## Features
- Accepts image and text query as input
- CLIP-based vision encoder
- FAISS-based retriever (hybrid text+image)
- Cross-modal fusion with attention, gated, or transformer-based fusion
- LLaMA/Mistral generator with LoRA/AdapterFusion support
- PyTorch Lightning training for adapters
- FastAPI server for inference (single and batch)
- Distributed and data-parallel support (configurable)
- Modular, SOLID, and DRY codebase

## Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Configuration
Edit `mmrag/configs/mmrag.yaml` to set model names, device, fusion type, and distributed/data-parallel options.

## Indexing Documents
```bash
python mmrag/scripts/add_to_index.py --docs path/to/docs.txt
```

## Running Inference
```bash
python mmrag/scripts/run_inference.py --image path/to/image.jpg --query "What is shown?"
```

## Training Adapters
```bash
python mmrag/main.py train_adapter --config mmrag/configs/mmrag.yaml
```

## FastAPI Server
```bash
uvicorn mmrag.api.app:app --reload
```

### Batch API
POST `/mmrag_batch` with multiple images and queries for batch inference.

## Evaluation & Ablation
- BLEU, Exact Match (EM), F1, and latency metrics
- Ablation runner: compare vision-only, retrieval-only, and combined

## Advanced Fusion
- Set `fusion_type` in config to `attention`, `gated`, or `transformer`

## Distributed & Data-Parallel
- Enable `distributed: true` or `data_parallel: true` in config for large-scale training/inference

## Extensibility & Research
- Easily add new encoders, retrievers, or fusion strategies
- Hot-swap LoRA/AdapterFusion adapters
- Modular for rapid research and ablation

## Example Query
```python
from PIL import Image
from mmrag.pipelines.mm_rag_pipeline import MMRAGPipeline
pipeline = MMRAGPipeline("mmrag/configs/mmrag.yaml")
img = Image.open("example.jpg")
answer = pipeline(img, "Describe the image and related facts.")
print(answer)
```

## Modules
- `models/`: Encoders, retriever, fusion, generator
- `pipelines/`: Full MM-RAG orchestration
- `trainers/`: Adapter training
- `evaluation/`: Metrics and ablation
- `api/`: FastAPI server
- `scripts/`: Indexing and inference scripts

## Requirements
- torch, transformers, sentence-transformers, faiss, peft, pytorch-lightning, fastapi, uvicorn, omegaconf, nltk, pillow, scikit-learn

## License
MIT 