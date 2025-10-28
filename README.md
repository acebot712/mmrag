# MM-RAG: Multimodal Retrieval-Augmented Generation

> A modular, production-ready framework for combining vision and text in retrieval-augmented generation. Built for researchers and practitioners who need flexible multimodal AI systems.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/abhijoysarkar/mmrag/blob/main/examples/MM_RAG_Demo.ipynb)

---

## Why MM-RAG?

Existing RAG systems are text-only. When you have images, diagrams, charts, or visual data, traditional RAG falls short. MM-RAG bridges this gap:

- **Multimodal Retrieval**: Combines CLIP vision encoder with text embeddings for hybrid search
- **Flexible Fusion**: Three fusion strategies (attention, gated, transformer) for combining modalities
- **Production-Ready**: FastAPI server, Docker support, comprehensive testing
- **Research-Friendly**: Modular architecture, LoRA adapters, PyTorch Lightning training
- **Actually Works**: Comes with working demo, tests, and benchmarks

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MM-RAG PIPELINE                          │
└─────────────────────────────────────────────────────────────────┘

Input: Image + Text Query
   │
   ├──► Vision Encoder (CLIP)  ──► Image Embedding (512-dim)
   │                                      │
   └──► Text Encoder  ───────────────────┼──► Hybrid Search
                                          │     │
                                          │     ▼
                                          │  FAISS Index
                                          │     │
                                          │     ▼
                                          │  Top-K Documents
                                          │     │
                                          ▼     ▼
                                    ┌─────────────────┐
                                    │  Fusion Module  │
                                    │  (Attention/    │
                                    │   Gated/        │
                                    │   Transformer)  │
                                    └─────────────────┘
                                            │
                                            ▼
                                    Fused Embedding
                                            │
                                            ▼
                                    ┌─────────────────┐
                                    │   LLM Generator │
                                    │   (LLaMA/GPT)   │
                                    └─────────────────┘
                                            │
                                            ▼
                                      Final Answer
```

## Quick Start

### Option 1: Try It Now (Colab)

Click the badge above to run a complete demo in your browser. No installation needed.

### Option 2: Install Locally

```bash
# Install via pip
pip install git+https://github.com/abhijoysarkar/mmrag.git

# Or clone and install
git clone https://github.com/abhijoysarkar/mmrag.git
cd mmrag
pip install -e .
```

### Option 3: Docker

```bash
docker-compose up mmrag-api
# API available at http://localhost:8000
```

## 5-Minute Demo

```bash
# Generate test image and run lightweight demo
python examples/quickstart_demo.py

# Or with full pipeline (requires GPU)
python examples/quickstart_demo.py --full
```

**What it does:**
1. Creates a test image of a tower
2. Indexes 10 sample documents about landmarks
3. Encodes image with CLIP
4. Performs hybrid text+image retrieval
5. Shows top results

**Output:**
```
Top Retrieved Documents:

1. [Score: 0.8234] doc_0
   The Eiffel Tower is a wrought-iron lattice tower located on the
   Champ de Mars in Paris, France...

✓ Demo completed successfully!
```

## Usage Examples

### Basic Pipeline

```python
from PIL import Image
from mmrag.pipelines.mm_rag_pipeline import MMRAGPipeline

# Initialize pipeline
pipeline = MMRAGPipeline("mmrag/configs/mmrag.yaml")

# Run query
image = Image.open("landmark.jpg")
answer = pipeline(image, "What landmark is this and where is it?")
print(answer)
```

### Retrieval Only (Lightweight)

```python
from mmrag.models.vision_encoder import VisionEncoder
from mmrag.models.retriever import FaissRetriever

# Initialize
encoder = VisionEncoder(device="auto")
retriever = FaissRetriever(dim=512, device="auto")

# Index documents
docs = ["Paris has the Eiffel Tower", "NYC has the Statue of Liberty"]
embeddings = retriever.encode_text(docs)
retriever.add(embeddings, doc_ids=["d1", "d2"], doc_texts=docs)

# Search
image = Image.open("tower.jpg")
image_emb = encoder.encode(image)
results = retriever.hybrid_search("famous tower", image_emb.cpu().numpy(), top_k=3)

for doc_id, score, text in results:
    print(f"[{score:.3f}] {text}")
```

### FastAPI Server

```bash
# Start server
uvicorn mmrag.api.app:app --reload

# Query via curl
curl -X POST "http://localhost:8000/mmrag" \
  -F "image=@tower.jpg" \
  -F "query=What is this structure?"
```

## Configuration

Edit `mmrag/configs/mmrag.yaml`:

```yaml
device: auto  # auto, cuda, or cpu

vision_encoder:
  model_name: openai/clip-vit-base-patch16
  device: auto

retriever:
  dim: 512
  text_encoder_name: all-MiniLM-L6-v2

fusion:
  fusion_type: attention  # attention, gated, or transformer
  num_heads: 8
  dropout: 0.1

generator:
  model_name: meta-llama/Llama-2-7b-hf
  use_lora: true
  lora_r: 8
```

## Benchmarks

Run benchmarks yourself:

```bash
python examples/benchmark.py --device auto
```

**Sample Results (CPU - M2 MacBook):**

| Component | Latency (ms) | Throughput |
|-----------|--------------|------------|
| Vision Encoder (batch=1) | 45.2 | 22.1 imgs/sec |
| Retriever (1K docs) | 12.3 | 81.3 QPS |
| Attention Fusion | 8.7 | 115 samples/sec |
| Gated Fusion | 3.2 | 312 samples/sec |
| Transformer Fusion | 15.4 | 64.9 samples/sec |
| **End-to-End** | **78.5** | **12.7 queries/sec** |

*Note: Without LLM generation. Add ~500-2000ms for LLaMA-7B generation.*

## Features

### Core
- ✅ CLIP-based vision encoding
- ✅ FAISS hybrid (text+image) retrieval
- ✅ Multiple fusion strategies
- ✅ LoRA/AdapterFusion support
- ✅ Auto device detection (CPU/GPU)
- ✅ Robust error handling

### Production
- ✅ FastAPI REST API
- ✅ Docker + docker-compose
- ✅ Batch inference
- ✅ Comprehensive tests (pytest)
- ✅ Type hints throughout
- ✅ Clean, modular architecture

### Research
- ✅ PyTorch Lightning training
- ✅ Multiple fusion strategies
- ✅ Ablation tools
- ✅ Metrics (BLEU, F1, EM)
- ✅ Easy to extend

## Project Structure

```
mmrag/
├── models/           # Core models
│   ├── vision_encoder.py   # CLIP encoder
│   ├── retriever.py         # FAISS retriever
│   ├── fusion.py            # Cross-modal fusion
│   └── generator.py         # LLM generator
├── pipelines/        # End-to-end pipelines
├── api/              # FastAPI server
├── trainers/         # Adapter training
├── evaluation/       # Metrics & ablation
├── tests/            # Unit & integration tests
└── configs/          # YAML configurations

examples/
├── quickstart_demo.py       # 5-min demo
├── MM_RAG_Demo.ipynb        # Colab notebook
├── benchmark.py             # Performance benchmarks
└── sample_documents.txt     # Test data
```

## Advanced Usage

### Custom Fusion Strategy

```python
from torch import nn
from mmrag.models.fusion import CrossModalFusionBlock

class CustomFusion(nn.Module):
    def forward(self, image_emb, text_emb, doc_emb):
        # Your fusion logic
        return fused_embedding

# Use in pipeline
pipeline.fusion = CustomFusion()
```

### Training LoRA Adapters

```bash
python mmrag/main.py train_adapter --config mmrag/configs/mmrag.yaml
```

### Distributed Training

```yaml
# In config
distributed: true
data_parallel: true
```

```bash
torchrun --nproc_per_node=4 mmrag/trainers/adapter_trainer.py
```

## Testing

```bash
# Run all tests
pytest mmrag/tests/

# Run with coverage
pytest --cov=mmrag mmrag/tests/

# Run specific test
pytest mmrag/tests/test_e2e_integration.py -v
```

## Comparison to Alternatives

| Feature | MM-RAG | LlamaIndex | LangChain | Haystack |
|---------|--------|------------|-----------|----------|
| Multimodal Retrieval | ✅ Native | ⚠️ Limited | ⚠️ Limited | ⚠️ Limited |
| Vision Encoding | ✅ CLIP | ❌ | ❌ | ❌ |
| Fusion Strategies | ✅ 3 types | ❌ | ❌ | ❌ |
| LoRA Training | ✅ Built-in | ❌ | ❌ | ❌ |
| Modular Design | ✅ | ⚠️ | ⚠️ | ✅ |
| Production API | ✅ FastAPI | ⚠️ | ⚠️ | ✅ |

**When to use MM-RAG:**
- You have visual data (images, diagrams, charts)
- You need flexible multimodal fusion
- You want to fine-tune for your domain
- You need production deployment

**When to use alternatives:**
- Text-only RAG is sufficient
- You need more LLM integrations
- You prefer higher-level abstractions

## Roadmap

- [ ] Support for more vision encoders (DINOv2, SigLIP)
- [ ] Video understanding support
- [ ] Multi-vector retrieval (ColBERT-style)
- [ ] Reranking module
- [ ] Streaming API responses
- [ ] Web UI demo
- [ ] Pre-trained domain adapters

## Contributing

Contributions welcome! Areas where we'd love help:

- **Encoders**: Add support for more vision/text models
- **Fusion**: Implement new fusion strategies
- **Benchmarks**: Run on different hardware/datasets
- **Docs**: Improve documentation and examples
- **Testing**: Add more test cases

## Citation

If you use MM-RAG in your research:

```bibtex
@software{mmrag2024,
  title={MM-RAG: Multimodal Retrieval-Augmented Generation},
  author={Sarkar, Abhijoy},
  year={2024},
  url={https://github.com/abhijoysarkar/mmrag}
}
```

## License

MIT License - see [LICENSE](LICENSE) file

## Acknowledgments

- CLIP vision encoder from OpenAI
- FAISS from Meta Research
- Sentence Transformers
- HuggingFace Transformers & PEFT

---

**Questions?** Open an issue or discussion on GitHub.
