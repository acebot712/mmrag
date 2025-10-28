# Contributing to MM-RAG

Thank you for your interest in contributing to MM-RAG! This document provides guidelines and instructions for contributing.

## Ways to Contribute

### 1. Report Bugs
- Use GitHub Issues
- Include minimal reproducible example
- Specify Python version, OS, and hardware
- Provide error messages and stack traces

### 2. Suggest Features
- Open a GitHub Issue with the "enhancement" label
- Describe the use case and expected behavior
- Consider submitting a PR if you can implement it

### 3. Improve Documentation
- Fix typos or clarify confusing sections
- Add examples or tutorials
- Improve API documentation

### 4. Submit Code
- Bug fixes
- New features
- Performance improvements
- Additional tests

## Development Setup

```bash
# Clone repository
git clone https://github.com/abhijoysarkar/mmrag.git
cd mmrag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install in development mode with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks (optional but recommended)
pip install pre-commit
pre-commit install
```

## Code Style

We follow standard Python conventions:

- **PEP 8** for code style
- **Type hints** for function signatures
- **Docstrings** for public APIs (Google style)
- **Black** for code formatting (line length 100)

Example:

```python
from typing import List, Optional
import torch

def encode_images(
    images: List[Image.Image],
    device: Optional[str] = None
) -> torch.Tensor:
    """
    Encode a batch of images using CLIP.

    Args:
        images: List of PIL Images to encode
        device: Device to use ('cuda', 'cpu', or 'auto')

    Returns:
        Tensor of shape (batch_size, embedding_dim)

    Raises:
        ValueError: If images list is empty
    """
    if not images:
        raise ValueError("Images list cannot be empty")
    # Implementation...
```

## Testing

### Running Tests

```bash
# Run all tests
pytest mmrag/tests/

# Run with coverage
pytest --cov=mmrag mmrag/tests/

# Run specific test file
pytest mmrag/tests/test_vision_encoder.py -v

# Run slow tests (marked with @pytest.mark.slow)
pytest -m slow
```

### Writing Tests

- Place tests in `mmrag/tests/`
- Name test files `test_*.py`
- Use descriptive test names: `test_vision_encoder_handles_batch_input`
- Use fixtures for common setup
- Mark slow tests with `@pytest.mark.slow`

Example:

```python
import pytest
from mmrag.models.vision_encoder import VisionEncoder

@pytest.fixture
def encoder():
    return VisionEncoder(device="cpu")

def test_encoder_outputs_correct_shape(encoder):
    from PIL import Image
    img = Image.new('RGB', (224, 224))
    emb = encoder.encode(img)
    assert emb.shape == (1, 512)

@pytest.mark.slow
def test_encoder_batch_processing(encoder):
    # Test that takes > 1 second
    pass
```

## Pull Request Process

1. **Fork the repository**

2. **Create a feature branch**
   ```bash
   git checkout -b feature/my-new-feature
   ```

3. **Make your changes**
   - Write code
   - Add tests
   - Update documentation

4. **Run tests and linting**
   ```bash
   pytest mmrag/tests/
   black mmrag/ --check
   flake8 mmrag/
   mypy mmrag/
   ```

5. **Commit with clear messages**
   ```bash
   git commit -m "Add support for DINOv2 vision encoder"
   ```

6. **Push to your fork**
   ```bash
   git push origin feature/my-new-feature
   ```

7. **Open a Pull Request**
   - Describe what changed and why
   - Reference related issues
   - Include screenshots/demos if applicable

### PR Checklist

- [ ] Tests pass locally
- [ ] New code has tests
- [ ] Documentation is updated
- [ ] Code follows style guidelines
- [ ] Commit messages are clear
- [ ] No merge conflicts

## Areas We'd Love Help With

### High Priority
- **More Vision Encoders**: DINOv2, SigLIP, ViT variants
- **Reranking Module**: Cross-encoder reranking for better retrieval
- **Video Support**: Extend to video understanding
- **Performance**: Optimize inference speed, memory usage

### Medium Priority
- **Multi-vector Retrieval**: ColBERT-style dense retrieval
- **Better Fusion**: Novel fusion architectures
- **More Datasets**: Benchmarks on standard datasets
- **Web UI**: Gradio/Streamlit interface

### Low Priority (but still welcome!)
- **Documentation**: More examples, tutorials
- **Tests**: Increase coverage
- **Benchmarks**: Profile on different hardware
- **CI/CD**: GitHub Actions workflows

## Project Structure

```
mmrag/
├── models/              # Core models
│   ├── vision_encoder.py
│   ├── retriever.py
│   ├── fusion.py
│   └── generator.py
├── pipelines/           # End-to-end pipelines
├── api/                 # FastAPI server
├── trainers/            # Training code
├── evaluation/          # Metrics and evaluation
├── tests/               # Test suite
└── configs/             # Configuration files

examples/                # Example scripts and notebooks
docs/                    # Additional documentation
```

## Adding New Components

### New Vision Encoder

1. Create a new encoder class in `mmrag/models/vision_encoder.py` or a new file
2. Implement the `encode()` method with consistent interface
3. Add tests in `mmrag/tests/test_vision_encoder.py`
4. Update documentation

```python
class CustomVisionEncoder(nn.Module):
    def __init__(self, model_name: str, device: str = "auto"):
        # Initialize model
        pass

    @torch.no_grad()
    def encode(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        # Return (batch_size, embedding_dim) tensor
        pass
```

### New Fusion Strategy

1. Add to `CrossModalFusionBlock` in `mmrag/models/fusion.py`
2. Implement forward pass
3. Add tests
4. Update config options

```python
elif fusion_type == 'my_fusion':
    self.my_fusion_layer = nn.Sequential(...)
```

## Code Review Process

- Maintainers will review PRs within 1-2 weeks
- Address feedback in additional commits
- Once approved, maintainer will merge
- PRs may be closed if inactive for 30+ days

## Questions?

- Open a GitHub Issue with the "question" label
- Start a GitHub Discussion
- Check existing issues and documentation first

## Code of Conduct

Be respectful and constructive in all interactions. We aim to maintain a welcoming community for everyone.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for contributing to MM-RAG!
