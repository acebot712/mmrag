import torch
from torch import nn
from typing import List, Optional, Union
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

class VisionEncoder(nn.Module):
    """
    Vision encoder using CLIP from HuggingFace Transformers.
    Encodes images into dense embeddings for retrieval or fusion.
    """
    def __init__(self, model_name: str = 'openai/clip-vit-base-patch16', device: Optional[Union[str, torch.device]] = None):
        super().__init__()
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.eval()

    @torch.no_grad()
    def encode(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Encode a single image or a batch of images into CLIP embeddings.
        Args:
            images: PIL Image or list of PIL Images
        Returns:
            torch.Tensor: Image embeddings of shape (batch_size, embed_dim)
        """
        if isinstance(images, Image.Image):
            images = [images]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        image_features = self.model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    def forward(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        return self.encode(images) 