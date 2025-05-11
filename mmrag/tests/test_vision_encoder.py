import torch
from PIL import Image
import numpy as np
import pytest
from mmrag.models.vision_encoder import VisionEncoder

def test_vision_encoder_output_shape():
    encoder = VisionEncoder()
    img = Image.fromarray(np.uint8(np.random.rand(224, 224, 3) * 255))
    emb = encoder.encode(img)
    assert emb.shape[0] == 1
    assert emb.shape[1] == encoder.model.config.projection_dim 