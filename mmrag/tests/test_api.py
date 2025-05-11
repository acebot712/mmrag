import pytest
from fastapi.testclient import TestClient
from mmrag.api.app import app
from PIL import Image
import io

client = TestClient(app)

def test_mmrag_endpoint():
    img = Image.new("RGB", (224, 224))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    response = client.post(
        "/mmrag",
        files={"image": ("test.png", buf, "image/png")},
        data={"query": "What is in the image?"}
    )
    assert response.status_code == 200
    assert "answer" in response.json() 