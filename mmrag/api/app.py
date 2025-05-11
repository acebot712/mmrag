from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from PIL import Image
import io
from mmrag.pipelines.mm_rag_pipeline import MMRAGPipeline

app = FastAPI(title="MM-RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[MMRAGPipeline] = None

class MMRAgResponse(BaseModel):
    answer: str

@app.on_event("startup")
def load_pipeline():
    global pipeline
    pipeline = MMRAGPipeline("mmrag/configs/mmrag.yaml")

@app.post("/mmrag", response_model=MMRAgResponse)
def mmrag_endpoint(
    image: UploadFile = File(...),
    query: str = Form(...)
):
    """
    Run MM-RAG on an image and text query.
    """
    img_bytes = image.file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    answer = pipeline(img, query)
    return MMRAgResponse(answer=answer) 