import torch
from typing import Any, Dict, List, Optional
from omegaconf import OmegaConf
from PIL import Image
from mmrag.models.vision_encoder import VisionEncoder
from mmrag.models.retriever import FaissRetriever
from mmrag.models.fusion import CrossModalFusionBlock
from mmrag.models.generator import Generator
import os

class MMRAGPipeline:
    """
    Multimodal RAG pipeline: image + text -> retrieval -> fusion -> generation.
    Supports single, batch, and distributed/data-parallel inference.
    """
    def __init__(self, config_path: str):
        self.config = OmegaConf.load(config_path)
        self.device = self.config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.vision_encoder = VisionEncoder(**self.config.vision_encoder)
        self.retriever = FaissRetriever(**self.config.retriever)
        self.fusion = CrossModalFusionBlock(**self.config.fusion)
        self.generator = Generator(**self.config.generator)
        self.distributed = getattr(self.config, "distributed", False)
        self.data_parallel = getattr(self.config, "data_parallel", False)
        if self.data_parallel:
            self.vision_encoder = torch.nn.DataParallel(self.vision_encoder)
            self.fusion = torch.nn.DataParallel(self.fusion)
            # Generator is usually large, wrap if needed
            self.generator.model = torch.nn.DataParallel(self.generator.model)
        if self.distributed:
            torch.distributed.init_process_group(backend="nccl")
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))

    def __call__(self, image: Image.Image, text_query: str, top_k: int = 5) -> str:
        # 1. Encode image
        image_emb = self.vision_encoder.encode(image)  # (1, D)
        # 2. Retrieve docs (hybrid)
        docs = self.retriever.hybrid_search(text_query, image_emb.cpu().numpy(), top_k=top_k)
        doc_texts = [d[2] for d in docs if d[2] is not None]
        # 3. Encode text query and docs
        text_emb = torch.tensor(self.retriever.encode_text(text_query), device=self.device)  # (1, D)
        doc_embs = torch.tensor(self.retriever.encode_text(doc_texts), device=self.device) if doc_texts else torch.zeros((1, 1, image_emb.shape[-1]), device=self.device)
        # 4. Prepare for fusion
        image_emb = image_emb.unsqueeze(1) if image_emb.dim() == 2 else image_emb  # (B, 1, D)
        text_emb = text_emb.unsqueeze(1) if text_emb.dim() == 2 else text_emb  # (B, 1, D)
        doc_embs = doc_embs.unsqueeze(0) if doc_embs.dim() == 2 else doc_embs  # (B, K, D)
        # 5. Fuse
        fused = self.fusion(image_emb, text_emb, doc_embs)  # (B, 1, D)
        # 6. Generate
        prompt = text_query + "\n" + "\n".join(doc_texts)
        answer = self.generator.generate(prompt, fused_emb=fused.squeeze(1))
        return answer

    def batch_call(self, images: List[Image.Image], text_queries: List[str], top_k: int = 5) -> List[str]:
        image_embs = self.vision_encoder.encode(images)  # (B, D)
        answers = []
        for i, (img_emb, query) in enumerate(zip(image_embs, text_queries)):
            docs = self.retriever.hybrid_search(query, img_emb.unsqueeze(0).cpu().numpy(), top_k=top_k)
            doc_texts = [d[2] for d in docs if d[2] is not None]
            text_emb = torch.tensor(self.retriever.encode_text(query), device=self.device)
            doc_embs = torch.tensor(self.retriever.encode_text(doc_texts), device=self.device) if doc_texts else torch.zeros((1, 1, img_emb.shape[-1]), device=self.device)
            img_emb = img_emb.unsqueeze(0).unsqueeze(1)  # (1, 1, D)
            text_emb = text_emb.unsqueeze(1)
            doc_embs = doc_embs.unsqueeze(0) if doc_embs.dim() == 2 else doc_embs
            fused = self.fusion(img_emb, text_emb, doc_embs)
            prompt = query + "\n" + "\n".join(doc_texts)
            answer = self.generator.generate(prompt, fused_emb=fused.squeeze(1))
            answers.append(answer)
        return answers 