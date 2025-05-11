from typing import List, Dict, Any
from PIL import Image
from mmrag.pipelines.mm_rag_pipeline import MMRAGPipeline
from mmrag.evaluation.metrics import compute_bleu, compute_em, compute_f1
import time

class AblationRunner:
    """
    Runs ablation experiments: vision-only, retrieval-only, and combined.
    Returns BLEU, EM, F1, and latency.
    """
    def __init__(self, pipeline: MMRAGPipeline):
        self.pipeline = pipeline

    def run(
        self,
        dataset: List[Dict[str, Any]],
        mode: str = "combined"
    ) -> Dict[str, float]:
        """
        Args:
            dataset: List of dicts with 'image', 'query', 'reference' keys
            mode: 'vision', 'retrieval', or 'combined'
        Returns:
            Dict with BLEU, EM, F1, and latency
        """
        bleu_scores = []
        em_scores = []
        f1_scores = []
        latencies = []
        for sample in dataset:
            image = sample["image"]
            query = sample["query"]
            reference = sample["reference"]
            start = time.time()
            if mode == "vision":
                # Only use image, ignore retrieval
                answer = self.pipeline.generator.generate(query, fused_emb=self.pipeline.vision_encoder.encode(image))
            elif mode == "retrieval":
                # Only use retrieval, ignore image
                docs = self.pipeline.retriever.hybrid_search(query, image_emb=None, top_k=5)
                doc_texts = [d[2] for d in docs if d[2] is not None]
                prompt = query + "\n" + "\n".join(doc_texts)
                answer = self.pipeline.generator.generate(prompt)
            else:
                # Combined
                answer = self.pipeline(image, query)
            latency = time.time() - start
            bleu_scores.append(compute_bleu(reference, answer))
            em_scores.append(compute_em(reference, answer))
            f1_scores.append(compute_f1(reference, answer))
            latencies.append(latency)
        return {
            "BLEU": sum(bleu_scores) / len(bleu_scores),
            "EM": sum(em_scores) / len(em_scores),
            "F1": sum(f1_scores) / len(f1_scores),
            "Latency": sum(latencies) / len(latencies)
        } 