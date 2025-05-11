from typing import List
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from sklearn.metrics import f1_score
import numpy as np

def compute_bleu(reference: str, hypothesis: str) -> float:
    """
    Compute BLEU score for a single reference and hypothesis.
    """
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)

def compute_em(reference: str, hypothesis: str) -> float:
    """
    Compute Exact Match (EM) score.
    """
    return float(reference.strip() == hypothesis.strip())

def compute_latency(latencies: List[float]) -> float:
    """
    Compute average latency from a list of times.
    """
    return float(np.mean(latencies))

def compute_f1(reference: str, hypothesis: str) -> float:
    """
    Compute F1 score for text generation (token-level, micro average).
    """
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()
    all_tokens = list(set(ref_tokens + hyp_tokens))
    ref_vec = [1 if t in ref_tokens else 0 for t in all_tokens]
    hyp_vec = [1 if t in hyp_tokens else 0 for t in all_tokens]
    return f1_score(ref_vec, hyp_vec, average='micro') 