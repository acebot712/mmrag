#!/usr/bin/env python3
"""
Benchmarking script for MM-RAG.
Compares different fusion strategies and provides performance metrics.
"""

import time
import torch
import numpy as np
from PIL import Image
from typing import List, Dict, Any
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from mmrag.models.vision_encoder import VisionEncoder
from mmrag.models.retriever import FaissRetriever
from mmrag.models.fusion import CrossModalFusionBlock
from mmrag.evaluation.metrics import compute_retrieval_metrics


class MMRAGBenchmark:
    """Benchmark MM-RAG components."""

    def __init__(self, device: str = "auto"):
        self.device = device
        self.results = {}

    def benchmark_vision_encoder(self, batch_sizes: List[int] = [1, 4, 8, 16]):
        """Benchmark vision encoder throughput."""
        print("\n" + "=" * 80)
        print("BENCHMARK: Vision Encoder")
        print("=" * 80)

        encoder = VisionEncoder(device=self.device)
        results = {}

        for batch_size in batch_sizes:
            # Create dummy images
            images = [Image.new('RGB', (224, 224), color='blue') for _ in range(batch_size)]

            # Warmup
            for _ in range(3):
                _ = encoder.encode(images)

            # Benchmark
            times = []
            for _ in range(10):
                start = time.time()
                _ = encoder.encode(images)
                if self.device != "cpu":
                    torch.cuda.synchronize()
                times.append(time.time() - start)

            avg_time = np.mean(times)
            throughput = batch_size / avg_time

            results[f"batch_{batch_size}"] = {
                "avg_time_ms": avg_time * 1000,
                "throughput_imgs_per_sec": throughput,
                "std_ms": np.std(times) * 1000
            }

            print(f"Batch size {batch_size:2d}: {avg_time*1000:6.2f}ms | "
                  f"{throughput:6.2f} imgs/sec | std: {np.std(times)*1000:.2f}ms")

        self.results['vision_encoder'] = results
        return results

    def benchmark_retriever(self, num_docs: int = 1000, query_count: int = 100):
        """Benchmark retriever performance."""
        print("\n" + "=" * 80)
        print(f"BENCHMARK: Retriever ({num_docs} docs)")
        print("=" * 80)

        retriever = FaissRetriever(dim=512, device=self.device)

        # Index random documents
        print(f"Indexing {num_docs} documents...")
        doc_embs = np.random.randn(num_docs, 512).astype(np.float32)
        doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
        doc_ids = [f"doc_{i}" for i in range(num_docs)]
        doc_texts = [f"Document {i} content" for i in range(num_docs)]

        start = time.time()
        retriever.add(doc_embs, doc_ids, doc_texts)
        index_time = time.time() - start

        print(f"✓ Indexing time: {index_time*1000:.2f}ms")

        # Benchmark search
        queries = [f"query {i}" for i in range(query_count)]
        search_times = []

        print(f"Running {query_count} queries...")
        for query in queries:
            start = time.time()
            _ = retriever.hybrid_search(query, top_k=10)
            search_times.append(time.time() - start)

        avg_search = np.mean(search_times)
        p95_search = np.percentile(search_times, 95)
        p99_search = np.percentile(search_times, 99)

        results = {
            "num_docs": num_docs,
            "index_time_ms": index_time * 1000,
            "avg_search_ms": avg_search * 1000,
            "p95_search_ms": p95_search * 1000,
            "p99_search_ms": p99_search * 1000,
            "qps": 1.0 / avg_search
        }

        print(f"Search latency (avg): {avg_search*1000:.2f}ms")
        print(f"Search latency (p95): {p95_search*1000:.2f}ms")
        print(f"Search latency (p99): {p99_search*1000:.2f}ms")
        print(f"Queries per second: {results['qps']:.2f}")

        self.results['retriever'] = results
        return results

    def benchmark_fusion_strategies(self, batch_size: int = 8):
        """Compare different fusion strategies."""
        print("\n" + "=" * 80)
        print("BENCHMARK: Fusion Strategies")
        print("=" * 80)

        fusion_types = ['attention', 'gated', 'transformer']
        embed_dim = 512
        results = {}

        # Create dummy inputs
        image_emb = torch.randn(batch_size, 1, embed_dim)
        text_emb = torch.randn(batch_size, 1, embed_dim)
        doc_emb = torch.randn(batch_size, 5, embed_dim)

        if self.device != "cpu" and torch.cuda.is_available():
            image_emb = image_emb.cuda()
            text_emb = text_emb.cuda()
            doc_emb = doc_emb.cuda()

        for fusion_type in fusion_types:
            print(f"\n{fusion_type.capitalize()} Fusion:")

            fusion = CrossModalFusionBlock(
                embed_dim=embed_dim,
                num_heads=8,
                dropout=0.1,
                fusion_type=fusion_type
            )

            if self.device != "cpu" and torch.cuda.is_available():
                fusion = fusion.cuda()

            # Warmup
            with torch.no_grad():
                for _ in range(10):
                    _ = fusion(image_emb, text_emb, doc_emb)

            # Benchmark
            times = []
            with torch.no_grad():
                for _ in range(100):
                    start = time.time()
                    output = fusion(image_emb, text_emb, doc_emb)
                    if self.device != "cpu" and torch.cuda.is_available():
                        torch.cuda.synchronize()
                    times.append(time.time() - start)

            avg_time = np.mean(times)
            throughput = batch_size / avg_time

            # Count parameters
            param_count = sum(p.numel() for p in fusion.parameters())

            results[fusion_type] = {
                "avg_time_ms": avg_time * 1000,
                "throughput_samples_per_sec": throughput,
                "std_ms": np.std(times) * 1000,
                "parameters": param_count
            }

            print(f"  Latency: {avg_time*1000:.2f}ms ± {np.std(times)*1000:.2f}ms")
            print(f"  Throughput: {throughput:.2f} samples/sec")
            print(f"  Parameters: {param_count:,}")

        self.results['fusion'] = results
        return results

    def benchmark_end_to_end(self, num_queries: int = 10):
        """Benchmark full pipeline (retrieval only, no LLM)."""
        print("\n" + "=" * 80)
        print("BENCHMARK: End-to-End Pipeline")
        print("=" * 80)

        # Initialize components
        vision_encoder = VisionEncoder(device=self.device)
        retriever = FaissRetriever(dim=512, device=self.device)
        fusion = CrossModalFusionBlock(embed_dim=512, num_heads=8)

        if self.device != "cpu" and torch.cuda.is_available():
            fusion = fusion.cuda()

        # Index dummy documents
        num_docs = 100
        doc_embs = np.random.randn(num_docs, 512).astype(np.float32)
        doc_embs = doc_embs / np.linalg.norm(doc_embs, axis=1, keepdims=True)
        doc_ids = [f"doc_{i}" for i in range(num_docs)]
        doc_texts = [f"Document {i}" for i in range(num_docs)]

        # Pad to 512
        doc_embs_384 = doc_embs[:, :384]
        doc_embs = np.pad(doc_embs_384, ((0, 0), (0, 128)), mode='constant')
        retriever.add(doc_embs, doc_ids, doc_texts)

        # Create test image
        test_image = Image.new('RGB', (224, 224), color='blue')

        # Benchmark full pipeline
        times = {
            'vision_encoding': [],
            'retrieval': [],
            'fusion': [],
            'total': []
        }

        print(f"Running {num_queries} end-to-end queries...")

        for i in range(num_queries):
            query = f"test query {i}"
            total_start = time.time()

            # Vision encoding
            start = time.time()
            image_emb = vision_encoder.encode(test_image)
            times['vision_encoding'].append(time.time() - start)

            # Retrieval
            start = time.time()
            results = retriever.hybrid_search(query, image_emb.cpu().numpy(), top_k=5)
            times['retrieval'].append(time.time() - start)

            # Fusion
            text_emb = torch.tensor(retriever.encode_text(query))
            doc_texts_retrieved = [r[2] for r in results]
            doc_embs_raw = retriever.encode_text(doc_texts_retrieved)
            doc_embs_tensor = np.pad(doc_embs_raw, ((0, 0), (0, 128)), mode='constant')
            doc_embs_tensor = torch.tensor(doc_embs_tensor)

            if image_emb.dim() == 2:
                image_emb = image_emb.unsqueeze(1)
            if text_emb.dim() == 2:
                text_emb = text_emb.unsqueeze(1)
            if doc_embs_tensor.dim() == 2:
                doc_embs_tensor = doc_embs_tensor.unsqueeze(0)

            start = time.time()
            with torch.no_grad():
                _ = fusion(image_emb, text_emb, doc_embs_tensor)
            times['fusion'].append(time.time() - start)

            times['total'].append(time.time() - total_start)

        # Compute statistics
        results = {}
        for stage, stage_times in times.items():
            results[stage] = {
                "avg_ms": np.mean(stage_times) * 1000,
                "p95_ms": np.percentile(stage_times, 95) * 1000,
                "p99_ms": np.percentile(stage_times, 99) * 1000,
                "std_ms": np.std(stage_times) * 1000
            }

            print(f"\n{stage.replace('_', ' ').title()}:")
            print(f"  Avg: {results[stage]['avg_ms']:.2f}ms")
            print(f"  P95: {results[stage]['p95_ms']:.2f}ms")
            print(f"  P99: {results[stage]['p99_ms']:.2f}ms")

        self.results['end_to_end'] = results
        return results

    def save_results(self, output_path: str = "benchmark_results.json"):
        """Save benchmark results to JSON."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")

    def print_summary(self):
        """Print benchmark summary."""
        print("\n" + "=" * 80)
        print("BENCHMARK SUMMARY")
        print("=" * 80)

        if 'vision_encoder' in self.results:
            ve = self.results['vision_encoder']['batch_1']
            print(f"\nVision Encoder (batch=1): {ve['avg_time_ms']:.2f}ms")

        if 'retriever' in self.results:
            ret = self.results['retriever']
            print(f"Retriever ({ret['num_docs']} docs): {ret['avg_search_ms']:.2f}ms avg, "
                  f"{ret['qps']:.1f} QPS")

        if 'fusion' in self.results:
            print("\nFusion Strategies:")
            for name, metrics in self.results['fusion'].items():
                print(f"  {name.capitalize()}: {metrics['avg_time_ms']:.2f}ms "
                      f"({metrics['parameters']:,} params)")

        if 'end_to_end' in self.results:
            e2e = self.results['end_to_end']['total']
            print(f"\nEnd-to-End Pipeline: {e2e['avg_ms']:.2f}ms (p95: {e2e['p95_ms']:.2f}ms)")

        print("=" * 80)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark MM-RAG")
    parser.add_argument("--device", default="auto", help="Device: auto, cpu, or cuda")
    parser.add_argument("--output", default="benchmark_results.json", help="Output file")
    parser.add_argument("--skip-vision", action="store_true", help="Skip vision encoder benchmark")
    parser.add_argument("--skip-retrieval", action="store_true", help="Skip retriever benchmark")
    parser.add_argument("--skip-fusion", action="store_true", help="Skip fusion benchmark")
    parser.add_argument("--skip-e2e", action="store_true", help="Skip end-to-end benchmark")
    args = parser.parse_args()

    print("=" * 80)
    print("MM-RAG BENCHMARK SUITE")
    print("=" * 80)
    print(f"Device: {args.device}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print("=" * 80)

    benchmark = MMRAGBenchmark(device=args.device)

    if not args.skip_vision:
        benchmark.benchmark_vision_encoder()

    if not args.skip_retrieval:
        benchmark.benchmark_retriever()

    if not args.skip_fusion:
        benchmark.benchmark_fusion_strategies()

    if not args.skip_e2e:
        benchmark.benchmark_end_to_end()

    benchmark.print_summary()
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()
