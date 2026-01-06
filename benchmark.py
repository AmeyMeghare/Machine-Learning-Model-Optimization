"""
Model benchmarking utilities.
"""

import time
import os
import psutil
import torch
import numpy as np
from config import Config


class ModelBenchmark:
    def __init__(self, model):
        self.model = model.to(Config.DEVICE)
        self.model.eval()

    def measure_inference_time(self, images):
        times = []

        for _ in range(Config.NUM_WARMUP):
            with torch.no_grad():
                self.model(images[0].unsqueeze(0))

        for img in images[:Config.NUM_ITERATIONS]:
            start = time.perf_counter()
            with torch.no_grad():
                self.model(img.unsqueeze(0))
            end = time.perf_counter()
            times.append((end - start) * 1000)

        arr = np.array(times)
        return {
            "mean": arr.mean(),
            "std": arr.std(),
            "min": arr.min(),
            "max": arr.max(),
            "median": np.median(arr),
        }

    def measure_model_size(self, save_path):
        torch.save(self.model.state_dict(), save_path)
        return os.path.getsize(save_path) / (1024 * 1024)

    def measure_memory_usage(self, image):
        process = psutil.Process()
        before = process.memory_info().rss / (1024 * 1024)

        with torch.no_grad():
            self.model(image.unsqueeze(0))

        after = process.memory_info().rss / (1024 * 1024)
        return max(0.0, after - before)

    def measure_accuracy(self, dataloader):
        correct_top1 = 0
        correct_top5 = 0
        total = 0

        with torch.no_grad():
            for images, labels in dataloader:
                outputs = self.model(images)
                _, preds = outputs.topk(5, dim=1)

                correct_top1 += (preds[:, 0] == labels).sum().item()
                correct_top5 += (preds == labels.unsqueeze(1)).any(dim=1).sum().item()
                total += labels.size(0)

        return {
            "top_1": (correct_top1 / total) * 100,
            "top_5": (correct_top5 / total) * 100,
        }

    def measure_throughput(self, images, duration=5.0):
        count = 0
        start = time.time()

        while time.time() - start < duration:
            with torch.no_grad():
                self.model(images[count % len(images)].unsqueeze(0))
            count += 1

        return count / duration

    def run_full_benchmark(self, test_loader, test_images, save_path, model_name):
        inference = self.measure_inference_time(test_images)
        size = self.measure_model_size(save_path)
        memory = self.measure_memory_usage(test_images[0])
        accuracy = self.measure_accuracy(test_loader)
        throughput = self.measure_throughput(test_images)

        return {
            "model_name": model_name,
            "inference_time_ms": inference,
            "model_size_mb": round(size, 2),
            "memory_usage_mb": round(memory, 2),
            "accuracy": accuracy,
            "throughput_ips": round(throughput, 2),
        }
