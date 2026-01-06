"""
Utility classes for logging metrics and tracking progress.
"""

from pathlib import Path
from datetime import datetime


class MetricsLogger:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_text_report(self, metrics: dict, filename: str):
        path = self.output_dir / filename

        with open(path, "w") as f:
            f.write(f"Model: {metrics['model_name']}\n")
            f.write(f"Timestamp: {datetime.now()}\n\n")

            f.write("Inference Time (ms):\n")
            for k, v in metrics["inference_time_ms"].items():
                f.write(f"  {k}: {v:.2f}\n")

            f.write(f"\nModel Size (MB): {metrics['model_size_mb']}\n")
            f.write(f"Memory Usage (MB): {metrics['memory_usage_mb']}\n")

            f.write("\nAccuracy:\n")
            f.write(f"  Top-1: {metrics['accuracy']['top_1']:.2f}%\n")
            f.write(f"  Top-5: {metrics['accuracy']['top_5']:.2f}%\n")

            f.write(f"\nThroughput (images/sec): {metrics['throughput_ips']}\n")


class ProgressTracker:
    def __init__(self, total_steps: int, description: str):
        self.total_steps = total_steps
        self.current_step = 0
        self.description = description

    def start(self):
        print(self.description)

    def update(self, status: str = ""):
        self.current_step += 1
        percent = (self.current_step / self.total_steps) * 100
        print(f"[{percent:.1f}%] {status}")

    def finish(self):
        print("Pipeline completed.")
