import torch
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path

from config import Config
from utils import MetricsLogger, ProgressTracker
from benchmark import ModelBenchmark
from model_optimization import ModelOptimizer


def load_pretrained_model():
    """Load MobileNetV2 pre-trained on ImageNet."""
    model = torchvision.models.mobilenet_v2(
        weights=torchvision.models.MobileNet_V2_Weights.DEFAULT
    )
    model.eval()
    return model


def load_test_dataset(num_samples):
    """
    Load CIFAR-10 dataset for benchmarking.

    CIFAR-10 is used as a proxy dataset to measure inference performance.
    Accuracy values are not representative due to label mismatch.
    """
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = datasets.CIFAR10(
        root=Config.DATA_DIR,
        train=False,
        download=True,
        transform=transform
    )

    subset = Subset(dataset, range(num_samples))
    dataloader = DataLoader(subset, batch_size=32, shuffle=False)
    sample_images = [subset[i][0] for i in range(min(100, len(subset)))]

    return dataloader, sample_images


def print_summary(original, optimized):
    """Print a clean comparison summary to the console."""
    orig_time = original["inference_time_ms"]["mean"]
    opt_time = optimized["inference_time_ms"]["mean"]

    orig_size = original["model_size_mb"]
    opt_size = optimized["model_size_mb"]

    speed_improvement = ((orig_time - opt_time) / orig_time) * 100
    size_reduction = ((orig_size - opt_size) / orig_size) * 100

    print("\n" + "=" * 60)
    print("MODEL OPTIMIZATION SUMMARY")
    print("=" * 60)

    print("Inference Time (ms per image)")
    print(f"  Original : {orig_time:.2f}")
    print(f"  Optimized: {opt_time:.2f}")
    print(f"  Improvement: {speed_improvement:.1f}%\n")

    print("Model Size (MB)")
    print(f"  Original : {orig_size:.2f}")
    print(f"  Optimized: {opt_size:.2f}")
    print(f"  Reduction: {size_reduction:.1f}%\n")

    print("Accuracy")
    print("  Accuracy values are not representative due to")
    print("  ImageNet (training) → CIFAR-10 (evaluation) mismatch.\n")

    print("Edge Deployment Recommendation")
    print("  - Use INT8 PyTorch model for CPU-based edge inference")
    print("  - Use FP32 ONNX model for portable, cross-platform deployment")

    print("=" * 60 + "\n")


def main():
    """Main execution pipeline."""
    Config.create_directories()
    Config.print_config()

    logger = MetricsLogger(Config.RESULTS_DIR)
    progress = ProgressTracker(total_steps=6, description="Optimization Pipeline")
    progress.start()

    # Step 1: Load original model
    progress.update(status="Loading original model")
    original_model = load_pretrained_model()

    # Step 2: Load test data
    progress.update(status="Loading test data")
    test_loader, test_images = load_test_dataset(Config.NUM_TEST_SAMPLES)

    # Step 3: Benchmark original model
    progress.update(status="Benchmarking original model")
    baseline_benchmark = ModelBenchmark(original_model)

    baseline_results = baseline_benchmark.run_full_benchmark(
        test_loader=test_loader,
        test_images=test_images,
        save_path=Config.ORIGINAL_MODEL_DIR / "mobilenet_v2_original.pth",
        model_name="mobilenet_v2_original"
    )

    logger.save_text_report(baseline_results, "original_metrics.txt")

    # Step 4: Apply optimizations
    progress.update(status="Applying optimizations")
    optimizer = ModelOptimizer(original_model)

    # Optimization 1: PyTorch INT8 dynamic quantization
    quantized_model = optimizer.apply_dynamic_quantization()

    # Optimization 2: ONNX conversion (FP32 model)
    onnx_path = Config.ONNX_MODEL_DIR / "mobilenet_v2.onnx"
    optimizer.convert_to_onnx(
        model=original_model,
        save_path=onnx_path,
        input_shape=(1, 3, 224, 224),
        opset_version=18
    )

    # Step 5: Benchmark optimized model
    progress.update(status="Benchmarking optimized model")
    optimized_benchmark = ModelBenchmark(quantized_model)

    optimized_results = optimized_benchmark.run_full_benchmark(
        test_loader=test_loader,
        test_images=test_images,
        save_path=Config.QUANTIZED_MODEL_DIR / "mobilenet_v2_dynamic_int8.pth",
        model_name="mobilenet_v2_dynamic_int8"
    )

    logger.save_text_report(optimized_results, "optimized_metrics.txt")

    # Step 6: Final summary
    progress.update(status="Finalizing")
    progress.finish()

    print_summary(baseline_results, optimized_results)

    print("Generated artifacts:")
    print("  - results/original_metrics.txt")
    print("  - results/optimized_metrics.txt")
    print("  - models/onnx/mobilenet_v2.onnx")


if __name__ == "__main__":
    main()
