"""
Central configuration for the ML model optimization project.
"""

import torch
from pathlib import Path


class Config:
    # Model settings
    MODEL_NAME = "mobilenet_v2"
    INPUT_SIZE = (224, 224)
    NUM_CLASSES = 1000

    # Benchmark settings
    BATCH_SIZE = 32
    NUM_ITERATIONS = 100
    NUM_WARMUP = 10
    NUM_TEST_SAMPLES = 1000

    # Quantization
    QUANTIZATION_DTYPE = torch.qint8

    # ONNX
    ONNX_OPSET_VERSION = 18

    # Device
    DEVICE = "cpu"
    NUM_THREADS = 4

    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODELS_DIR = BASE_DIR / "models"
    RESULTS_DIR = BASE_DIR / "results"

    ORIGINAL_MODEL_DIR = MODELS_DIR / "original"
    QUANTIZED_MODEL_DIR = MODELS_DIR / "quantized"
    ONNX_MODEL_DIR = MODELS_DIR / "onnx"

    @classmethod
    def create_directories(cls):
        directories = [
            cls.DATA_DIR,
            cls.MODELS_DIR,
            cls.RESULTS_DIR,
            cls.ORIGINAL_MODEL_DIR,
            cls.QUANTIZED_MODEL_DIR,
            cls.ONNX_MODEL_DIR,
        ]
        for d in directories:
            d.mkdir(parents=True, exist_ok=True)

    @classmethod
    def print_config(cls):
        print(f"Model: {cls.MODEL_NAME}")
        print(f"Input Size: {cls.INPUT_SIZE}")
        print(f"Device: {cls.DEVICE}")
        print(f"Quantization: {cls.QUANTIZATION_DTYPE}")
        print(f"Benchmark Iterations: {cls.NUM_ITERATIONS}")
        print(f"Test Samples: {cls.NUM_TEST_SAMPLES}")


torch.set_num_threads(Config.NUM_THREADS)
