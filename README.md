Model Optimization Techniques and Trade-offs
Overview

This project focuses on optimizing a pre-trained deep learning model for edge-device deployment. MobileNetV2, pre-trained on ImageNet, was selected due to its lightweight architecture and efficiency on resource-constrained platforms.

Baseline Evaluation

The original model was benchmarked on CPU using inference latency, model size, memory usage, and accuracy. CIFAR-10 was used as a proxy dataset for performance evaluation. Accuracy is not representative due to the mismatch between ImageNet training labels and CIFAR-10 evaluation labels and is included only for completeness.

Applied Optimizations

PyTorch dynamic INT8 quantization was applied to reduce model size and improve inference speed without retraining. Additionally, the model was converted to ONNX format to enable cross-platform deployment and compatibility with ONNX Runtime. ONNX dynamic quantization was intentionally avoided due to known shape inference limitations in MobileNetV2.

Trade-offs and Recommendation

The optimized model achieved reduced size and improved latency, making it suitable for edge deployment. The recommended approach is to use the INT8 PyTorch model for CPU inference and the FP32 ONNX model for portable deployment.
