# My-Custom-CUDA-Operator-Library
收集了一些经过优化的常用算子以及一个简单版本的 Flash-Attention
未深度优化，仅供学习使用，后续会持续更新 优化 & 新算子 😋

## 🌟 Features
- Flash Attention implementation (CUDA kernel)
- SGEMM (Single-precision General Matrix Multiplication)
- Elementwise operations (add/relu/sigmoid)
- Reduce operations (sum/softmax)

## 📋 Dependencies
- Python >= 3.8
- PyTorch >= 2.0
- CUDA Toolkit >= 11.7 (match your PyTorch CUDA version)
- GCC >= 9.0 (Linux)
