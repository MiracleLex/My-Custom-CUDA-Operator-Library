# My-Custom-CUDA-Operator-Library
收集了一些常用算子以及一个简单版本的 Flash-Attention

部分简单算子经过初步优化，性能可以达到官方实现的 75% ~ 90%，且具备反向传播能力

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

## 🛠️ Installation
### 1. Clone the repository
```bash
git clone https://github.com/MiracleLex/My-Custom-CUDA-Operator-Library.git
```
### 2. Build the CUDA operators
```bash
cd My-Custom-CUDA-Operator-Library
python setup.py build_ext --inplace
```
## 🚀 Usage Example
### 1. Test Flash-Attention
```python
import torch
import MyCustomCUDAOperatorLibrary._C as ops

# 随机生成 Q/K/V (B, H, N, d)
batch, heads, seq_len, dim = 2, 8, 1024, 64
q = torch.randn(batch, heads, seq_len, dim).cuda().float()
k = torch.randn(batch, heads, seq_len, dim).cuda().float()
v = torch.randn(batch, heads, seq_len, dim).cuda().float()

# 调用 Flash Attention
out = ops.flash_attention(q, k, v)
print(f"Flash Attention output shape: {out.shape}")
```
### 2. Test other operators
```bash
python flash_attention_test.py
python sgemm_test.py
python softmax_test.py
```
## 📄 License
 - MIT License
