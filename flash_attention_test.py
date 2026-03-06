import torch
import MyCustomCUDAOperatorLibrary._C as ops
import time

def test_flash_attention():
    # 参数设置
    B, H, N, d = 1, 1, 128, 64  # 注意：当前 Demo Kernel 仅支持 d=64
    
    # 初始化数据
    Q = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    K = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)
    V = torch.randn(B, H, N, d, device='cuda', dtype=torch.float32)

    # 1. PyTorch 标准实现 用于验证正确性
    # S = QK^T / sqrt(d)
    # A = softmax(S)
    # O = AV
    att = torch.matmul(Q, K.transpose(-2, -1)) / (d ** 0.5)
    att = torch.softmax(att, dim=-1)
    O_ref = torch.matmul(att, V)

    # 2. 你的 FlashAttention 实现
    O_my = ops.flash_attention(Q, K, V)

    # 3. 验证
    diff = (O_ref - O_my).abs().max()
    print(f"Max difference: {diff.item()}")
    
    if diff < 1e-4:
        print("Test Passed!")
    else:
        print("Test Failed!")

if __name__ == "__main__":
    test_flash_attention()