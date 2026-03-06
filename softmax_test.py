import torch
import time
import MyCustomCUDAOperatorLibrary._C as ops
import torch.nn.functional as F

def benchmark_softmax(N, C, iterations=100, warmup=10, dtype=torch.float32):
    """
    对比自定义 Softmax 与 PyTorch 官方实现的性能
    
    参数:
        N: Batch size (行数)
        C: Feature size (列数/Softmax 计算维度)
        iterations: 测试迭代次数
        warmup: 预热次数
        dtype: 数据类型 (torch.float32 或 torch.float16)
    """
    device = torch.device('cuda')
    
    # 1. 准备数据
    # 生成随机输入，模拟 logits (通常范围较大)
    x = torch.randn(N, C, device=device, dtype=dtype)
    grad_out = torch.ones_like(x) # 反向传播的梯度输入
    
    # -----------------------------------------------------------
    # 2. 正确性验证
    # -----------------------------------------------------------
    x_cuda = x.clone().requires_grad_(True)
    x_torch = x.clone().requires_grad_(True)
    
    # Forward
    y_cuda = ops.vec_softmax(x_cuda)
    y_torch = F.softmax(x_torch, dim=-1)
    
    # 检查前向结果误差
    max_fwd_error = (y_cuda - y_torch).abs().max().item()
    
    # Backward
    y_cuda.backward(grad_out, retain_graph=True)
    y_torch.backward(grad_out, retain_graph=True)
    
    # 检查反向梯度误差
    max_bwd_error = (x_cuda.grad - x_torch.grad).abs().max().item()
    
    print(f"\n{'='*60}")
    print(f"Config: N={N}, C={C}, Dtype={dtype}")
    print(f"Max Forward Error:  {max_fwd_error:.2e}")
    print(f"Max Backward Error: {max_bwd_error:.2e}")
    
    if max_fwd_error > 1e-3 or max_bwd_error > 1e-3:
        print("⚠️  WARNING: Large error detected! Results may be incorrect.")
    
    # -----------------------------------------------------------
    # 3. 性能测试
    # -----------------------------------------------------------
    
    # 辅助函数：计时
    def timed_execution(func, iters):
        # 预热
        for _ in range(warmup):
            func()
        torch.cuda.synchronize()
        
        # 开始计时
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()
        for _ in range(iters):
            func()
        end.record()
        torch.cuda.synchronize()
        
        return start.elapsed_time(end) / iters  # 返回平均耗时

    # --- 测试自定义 CUDA 实现 ---
    def run_custom():
        x_tmp = x.clone().requires_grad_(True)
        y = ops.vec_softmax(x_tmp)
        y.backward(grad_out, retain_graph=True)
        
    time_custom = timed_execution(run_custom, iterations)
    
    # --- 测试 PyTorch 官方实现 ---
    def run_torch():
        x_tmp = x.clone().requires_grad_(True)
        y = F.softmax(x_tmp, dim=-1)
        y.backward(grad_out, retain_graph=True)
        
    time_torch = timed_execution(run_torch, iterations)

    # -----------------------------------------------------------
    # 4. 计算带宽
    # -----------------------------------------------------------
    # Softmax 内存访问量估算 (近似)：
    # Forward: Read N*C, Write N*C
    # Backward: Read Y (N*C), Read GradOut (N*C), Write GradIn (N*C)
    # 总读写量 ≈ 5 * N * C * sizeof(dtype)
    element_size = 4 if dtype == torch.float32 else 2
    data_size_bytes = 5 * N * C * element_size
    
    bandwidth_custom = (data_size_bytes / (time_custom * 1e-3)) / 1e9  # GB/s
    bandwidth_torch = (data_size_bytes / (time_torch * 1e-3)) / 1e9   # GB/s
    
    speedup = time_torch / time_custom

    # 打印结果
    print(f"{'-'*60}")
    print(f"{'Implementation':<15} {'Time (ms)':<12} {'Bandwidth (GB/s)':<18} {'Status'}")
    print(f"{'Custom CUDA':<15} {time_custom:<12.4f} {bandwidth_custom:<18.2f} {'✅' if speedup >= 1.0 else '🚀'}")
    print(f"{'PyTorch Native':<15} {time_torch:<12.4f} {bandwidth_torch:<18.2f} {'Reference'}")
    print(f"{'Speedup':<15} {speedup:.2f}x")
    print(f"{'='*60}")

if __name__ == "__main__":
    # 确保 GPU 可用
    if not torch.cuda.is_available():
        print("CUDA is not available. Exiting.")
        exit()
    
    print("Starting Softmax Benchmark...")
    
    # 测试场景 1: 典型 Transformer 维度 (Batch size 较小, Hidden dim 较大)
    # 类似于 Attention 中的 softmax
    benchmark_softmax(N=4096, C=4096)      # Large
    
    # 测试场景 2: Batch 较大，维度较小
    benchmark_softmax(N=10000, C=1000)
    
    # 测试场景 3: 非常大的维度 (测试归约性能)
    benchmark_softmax(N=128, C=50000)

    # (可选) 测试 FP16 性能
    # 如果你的内核支持 FP16，可以取消下面这行的注释
    # benchmark_softmax(N=4096, C=4096, dtype=torch.float16)