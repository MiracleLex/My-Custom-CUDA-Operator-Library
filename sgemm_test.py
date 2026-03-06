import torch
import MyCustomCUDAOperatorLibrary._C as ops
import time

def benchmark_sgemm(M, N, K, warmup=10, repeat=100):
    # 初始化数据
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    C = torch.randn(M, N, device='cuda', dtype=torch.float32)
    
    alpha = 1.0
    beta = 0.0

    # --------------------------
    # 1. 预热
    # --------------------------
    for _ in range(warmup):
        ops.sgemm(A, B, C, alpha, beta)
    
    # 确保预热完成
    torch.cuda.synchronize()

    # --------------------------
    # 2. 使用 CUDA Events 计时
    # --------------------------
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    # 记录开始时间
    start_event.record()
    
    for _ in range(repeat):
        ops.sgemm(A, B, C, alpha, beta)
        
    # 记录结束时间
    end_event.record()
    torch.cuda.synchronize() # 等待 GPU 完成

    # 计算平均耗时 (毫秒)
    elapsed_time_ms = start_event.elapsed_time(end_event) / repeat

    # --------------------------
    # 3. 计算 GFLOPS
    # --------------------------
    # FLOPs = 2 * M * N * K (乘加各算一次)
    flops = 2.0 * M * N * K
    # GFLOPS = FLOPs / (时间(秒) * 1e9)
    gflops = (flops / (elapsed_time_ms * 1e6))

    print(f"Size: {M}x{N}x{K} | Time: {elapsed_time_ms:.4f} ms | Performance: {gflops:.2f} GFLOPS")
    return gflops

def benchmark_pytorch(M, N, K, warmup=10, repeat=100):
    # 对比 PyTorch 官方实现
    A = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B = torch.randn(K, N, device='cuda', dtype=torch.float32)
    C = torch.randn(M, N, device='cuda', dtype=torch.float32)

    # 预热
    for _ in range(warmup):
        # torch.addmm 计算 C = beta*C + alpha*A@B
        torch.addmm(C, A, B, beta=0.0, alpha=1.0)
    
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(repeat):
        torch.addmm(C, A, B, beta=0.0, alpha=1.0)
    end_event.record()
    
    torch.cuda.synchronize()
    
    elapsed_time_ms = start_event.elapsed_time(end_event) / repeat
    flops = 2.0 * M * N * K
    gflops = (flops / (elapsed_time_ms * 1e6))
    
    print(f"[PyTorch] Size: {M}x{N}x{K} | Time: {elapsed_time_ms:.4f} ms | Performance: {gflops:.2f} GFLOPS")

if __name__ == "__main__":
    # 测试不同尺寸
    sizes = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]
    
    print("------ My CUDA Op ------")
    for m, n, k in sizes:
        benchmark_sgemm(m, n, k)
        
    print("\n------ PyTorch (cuBLAS) ------")
    for m, n, k in sizes:
        benchmark_pytorch(m, n, k)