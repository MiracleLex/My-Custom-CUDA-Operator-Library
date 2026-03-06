#include "reduce.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define CEIL(a, b) ((a + b - 1) / (b))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define FLOAT4C(value) (reinterpret_cast<const float4*>(&(value))[0])

// ========== vec_sum ==========

__global__ void vec_sum_forward_kernel(
    const float* __restrict__ d_x,
    float* __restrict__ d_y,
    const int N
) {
    __shared__ float s_y[32];  // 一个block最多32个warp
    
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    int warpId = threadIdx.x / warpSize;
    int laneId = threadIdx.x % warpSize;
    
    // 每个线程加载并累加4个元素
    float val = 0.0f;
    if (idx < N) {
        if (idx + 3 < N) {
            float4 tmp_x = FLOAT4C(d_x[idx]);
            val = tmp_x.x + tmp_x.y + tmp_x.z + tmp_x.w;
        } else {
            // 尾部处理
            for (int i = 0; i < 4 && idx + i < N; i++) {
                val += d_x[idx + i];
            }
        }
    }
    
    // Warp 内归约（使用 shuffle 指令）
    #pragma unroll
    for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    
    // 每个 warp 的第一个线程写入共享内存
    if (laneId == 0) {
        s_y[warpId] = val;
    }
    __syncthreads();
    
    // 第一个 warp 对共享内存中的结果进行最终归约
    if (warpId == 0) {
        int warpNum = (blockDim.x + warpSize - 1) / warpSize;
        val = (laneId < warpNum) ? s_y[laneId] : 0.0f;
        
        #pragma unroll
        for (int offset = warpSize >> 1; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        }
        
        // 只有 lane 0 写入全局结果（原子加）
        if (laneId == 0) {
            atomicAdd(d_y, val);
        }
    }
}

torch::Tensor vec_sum_forward(torch::Tensor x) {
    // 输出是单个标量
    auto y = torch::zeros({1}, x.options());
    
    int N = x.numel();
    if (N == 0) return y;
    
    int threads = 256;  // 必须是32的倍数
    int blocks = CEIL(N, threads * 4);
    
    // 启动核函数
    vec_sum_forward_kernel<<<blocks, threads>>>(
        x.data_ptr<float>(),
        y.data_ptr<float>(),
        N
    );
    
    return y;
}

// ========== vec_sum 反向 ==========
// y = sum(x)
// ∂y/∂x_i = 1
// grad_x_i = grad_y × 1 = grad_y
// 每个输入元素的梯度都等于上游梯度（广播）

__global__ void vec_sum_backward_kernel(
    const float grad_y,
    float* __restrict__ grad_x,
    const int N
) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    
    if (idx < N) {
        if (idx + 3 < N) {
            float4 tmp;
            tmp.x = grad_y;
            tmp.y = grad_y;
            tmp.z = grad_y;
            tmp.w = grad_y;
            FLOAT4(grad_x[idx]) = tmp;
        } else {
            for (int i = 0; i < 4 && idx + i < N; i++) {
                grad_x[idx + i] = grad_y;
            }
        }
    }
}

torch::Tensor vec_sum_backward_cuda(torch::Tensor grad_y, torch::Tensor x) {
    auto grad_x = torch::empty_like(x);  // 直接用 x 的形状
    int N = x.numel();                   // 从 x 获取长度
    
    int threads = 256;
    int blocks = CEIL(N, threads * 4);
    
    float grad_val = grad_y.item<float>();
    
    vec_sum_backward_kernel<<<blocks, threads>>>(
        grad_val,
        grad_x.data_ptr<float>(),
        N
    );
    
    return grad_x;
}

// ========== vec_softmax ==========

// Warp 级归约：求最大值
__device__ float warpReduceMax(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Warp 级归约：求和
__device__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void softmax_forward_kernel(
    float* __restrict__ out,
    const float* __restrict__ inp,
    int N,
    int C
) {
    extern __shared__ float shared[];

    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpsPerBlock = blockDim.x / 32;

    float* maxvals = shared;
    float* sumvals = &shared[warpsPerBlock];

    const float* x = inp + idx * C;
    float* y = out + idx * C;

    // ========== 第一步：求最大值（向量化） ==========
    float maxval = -INFINITY;
    
    // 向量化处理：每次处理 4 个元素
    int vec_C = C / 4;
    for (int i = tid; i < vec_C; i += blockDim.x) {
        float4 val = FLOAT4C(x[i * 4]);
        maxval = fmaxf(maxval, val.x);
        maxval = fmaxf(maxval, val.y);
        maxval = fmaxf(maxval, val.z);
        maxval = fmaxf(maxval, val.w);
    }
    
    // 处理尾部
    int tail_start = vec_C * 4;
    for (int i = tail_start + tid; i < C; i += blockDim.x) {
        maxval = fmaxf(maxval, x[i]);
    }
    
    // Warp 归约
    maxval = warpReduceMax(maxval);
    if (laneId == 0) maxvals[warpId] = maxval;
    __syncthreads();

    // Block 归约
    if (warpId == 0) {
        maxval = (laneId < warpsPerBlock) ? maxvals[laneId] : -INFINITY;
        maxval = warpReduceMax(maxval);
        if (laneId == 0) maxvals[0] = maxval;
    }
    __syncthreads();

    float offset = maxvals[0];

    // ========== 第二步 + 第三步融合：计算 exp 和 sum ==========
    float sumval = 0.0f;
    
    for (int i = tid; i < vec_C; i += blockDim.x) {
        float4 val = FLOAT4C(x[i * 4]);
        val.x = expf(val.x - offset);
        val.y = expf(val.y - offset);
        val.z = expf(val.z - offset);
        val.w = expf(val.w - offset);
        
        // 写入 exp 结果
        FLOAT4(y[i * 4]) = val;
        
        // 累加
        sumval += val.x + val.y + val.z + val.w;
    }
    
    // 处理尾部
    for (int i = tail_start + tid; i < C; i += blockDim.x) {
        float val = expf(x[i] - offset);
        y[i] = val;
        sumval += val;
    }

    // Warp 归约 sum
    sumval = warpReduceSum(sumval);
    if (laneId == 0) sumvals[warpId] = sumval;
    __syncthreads();

    // Block 归约
    if (warpId == 0) {
        sumval = (laneId < warpsPerBlock) ? sumvals[laneId] : 0.0f;
        sumval = warpReduceSum(sumval);
        if (laneId == 0) sumvals[0] = sumval;
    }
    __syncthreads();

    float sum = sumvals[0];
    float inv_sum = 1.0f / sum;

    // ========== 第四步：归一化 ==========
    for (int i = tid; i < vec_C; i += blockDim.x) {
        float4 val = FLOAT4(y[i * 4]);
        val.x *= inv_sum;
        val.y *= inv_sum;
        val.z *= inv_sum;
        val.w *= inv_sum;
        FLOAT4(y[i * 4]) = val;
    }
    
    for (int i = tail_start + tid; i < C; i += blockDim.x) {
        y[i] *= inv_sum;
    }
}

// ========== softmax 反向传播（向量化版） ==========

__global__ void softmax_backward_kernel(
    const float* __restrict__ grad_y,   // 上游梯度 [N, C]
    const float* __restrict__ y,        // 前向输出（softmax结果）[N, C]
    float* __restrict__ grad_x,         // 输出梯度 [N, C]
    int N,
    int C
) {
    extern __shared__ float shared[];

    int idx = blockIdx.x;
    int tid = threadIdx.x;
    int warpId = threadIdx.x / 32;
    int laneId = threadIdx.x % 32;
    int warpsPerBlock = blockDim.x / 32;

    float* sumvals = shared;

    const float* grad_y_row = grad_y + idx * C;
    const float* y_row = y + idx * C;
    float* grad_x_row = grad_x + idx * C;

    // ========== 第一步：计算 dot(grad_y, y) = Σ(grad_y_i × y_i) ==========
    float dotval = 0.0f;
    
    int vec_C = C / 4;
    int tail_start = vec_C * 4;
    
    // 向量化计算点积
    for (int i = tid; i < vec_C; i += blockDim.x) {
        float4 gy = FLOAT4C(grad_y_row[i * 4]);
        float4 ys = FLOAT4C(y_row[i * 4]);
        dotval += gy.x * ys.x;
        dotval += gy.y * ys.y;
        dotval += gy.z * ys.z;
        dotval += gy.w * ys.w;
    }
    
    // 处理尾部
    for (int i = tail_start + tid; i < C; i += blockDim.x) {
        dotval += grad_y_row[i] * y_row[i];
    }

    // Warp 归约
    dotval = warpReduceSum(dotval);
    if (laneId == 0) sumvals[warpId] = dotval;
    __syncthreads();

    // Block 归约
    if (warpId == 0) {
        dotval = (laneId < warpsPerBlock) ? sumvals[laneId] : 0.0f;
        dotval = warpReduceSum(dotval);
        if (laneId == 0) sumvals[0] = dotval;
    }
    __syncthreads();

    float dot = sumvals[0];

    // ========== 第二步：计算 grad_x = y × (grad_y - dot) ==========
    
    // 向量化计算
    for (int i = tid; i < vec_C; i += blockDim.x) {
        float4 gy = FLOAT4C(grad_y_row[i * 4]);
        float4 ys = FLOAT4C(y_row[i * 4]);
        float4 gx;
        
        gx.x = ys.x * (gy.x - dot);
        gx.y = ys.y * (gy.y - dot);
        gx.z = ys.z * (gy.z - dot);
        gx.w = ys.w * (gy.w - dot);
        
        FLOAT4(grad_x_row[i * 4]) = gx;
    }
    
    // 处理尾部
    for (int i = tail_start + tid; i < C; i += blockDim.x) {
        grad_x_row[i] = y_row[i] * (grad_y_row[i] - dot);
    }
}

torch::Tensor vec_softmax_forward(torch::Tensor x) {
    // x 形状: [N, C]
    int N = x.size(0);
    int C = x.size(1);
    auto out = torch::empty_like(x);

    int threads = 256;  // 必须是 32 的倍数
    int blocks = N;
    int warpsPerBlock = threads / 32;
    size_t shared_mem = warpsPerBlock * 2 * sizeof(float);  // maxvals + sumvals

    softmax_forward_kernel<<<blocks, threads, shared_mem>>>(
        out.data_ptr<float>(),
        x.data_ptr<float>(),
        N,
        C
    );

    return out;
}

torch::Tensor vec_softmax_backward(torch::Tensor grad_y, torch::Tensor y) {
    int N = y.size(0);
    int C = y.size(1);
    
    auto grad_x = torch::empty_like(y);

    int threads = 256;  // 必须是 32 的倍数
    int blocks = N;
    int warpsPerBlock = threads / 32;
    size_t shared_mem = warpsPerBlock * sizeof(float);  // 只需要 sumvals

    softmax_backward_kernel<<<blocks, threads, shared_mem>>>(
        grad_y.data_ptr<float>(),
        y.data_ptr<float>(),
        grad_x.data_ptr<float>(),
        N,
        C
    );

    return grad_x;
}

