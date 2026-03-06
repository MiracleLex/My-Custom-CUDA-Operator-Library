#include "elementwise.h"
#include <cuda.h>
#include <cuda_runtime.h>

#define CEIL(a, b) ((a + b - 1) / (b))
#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define FLOAT4C(value) (reinterpret_cast<const float4*>(&(value))[0])

__global__ void vec_add_forward_kernel(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    int N
) {
    // 每个线程处理4个元素
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx < N) {
        // 尾部检查：只有当剩余元素>=4时才用向量化
        if (idx + 3 < N) {
            float4 tmp_a = FLOAT4C(a[idx]);
            float4 tmp_b = FLOAT4C(b[idx]);
            float4 tmp_c;
            tmp_c.x = tmp_a.x + tmp_b.x;
            tmp_c.y = tmp_a.y + tmp_b.y;
            tmp_c.z = tmp_a.z + tmp_b.z;
            tmp_c.w = tmp_a.w + tmp_b.w;
            FLOAT4(c[idx]) = tmp_c;
        } else {
            // 尾部逐元素处理
            for (int i = 0; i < 4 && idx + i < N; i++) {
                c[idx + i] = a[idx + i] + b[idx + i];
            }
        }
    }
}
// ========== CUDA封装函数 ==========
torch::Tensor vec_add_forward(torch::Tensor a, torch::Tensor b) {
    auto c = torch::empty_like(a);
    int N = a.numel();
    int threads = 256;
    int blocks = CEIL(N, threads * 4);  // 注意：每个线程处理4个
    vec_add_forward_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), c.data_ptr<float>(), N);
    return c;
}
// ========== vec_add 反向 ==========
// c = a + b
// ∂L/∂a = ∂L/∂c, ∂L/∂b = ∂L/∂c
std::tuple<torch::Tensor, torch::Tensor> vec_add_backward(torch::Tensor grad_c) {
    return {grad_c, grad_c};  // 梯度直接传递
}

__global__ void vec_relu_forward_kernel(
    const float* __restrict__ a,
    float* __restrict__ b,
    int N
) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx < N) {
        if (idx + 3 < N) {
            float4 tmp_a = FLOAT4C(a[idx]);
            float4 tmp_b;
            tmp_b.x = fmaxf(0.0f, tmp_a.x);
            tmp_b.y = fmaxf(0.0f, tmp_a.y);
            tmp_b.z = fmaxf(0.0f, tmp_a.z);
            tmp_b.w = fmaxf(0.0f, tmp_a.w);
            FLOAT4(b[idx]) = tmp_b;
        } else {
            for (int i = 0; i < 4 && idx + i < N; i++) {
                b[idx + i] = fmaxf(0.0f, a[idx + i]);
            }
        }
    }
}
__global__ void vec_relu_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,  // 前向传播的输入
    float* __restrict__ grad_x,
    int N
) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx < N) {
        if (idx + 3 < N) {
            float4 tmp_g = FLOAT4C(grad_out[idx]);
            float4 tmp_x = FLOAT4C(x[idx]);
            float4 tmp_dx;
            // ReLU导数: 1 if x > 0, else 0
            tmp_dx.x = tmp_x.x > 0.0f ? tmp_g.x : 0.0f;
            tmp_dx.y = tmp_x.y > 0.0f ? tmp_g.y : 0.0f;
            tmp_dx.z = tmp_x.z > 0.0f ? tmp_g.z : 0.0f;
            tmp_dx.w = tmp_x.w > 0.0f ? tmp_g.w : 0.0f;
            FLOAT4(grad_x[idx]) = tmp_dx;
        } else {
            for (int i = 0; i < 4 && idx + i < N; i++) {
                grad_x[idx + i] = x[idx + i] > 0.0f ? grad_out[idx + i] : 0.0f;
            }
        }
    }
}
torch::Tensor vec_relu_forward(torch::Tensor a) {
    auto b = torch::empty_like(a);
    int N = a.numel();
    int threads = 256;
    int blocks = CEIL(N, threads * 4);
    vec_relu_forward_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), N);
    return b;
}
torch::Tensor vec_relu_backward(torch::Tensor grad_out, torch::Tensor x) {
    auto grad_x = torch::empty_like(x);
    int N = x.numel();
    int threads = 256;
    int blocks = CEIL(N, threads * 4);
    vec_relu_backward_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(), x.data_ptr<float>(), grad_x.data_ptr<float>(), N);
    return grad_x;
}

__global__ void vec_sigmoid_forward_kernel(
    const float* __restrict__ a,
    float* __restrict__ b,
    int N
) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx < N) {
        if (idx + 3 < N) {
            float4 tmp_a = FLOAT4C(a[idx]);
            float4 tmp_b;
            tmp_b.x = tmp_a.x >= 0.0f ? 
                1.0f / (1.0f + expf(-tmp_a.x)) : 
                expf(tmp_a.x) / (1.0f + expf(tmp_a.x));
            tmp_b.y = tmp_a.y >= 0.0f ? 
                1.0f / (1.0f + expf(-tmp_a.y)) : 
                expf(tmp_a.y) / (1.0f + expf(tmp_a.y));
            tmp_b.z = tmp_a.z >= 0.0f ? 
                1.0f / (1.0f + expf(-tmp_a.z)) : 
                expf(tmp_a.z) / (1.0f + expf(tmp_a.z));
            tmp_b.w = tmp_a.w >= 0.0f ? 
                1.0f / (1.0f + expf(-tmp_a.w)) : 
                expf(tmp_a.w) / (1.0f + expf(tmp_a.w));
            FLOAT4(b[idx]) = tmp_b;
        } else {
            for (int i = 0; i < 4 && idx + i < N; i++) {
                float x = a[idx + i];
                b[idx + i] = x >= 0.0f ? 
                    1.0f / (1.0f + expf(-x)) : 
                    expf(x) / (1.0f + expf(x));
            }
        }
    }
}
__global__ void vec_sigmoid_backward_kernel(
    const float* __restrict__ grad_out,
    const float* __restrict__ y,  // 前向传播的输出 sigmoid(x)
    float* __restrict__ grad_x,
    int N
) {
    int idx = (blockDim.x * blockIdx.x + threadIdx.x) * 4;
    if (idx < N) {
        if (idx + 3 < N) {
            float4 tmp_g = FLOAT4C(grad_out[idx]);
            float4 tmp_y = FLOAT4C(y[idx]);
            float4 tmp_dx;
            // sigmoid导数: y * (1 - y)
            tmp_dx.x = tmp_g.x * tmp_y.x * (1.0f - tmp_y.x);
            tmp_dx.y = tmp_g.y * tmp_y.y * (1.0f - tmp_y.y);
            tmp_dx.z = tmp_g.z * tmp_y.z * (1.0f - tmp_y.z);
            tmp_dx.w = tmp_g.w * tmp_y.w * (1.0f - tmp_y.w);
            FLOAT4(grad_x[idx]) = tmp_dx;
        } else {
            for (int i = 0; i < 4 && idx + i < N; i++) {
                float yi = y[idx + i];
                grad_x[idx + i] = grad_out[idx + i] * yi * (1.0f - yi);
            }
        }
    }
}
torch::Tensor vec_sigmoid_forward(torch::Tensor a) {
    auto b = torch::empty_like(a);
    int N = a.numel();
    int threads = 256;
    int blocks = CEIL(N, threads * 4);
    vec_sigmoid_forward_kernel<<<blocks, threads>>>(
        a.data_ptr<float>(), b.data_ptr<float>(), N);
    return b;
}
torch::Tensor vec_sigmoid_backward(torch::Tensor grad_out, torch::Tensor y) {
    auto grad_x = torch::empty_like(y);
    int N = y.numel();
    int threads = 256;
    int blocks = CEIL(N, threads * 4);
    vec_sigmoid_backward_kernel<<<blocks, threads>>>(
        grad_out.data_ptr<float>(), y.data_ptr<float>(), grad_x.data_ptr<float>(), N);
    return grad_x;
}