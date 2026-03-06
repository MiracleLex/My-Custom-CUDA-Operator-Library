#include "flash_attention.h"
#include <cuda_runtime.h>
#include <c10/cuda/CUDAException.h>

#define OFFSET(row, col, ld) ((row) * (ld) + (col))
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])

// 定义分块大小
// Br: 必须小于等于 BlockSize (128)
// Bc: K/V 循环块大小
template<const int Br, const int Bc, const int d_model>
__global__ void flash_attention_kernel(float* Q, float* K, float* V, float* O, 
                                       int B, int H, int N, int d) {
    // 1. 索引计算
    int bx = blockIdx.x; // Batch
    int by = blockIdx.y; // Head
    int bz = blockIdx.z; // N 维度 Block 索引
    
    int tx = threadIdx.x; // 线程 ID (0 to 127)
    
    // 当前 Block 负责的 Query 起始行
    int q_start = bz * Br;
    
    // 当前线程负责计算的 Query 行索引 (相对于 Block)
    // 我们让线程 0 处理第 0 行，线程 1 处理第 1 行...
    // 如果线程数 > Br，多余的线程只是辅助搬运数据，不参与计算
    int q_row_idx = tx;

    // 判断当前线程是否负责计算
    // 只有 tx < Br 的线程才拥有计算状态，其他线程只帮忙搬运数据
    bool is_active_thread = (q_row_idx < Br) && (q_start + q_row_idx < N); // (q_start + q_row_idx < N)尾部防溢出

    // 全局偏移量
    int qkv_offset = (bx * H + by) * N * d;
    
    // 2. 共享内存分配
    __shared__ float Q_s[Br * d_model];
    __shared__ float K_s[Bc * d_model];
    __shared__ float V_s[Bc * d_model];

    // 3. 寄存器分配 (只有活跃线程需要初始化这些)
    float row_m_prev = -INFINITY;
    float row_l_prev = 0.0f;
    float row_o[d_model] = {0.0f}; 

    // ---------------------------------------------------------
    // 阶段 1: 加载 Q 到 Shared Memory
    // 所有线程协作加载 (Cooperative Load)
    // ---------------------------------------------------------
    // 总数据量: Br * d
    // 每个线程搬运 (Br * d) / blockDim.x 个元素
    // 假设 blockDim.x = 128, Br=16, d=64 -> 1024 floats. 每个线程搬运 8 个 float
    for (int i = tx; i < Br * d; i += blockDim.x) {
        int r = i / d;
        int c = i % d;
        int global_idx = qkv_offset + (q_start + r) * d + c;
        // 只有在有效范围内才加载，防止越界
        if (q_start + r < N) {
            Q_s[i] = Q[global_idx];
        }
    }
    __syncthreads(); // 确保加载完成

    // ---------------------------------------------------------
    // 阶段 2: 主循环 (遍历 K/V)
    // ---------------------------------------------------------
    int Tc = (N + Bc - 1) / Bc;

    for (int j = 0; j < Tc; j++) {
        int k_start = j * Bc;

        // 2.1 协作加载 K, V 块到 Shared Memory
        for (int i = tx; i < Bc * d; i += blockDim.x) {
            int r = i / d;
            int c = i % d;
            int global_idx_k = qkv_offset + (k_start + r) * d + c;
            int global_idx_v = qkv_offset + (k_start + r) * d + c;
            
            if (k_start + r < N) {
                K_s[i] = K[global_idx_k];
                V_s[i] = V[global_idx_v];
            }
        }
        __syncthreads(); // 等待 K, V 加载完成

        // 2.2 计算 Softmax 和 Attention
        // 只有负责计算 Q 行的线程才执行这段逻辑
        if (is_active_thread) {
            float S_block[Bc];  // 存储“1行Q × 1个K块”的结果（1×Bc）
            // 这段代码的执行者是单个线程，对应 Q 块中「某一行」（q_row_idx），而不是整个 Q 块（Br 行）
            
            // Step A: Q * K^T 单个 Q 行 × 整个小 K 块
            for (int k_idx = 0; k_idx < Bc; k_idx++) {
                if (k_start + k_idx >= N) {
                    S_block[k_idx] = -INFINITY; // Padding
                    continue;
                }
                
                float sum = 0.0f;
                // 计算 Q_s[q_row_idx] 与 K_s[k_idx] 的点积
                for (int dim = 0; dim < d; dim++) {
                    sum += Q_s[q_row_idx * d + dim] * K_s[k_idx * d + dim];
                }
                S_block[k_idx] = sum / sqrtf((float)d);
            }

            // Step B: Online Softmax Update
            // 1. 找当前块最大值
            float row_m_block = -INFINITY;
            for (int k_idx = 0; k_idx < Bc; k_idx++) {
                row_m_block = fmaxf(row_m_block, S_block[k_idx]);
            }

            // 2. 更新全局最大值
            float m_new = fmaxf(row_m_prev, row_m_block);

            // 3. 计算当前块的指数和 l_block
            // 同时修正 P 值用于后续计算
            float l_block = 0.0f;
            for (int k_idx = 0; k_idx < Bc; k_idx++) {
                if (k_start + k_idx >= N) continue;
                
                // 注意：P 的计算必须基于 m_new
                float p = expf(S_block[k_idx] - m_new);
                S_block[k_idx] = p; // 缓存 P 值，分子向量
                l_block += p; // 分母sum
            }

            // 4. 更新全局 l
            // l_new = l_old * exp(m_old - m_new) + l_block
            float l_new = expf(row_m_prev - m_new) * row_l_prev + l_block;

            // Step C: 更新输出 O
            // 修正因子：之前累加的结果需要乘以 exp(m_prev - m_new) * (l_prev / l_new)
            // 之前的 O 实际上存储的是： sum(P_old * V) / l_old
            // 我们需要的更新公式： O_new = (Correction * O_old + P_new * V) / l_new
            
            // Correction = (l_prev / l_new) * exp(m_prev - m_new)
            float correction = (row_l_prev / l_new) * expf(row_m_prev - m_new);

            for (int dim = 0; dim < d; dim++) {
                row_o[dim] *= correction; // 修正之前的累加值
            } // row_o 里存的是softmax后得到的向量（1*Bc）与 V 块（Bc*d）的点积结果，维度是 1*d

            // 累加当前块的 P * V
            for (int k_idx = 0; k_idx < Bc; k_idx++) {
                if (k_start + k_idx >= N) continue;
                
                // P 值已经算过了，存在 S_block[k_idx]
                float p_val = S_block[k_idx]; 
                
                // 我们需要除以 l_new 来归一化
                p_val /= l_new;
                
                for (int dim = 0; dim < d; dim++) {
                    row_o[dim] += p_val * V_s[k_idx * d + dim];
                }
            }

            // 更新状态
            row_m_prev = m_new;
            row_l_prev = l_new;
        }
        
        __syncthreads(); // 确保计算完成，开始下一轮加载
    }

    // ---------------------------------------------------------
    // 阶段 3: 写回结果
    // ---------------------------------------------------------
    if (is_active_thread) {
        int q_abs = q_start + q_row_idx;
        for (int dim = 0; dim < d; dim++) {
            O[qkv_offset + q_abs * d + dim] = row_o[dim];
        }
    }
}

// 包装函数
torch::Tensor flash_attention_forward(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    TORCH_CHECK(Q.is_cuda() && K.is_cuda() && V.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(Q.dim() == 4, "Input must be 4D (B, H, N, d)");

    int B = Q.size(0);
    int H = Q.size(1);
    int N = Q.size(2);
    int d = Q.size(3);

    auto O = torch::zeros_like(Q);

    // 配置参数
    // Block 线程数设为 128，可以更好地掩盖访存延迟
    // Br=16, Bc=16
    const int Br = 16; 
    const int Bc = 16; 
    
    dim3 grid(B, H, (N + Br - 1) / Br);
    dim3 block(128); // 修改为 128 以支持更高效的数据搬运

    if (d == 64) {
        flash_attention_kernel<16, 16, 64><<<grid, block>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            O.data_ptr<float>(), B, H, N, d
        );
    } else if (d == 32) {
         flash_attention_kernel<16, 16, 32><<<grid, block>>>(
            Q.data_ptr<float>(), K.data_ptr<float>(), V.data_ptr<float>(),
            O.data_ptr<float>(), B, H, N, d
        );
    } else {
        TORCH_CHECK(false, "Currently only d=64 or d=32 is implemented for this demo");
    }

    C10_CUDA_CHECK(cudaGetLastError());
    return O;
}