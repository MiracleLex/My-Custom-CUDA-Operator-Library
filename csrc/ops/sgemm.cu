#include "sgemm.h"
#include <cuda_runtime.h>

// 定义缺失的宏
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4*>(&(pointer))[0])
#define OFFSET(row, col, ld) ((row) * (ld) + (col))

// ---------------- 你的核函数代码开始 ----------------
template<const int BM,
         const int BN,
         const int BK,
         const int TM,
         const int TN>
__global__ void sgemm_kernel(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    const int thread_num = block_row_thread * block_col_thread; 

    // 当前线程对应thread tile的左上角元素在block中的位置
    int tx = (threadIdx.x % block_row_thread) * TN;
    int ty = (threadIdx.x / block_row_thread) * TM;

    __shared__ float As[2][BK * BM]; 
    __shared__ float Bs[2][BK * BN];

    const int ldg_a_num = BK * BM / thread_num / 4; 
    const int ldg_b_num = BK * BN / thread_num / 4; 

    int a_tile_row = threadIdx.x / (BK / 4); 
    int a_tile_col = threadIdx.x % (BK / 4) * 4;
    int a_tile_stride = BM / ldg_a_num; 

    int b_tile_row = threadIdx.x / (BN / 4); 
    int b_tile_col = threadIdx.x % (BN / 4) * 4;
    int b_tile_stride = BK / ldg_b_num; 

    float accum[TM][TN] = {0.}; 

    float ldg_a_reg[4 * ldg_a_num] = {0.}; 
    float ldg_b_reg[4 * ldg_b_num] = {0.}; 

    float a_frag[2][TM];  
    float b_frag[2][TN];  

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    // first global to shared
    #pragma unroll
    for (int i = 0; i < BM; i += a_tile_stride) {
        int ldg_index = i / a_tile_stride * 4;  
        FETCH_FLOAT4(ldg_a_reg[ldg_index]) = 
                FETCH_FLOAT4(A[OFFSET(a_tile_row + i, a_tile_col, K)]);
        As[0][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index]; // 竖向装载
        As[0][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
        As[0][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
        As[0][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
    }
    #pragma unroll
    for (int i = 0; i < BK; i += b_tile_stride) {
        FETCH_FLOAT4(Bs[0][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                FETCH_FLOAT4(B[OFFSET(b_tile_row + i, b_tile_col, N)]); 
    }
    
    int write_index = 1; // 是准备从reg写入【下一轮】要用的数据的
    int load_index; // 是读出到frag准备计算的
    int k = 0;
    do {  
        __syncthreads();  
        k += BK;
        if (k < K) {
            #pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) { // 这里做【下一BK】的从全局到寄存器reg，就是上面的第一步，没忙着做第二步从reg到shared memory
                int ldg_index = i / a_tile_stride * 4;  
                FETCH_FLOAT4(ldg_a_reg[ldg_index]) =
                        FETCH_FLOAT4(A[OFFSET(a_tile_row + i, k + a_tile_col, K)]);
            }
            #pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                int ldg_index = i / b_tile_stride * 4;  
                FETCH_FLOAT4(ldg_b_reg[ldg_index]) =
                        FETCH_FLOAT4(B[OFFSET(k + b_tile_row + i, b_tile_col, N)]);
            }
        }
        load_index = write_index ^ 1; 
        #pragma unroll
        for (int m = 0; m < TM; m += 4) { // shared memory到reg，后续计算在reg中进行
            FETCH_FLOAT4(a_frag[0][m]) = FETCH_FLOAT4(
                   As[load_index][OFFSET(0, ty + m, BM)]); 
        }
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            FETCH_FLOAT4(b_frag[0][n]) = FETCH_FLOAT4(
                    Bs[load_index][OFFSET(0, tx + n, BN)]); 
        }
        #pragma unroll
        for (int bk = 0; bk < BK - 1; bk++) {  
            for (int m = 0; m < TM; m += 4) { // 双缓冲，计算的同时加载下一步的shared memory到reg
                FETCH_FLOAT4(a_frag[(bk + 1) % 2][m]) = FETCH_FLOAT4(
                        As[load_index][OFFSET(bk + 1, ty + m, BM)]); 
            }
            #pragma unroll
            for (int n = 0; n < TN; n += 4) {
                FETCH_FLOAT4(b_frag[(bk + 1) % 2][n]) = FETCH_FLOAT4(
                        Bs[load_index][OFFSET(bk + 1, tx + n, BN)]); 
            }
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                for (int n = 0; n < TN; n++) {
                    accum[m][n] += a_frag[bk % 2][m] * b_frag[bk % 2][n];
                }
            }
        }
        #pragma unroll
        for (int m = 0; m < TM; m++) { // 扫尾，计算最后一个BK的结果
            #pragma unroll
            for (int n = 0; n < TN; n++) {
                accum[m][n] += a_frag[(BK - 1) % 2][m] * b_frag[(BK - 1) % 2][n];
            }
        }
        if (k < K) {
            #pragma unroll
            for (int i = 0; i < BM; i += a_tile_stride) { // 这里做【下一BK】数据准备的第二步，从reg到shared memory
                int ldg_index = i / a_tile_stride * 4;
                As[write_index][OFFSET(a_tile_col, i + a_tile_row, BM)] = ldg_a_reg[ldg_index];
                As[write_index][OFFSET(a_tile_col + 1, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 1];
                As[write_index][OFFSET(a_tile_col + 2, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 2];
                As[write_index][OFFSET(a_tile_col + 3, i + a_tile_row, BM)] = ldg_a_reg[ldg_index + 3];
            }
            #pragma unroll
            for (int i = 0; i < BK; i += b_tile_stride) {
                int ldg_index = i / b_tile_stride * 4;
                FETCH_FLOAT4(Bs[write_index][OFFSET(b_tile_row + i, b_tile_col, BN)]) =
                        FETCH_FLOAT4(ldg_b_reg[ldg_index]);
            }
            write_index ^= 1;
        }
    } while (k < K);
    
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n += 4) {
            float4 ctmp = FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]);
            ctmp.x = alpha * accum[m][n] + beta * ctmp.x;
            ctmp.y = alpha * accum[m][n + 1] + beta * ctmp.y;
            ctmp.z = alpha * accum[m][n + 2] + beta * ctmp.z;
            ctmp.w = alpha * accum[m][n + 3] + beta * ctmp.w;
            FETCH_FLOAT4(C[OFFSET(ty + m, tx + n, N)]) = ctmp;
        }
    }
}
// ---------------- 你的核函数代码结束 ----------------

// 包装函数，处理 Grid/Block 配置
void sgemm(torch::Tensor A, torch::Tensor B, torch::Tensor C, float alpha, float beta) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(1); // B: K x N
    
    // 选择一组常用的优化参数配置
    // BM=128, BN=128, BK=8, TM=8, TN=8 是比较经典的配置
    // 如果需要极致性能，可以根据 M, N, K 动态选择不同配置
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    // 计算线程块数量
    dim3 gridDim((N + BN - 1) / BN, (M + BM - 1) / BM);
    
    // 计算每个 Block 的线程数
    // 由核函数内部逻辑可知: thread_num = (BM/TM) * (BN/TN)
    const int block_row_thread = BN / TN;
    const int block_col_thread = BM / TM;
    dim3 blockDim(block_row_thread * block_col_thread);

    // 获取数据指针
    float* d_A = A.data_ptr<float>();
    float* d_B = B.data_ptr<float>();
    float* d_C = C.data_ptr<float>();

    // 启动核函数
    sgemm_kernel<BM, BN, BK, TM, TN><<<gridDim, blockDim>>>(
        M, N, K, alpha, d_A, d_B, beta, d_C
    );
}