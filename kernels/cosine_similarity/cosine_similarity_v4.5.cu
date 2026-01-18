#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <utils.h>
using namespace nvcuda;

#define WARP_SIZE 32
#define DEVICE_INLINE __device__ inline
#define HOST_DEVICE_INLINE __device__ __host__ inline
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST64BITS(value) (reinterpret_cast<float2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// gmem -> smem
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) \
    asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

// ca(cache all, L1 + L2): support 4, 8, 16 bytes
#define CP_ASYNC_CA(dst, src, bytes)                                       \
    asm volatile(                                                          \
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(bytes))

// cg(cache global, L2): only support 16 bytes
#define CP_ASYNC_CG(dst, src, bytes)                                       \
    asm volatile(                                                          \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(bytes))

// ldmatrix
#define LDMATRIX_X2(R0, R1, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                 : "=r"(R0), "=r"(R1)                                         \
                 : "r"(addr))

#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                    \
    asm volatile(                                                            \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
        : "r"(addr))

// mma m16n8k16
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)             \
    asm volatile(                                                               \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, " \
        "%4, %5}, {%6, %7}, {%8, %9};\n"                                        \
        : "=r"(RD0), "=r"(RD1)                                                  \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), \
          "r"(RC1))

HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// -----------------------------------------------------------------------------
// Kernel: 16x128 Tile (Block Size = 32)
// MMA_TILE_M = 1 (16 rows)
// -----------------------------------------------------------------------------
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
            const int INNER_DIM = 128, const int MMA_TILE_M = 1,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 3>
__global__ void __launch_bounds__(32)
    cosine_kernel_tn_16x128(half *A, half *B,
                  half *C, int M,
                  int N, int K)
{
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);
    extern __shared__ half smem[];
    half *s_b = smem;
    // A 矩阵位于 B 矩阵之后
    half *s_a = s_b + MMA_N * INNER_DIM; 
    
    // 计算 Stage Offset
    const int s_a_stage_offset = MMA_M  * MMA_TILE_M * (INNER_DIM + A_PAD);
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = 0; // 只有一个 Warp
    const int lane_id = tid % WARP_SIZE;

    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

    // ========================================================================
    // Prologue: 加载 B 矩阵 (Gmem -> Smem)
    // ========================================================================
    // B 矩阵大小: 8行 * 128列 * 2字节 = 2048 字节
    // 线程数: 32
    // 每个线程需要搬运: 2048 / 32 = 64 字节
    // cp.async.cg 限制: 每次最大 16 字节
    // 动作: 每个线程执行 4 次加载
    // 映射: 简单的线性映射，因为 B 矩阵常驻且连续
    
#pragma unroll
    for(int i = 0; i < 4; i++) {
        // stride = 32线程 * 16字节 = 512字节
        int offset_bytes = i * 512 + tid * 16;
        int offset_halfs = offset_bytes / sizeof(half);
        
        uint32_t b_smem_addr = smem_b_base_ptr + offset_bytes;
        // 注意: 这里假设 N*K 小于 B 的总大小，或者边界安全。
        // 在 N=8, K=128 的固定场景下是安全的。
        CP_ASYNC_CG(b_smem_addr, &B[offset_halfs], 16);
    }

    // ========================================================================
    // Prologue: 加载 A 矩阵 (Gmem -> Smem)
    // ========================================================================
    // A Tile (1 Stage): 16行 * 16列 * 2字节 = 512 字节
    // 线程数: 32
    // 每个线程搬运: 16 字节 (正好 1 次 cp.async)
    // 映射: tid / 2 (行: 0-15), tid % 2 * 8 (列: 0或8)
    int load_smem_a_m = tid / 2; 
    int load_smem_a_k = (tid % 2) * 8;
    int load_gmem_a_m = by * MMA_M * MMA_TILE_M + load_smem_a_m; // MMA_TILE_M=1

    if (load_gmem_a_m >= M) return;

#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); ++k) {     
        int load_gmem_a_k = k * MMA_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * INNER_DIM + load_gmem_a_k;
        uint32_t load_smem_a_addr = (smem_a_base_ptr + 
            (k * s_a_stage_offset + load_smem_a_m * MMA_K + load_smem_a_k) * sizeof(half));
        
        CP_ASYNC_CG(load_smem_a_addr, &A[load_gmem_a_addr], 16);
        CP_ASYNC_COMMIT_GROUP(); 
    }
    
    // 等待 Prologue 加载完成 (预留 2 个 Stage 正在飞)
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    // ========================================================================
    // Main Loop
    // ========================================================================
    uint32_t RA[4];
    uint32_t RB[2];
    uint32_t RC[2] = {0, 0};
    
    half2 RA_NORM_H2[2] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f)};
    half2 RB_NORM_H2 = __float2half2_rn(0.0f);

    int b_stage = 0;
#pragma unroll
    for(int k = (K_STAGE - 1); k < NUM_K_TILES; k++){
        int compute_stage = (k + 1) % K_STAGE;
        int gmem_load_stage = k % K_STAGE;
        
        // Pipeline: 加载下一个 Stage 的 A
        int load_a_gmem_k = k * MMA_K + load_smem_a_k;
        int load_a_gmem_addr = load_gmem_a_m * INNER_DIM + load_a_gmem_k;
        uint32_t load_smem_a_addr = (smem_a_base_ptr + 
            (gmem_load_stage * s_a_stage_offset + load_smem_a_m * MMA_K + load_smem_a_k) * sizeof(half));
        CP_ASYNC_CG(load_smem_a_addr, &A[load_a_gmem_addr], 16);
        CP_ASYNC_COMMIT_GROUP(); 

        // Ldmatrix B (Operand B)
        int lane_smem_b_n = lane_id % 8;
        int lane_smem_b_k = lane_id / 8 * 8;
        uint32_t lane_smem_b_addr = smem_b_base_ptr + 
            (lane_smem_b_n * INNER_DIM + (b_stage * 16 + lane_smem_b_k)) * sizeof(half);
        LDMATRIX_X2(RB[0], RB[1], lane_smem_b_addr);
        b_stage ++;

        // Ldmatrix A (Operand A)
        // Warp 0 负责 A 的 0-15 行。
        // lane_smem_a_m = 0 + lane_id % 16 (范围 0-15)
        int lane_smem_a_m = warp_id * MMA_M + lane_id % 16;
        int lane_smem_a_k = lane_id / 16 * 8;
        uint32_t lane_smem_a_addr = smem_a_base_ptr + 
            (compute_stage * s_a_stage_offset + lane_smem_a_m * MMA_K + lane_smem_a_k) * sizeof(half);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], lane_smem_a_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
        
        // Norm 累加
        {
            half2 val_h2 = reinterpret_cast<half2&>(RA[0]);
            RA_NORM_H2[0] = __hfma2(val_h2, val_h2, RA_NORM_H2[0]);
            val_h2 = reinterpret_cast<half2&>(RA[2]);
            RA_NORM_H2[0] = __hfma2(val_h2, val_h2, RA_NORM_H2[0]);
            val_h2 = reinterpret_cast<half2&>(RA[1]);
            RA_NORM_H2[1] = __hfma2(val_h2, val_h2, RA_NORM_H2[1]);
            val_h2 = reinterpret_cast<half2&>(RA[3]);
            RA_NORM_H2[1] = __hfma2(val_h2, val_h2, RA_NORM_H2[1]);
        }
        {
            half2 val_h2 = reinterpret_cast<half2&>(RB[0]);
            RB_NORM_H2 = __hfma2(val_h2, val_h2, RB_NORM_H2);
            val_h2 = reinterpret_cast<half2&>(RB[1]);
            RB_NORM_H2 = __hfma2(val_h2, val_h2, RB_NORM_H2);
        }

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }

    // ========================================================================
    // Pipeline Cleanup
    // ========================================================================
    if ((K_STAGE - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // 计算剩下的 Stage
    for(int k = 0; k < (K_STAGE - 1); k++){
        int remain_compute_stage = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);

        int lane_smem_b_n = lane_id % 8;
        int lane_smem_b_k = lane_id / 8 * 8;
        uint32_t lane_smem_b_addr = smem_b_base_ptr + 
            (lane_smem_b_n * INNER_DIM + (b_stage * 16 + lane_smem_b_k)) * sizeof(half);
        LDMATRIX_X2(RB[0], RB[1], lane_smem_b_addr);
        b_stage ++;

        int lane_smem_a_m = warp_id * MMA_M + lane_id % 16;
        int lane_smem_a_k = lane_id / 16 * 8;
        uint32_t lane_smem_a_addr = smem_a_base_ptr + 
            (remain_compute_stage * s_a_stage_offset + lane_smem_a_m * MMA_K + lane_smem_a_k) * sizeof(half);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], lane_smem_a_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

        {
            half2 val_h2 = reinterpret_cast<half2&>(RA[0]);
            RA_NORM_H2[0] = __hfma2(val_h2, val_h2, RA_NORM_H2[0]);
            val_h2 = reinterpret_cast<half2&>(RA[2]);
            RA_NORM_H2[0] = __hfma2(val_h2, val_h2, RA_NORM_H2[0]);
            val_h2 = reinterpret_cast<half2&>(RA[1]);
            RA_NORM_H2[1] = __hfma2(val_h2, val_h2, RA_NORM_H2[1]);
            val_h2 = reinterpret_cast<half2&>(RA[3]);
            RA_NORM_H2[1] = __hfma2(val_h2, val_h2, RA_NORM_H2[1]);
        }
        {
            half2 val_h2 = reinterpret_cast<half2&>(RB[0]);
            RB_NORM_H2 = __hfma2(val_h2, val_h2, RB_NORM_H2);
            val_h2 = reinterpret_cast<half2&>(RB[1]);
            RB_NORM_H2 = __hfma2(val_h2, val_h2, RB_NORM_H2);
        }
    }

    // ========================================================================
    // Reduction & Epilogue
    // ========================================================================
    float RA_NORM[2];
    {
        float2 t = __half22float2(RA_NORM_H2[0]);
        RA_NORM[0] = t.x + t.y;
        t = __half22float2(RA_NORM_H2[1]);
        RA_NORM[1] = t.x + t.y;
    }
    float RB_NORM;
    {
        float2 t = __half22float2(RB_NORM_H2);
        RB_NORM = t.x + t.y;
    }

    // Warp Reduction
    // 只有一个 Warp，所以不需要 __syncthreads() 和 Shared Memory 交换
    {
        float tmp = RA_NORM[0];
        tmp += __shfl_xor_sync(0xffffffff, tmp, 1);
        tmp += __shfl_xor_sync(0xffffffff, tmp, 2);
        RA_NORM[0] = tmp;

        tmp = RA_NORM[1];
        tmp += __shfl_xor_sync(0xffffffff, tmp, 1);
        tmp += __shfl_xor_sync(0xffffffff, tmp, 2);
        RA_NORM[1] = tmp;
    }
    {
        float tmp = RB_NORM;
        tmp += __shfl_xor_sync(0xffffffff, tmp, 1);
        tmp += __shfl_xor_sync(0xffffffff, tmp, 2);
        RB_NORM = tmp;
    }

    // Warp Shuffle 交换结果 (用于计算余弦)
    uint32_t RC_shfl[2][4];
    RC_shfl[0][0] = RC[0];
    RC_shfl[1][0] = RC[1];
    RC_shfl[0][1] = __shfl_sync((0xffffffff), RC[0], lane_id + 1);
    RC_shfl[0][2] = __shfl_sync((0xffffffff), RC[0], lane_id + 2);
    RC_shfl[0][3] = __shfl_sync((0xffffffff), RC[0], lane_id + 3);
    RC_shfl[1][1] = __shfl_sync((0xffffffff), RC[1], lane_id + 1);
    RC_shfl[1][2] = __shfl_sync((0xffffffff), RC[1], lane_id + 2);
    RC_shfl[1][3] = __shfl_sync((0xffffffff), RC[1], lane_id + 3);

    float rsqrt_a_norm0 = rsqrtf(RA_NORM[0]);
    float rsqrt_a_norm1 = rsqrtf(RA_NORM[1]);
#pragma unroll
    for (int k = 0; k < 4; k++) {
        half2 val_h2_0 = reinterpret_cast<half2&>(RC_shfl[0][k]);
        float2 val_f2_0 = __half22float2(val_h2_0);
        float b_norm0_f = __shfl_sync((0xffffffff), RB_NORM, 4 * (2 * k));
        float b_norm1_f = __shfl_sync((0xffffffff), RB_NORM, 4 * (2 * k + 1));
        float rsqrt_b_norm0 = rsqrtf(b_norm0_f);
        float rsqrt_b_norm1 = rsqrtf(b_norm1_f);
        float res0_f = val_f2_0.x * rsqrt_a_norm0 * rsqrt_b_norm0;
        float res1_f = val_f2_0.y * rsqrt_a_norm0 * rsqrt_b_norm1;
        half2 res_h2 = __float22half2_rn(make_float2(res0_f, res1_f));
        RC_shfl[0][k] = reinterpret_cast<uint32_t&>(res_h2);

        half2 val_h2_1 = reinterpret_cast<half2&>(RC_shfl[1][k]);
        float2 val_f2_1 = __half22float2(val_h2_1);
        float res2_f = val_f2_1.x * rsqrt_a_norm1 * rsqrt_b_norm0;
        float res3_f = val_f2_1.y * rsqrt_a_norm1 * rsqrt_b_norm1;
        half2 res_h2_1 = __float22half2_rn(make_float2(res2_f, res3_f));
        RC_shfl[1][k] = reinterpret_cast<uint32_t&>(res_h2_1);
    }

    // Store Result to Global Memory
    if(lane_id % 4 == 0){
        // warp_id = 0
        int store_smem_c_m = warp_id * MMA_M + lane_id / 4; 
        int store_smem_c_n = 0;
        int store_gmem_c_m = by * MMA_M * MMA_TILE_M + store_smem_c_m;
        
        // 补充定义 n 坐标
        int store_gmem_c_n = store_smem_c_n;

        // 边界检查 (M 可能不是 16 的倍数)
        if (store_gmem_c_m < M) {
            int store_gmem_c_addr0 = store_gmem_c_m * MMA_N + store_gmem_c_n;
            int store_gmem_c_addr1 = (store_gmem_c_m + 8) * MMA_N + store_gmem_c_n;
            
            LDST128BITS(C[store_gmem_c_addr0]) = LDST128BITS(RC_shfl[0][0]);
            
            // 检查 +8 是否越界
            if (store_gmem_c_m + 8 < M) {
                LDST128BITS(C[store_gmem_c_addr1]) = LDST128BITS(RC_shfl[1][0]);
            }
        }
    }
}

template <const int K_STAGE = 2>
void launch_cosine_kernel_tn_16(half *a, half *b, half *c, int M, int N, int K)
{
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int INNER_DIM = 128;
    
    // 修改: Tile M 设为 1，即 16 行
    constexpr int MMA_TILE_M = 1; 
    constexpr int A_PAD = 0;
    constexpr int B_PAD = 0;
    constexpr int BM = MMA_M * MMA_TILE_M;
    
    // Smem 计算
    // A: 16*16 * 2bytes * K_STAGE
    // B: 8*128 * 2bytes = 2048 bytes
    const int smem_size = MMA_N * INNER_DIM * sizeof(half) + 
                          K_STAGE * MMA_M * MMA_TILE_M * INNER_DIM * sizeof(half);
    
    cudaFuncSetAttribute(
        cosine_kernel_tn_16x128<MMA_M, MMA_N, MMA_K, INNER_DIM, MMA_TILE_M, A_PAD, B_PAD, K_STAGE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size + 1024);

    dim3 block(32); // 修改: 仅 32 线程
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, BM));

    // 这个代码将block size改为32，一个block处理16*128的A tile
    cosine_kernel_tn_16x128<MMA_M, MMA_N, MMA_K, INNER_DIM, MMA_TILE_M, A_PAD, B_PAD, K_STAGE>
        <<<grid, block, smem_size>>>(a, b, c, M, N, K);
}

#ifndef BUILD_PYTORCH_EXTENSION
int main(int argc, char *argv[])
{
    const int test_num = 1;
    int M_list[test_num];
    int N_list[test_num];
    int K_list[test_num];

    M_list[0] = 250000;
    N_list[0] = 8;
    K_list[0] = 128;

    int outer_repeat = 10, inner_repeat = 1, warmup = 1;

    printf("ALGO = COSINE SIMILARITY TN MMA=16x8x16 (BLOCK 32) WARP=1 STAGES=3\n");

    for (int j = 0; j < test_num; j++)
    {
        int M = M_list[j], N = N_list[j], K = K_list[j];

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int k = 0; k < outer_repeat; k++)
        {
            // 使用 launch_cosine_kernel_tn_16
            double this_sec = perf_gemm<half>(launch_cosine_kernel_tn_16<3>,
                                              M, N, K, inner_repeat, warmup);
            max_sec = max(max_sec, this_sec);
            min_sec = min(min_sec, this_sec);
            total_sec += this_sec;
        }

        double avg_sec = total_sec / outer_repeat;
        double avg_Tflops = ((double)M) * N * K * 2 * 1e-12 / avg_sec;

        printf("M N K = %6d %6d %6d, W = %1d, R = %2d ", M, N, K, warmup,
               inner_repeat);
        printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
        printf("AVG Performance = %10.4lf Tflops\n", avg_Tflops);
    }

    return 0;
}
#endif
