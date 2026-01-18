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
#define LDST32BITS(value) (reinterpret_cast<half2 *>(&(value))[0])
#define LDST128BITS(value) (reinterpret_cast<float4 *>(&(value))[0])

// CP_ASYNC definitions
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))

// cp.async.cg (Cache Global, bypassing L1) 通常用于 16 字节
#define CP_ASYNC_CG(dst, src, bytes)                                       \
    asm volatile(                                                          \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(bytes))

// cp.async.ca (Cache All, L1+L2) 支持 4, 8, 16 字节
// 我们用这个来处理 8 字节的 B 矩阵加载
#define CP_ASYNC_CA(dst, src, bytes)                                       \
    asm volatile(                                                          \
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(bytes))

// ldmatrix definitions
#define LDMATRIX_X2(R0, R1, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                 : "=r"(R0), "=r"(R1)                                         \
                 : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                    \
    asm volatile(                                                            \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
        : "r"(addr))

// mma definitions
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)             \
    asm volatile(                                                               \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, " \
        "%4, %5}, {%6, %7}, {%8, %9};\n"                                        \
        : "=r"(RD0), "=r"(RD1)                                                  \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), \
          "r"(RC1))

HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// 修改点: 
// 1. Block Size -> 256
// 2. MMA_TILE_M -> 8 (对应 128 行)
// 3. A_PAD 建议设为 8 以减少 Bank Conflict (可选)
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
            const int INNER_DIM = 128, const int MMA_TILE_M = 8,
          const int A_PAD = 8, const int B_PAD = 0, const int K_STAGE = 3>
__global__ void __launch_bounds__(256)
    cosine_kernel_tn(half *A, half *B,
                  half *C, int M,
                  int N, int K)
{
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);
    extern __shared__ half smem[];
    half *s_b = smem;
    // A 矩阵的 smem 起始地址
    half *s_a = s_b + MMA_N * INNER_DIM;
    
    const int s_a_stage_offset = MMA_M  * MMA_TILE_M * (INNER_DIM + A_PAD);
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

    // ----------------------------------------------------------------
    // 1. Prologue: 加载 B 矩阵 (一次性)
    // ----------------------------------------------------------------
    // 方案: 256 线程，全员参与，每人加载 8 Bytes (4 halfs)
    // 映射: 32线程(1个Warp)负责一行(128 halfs)
    int load_smem_b_n = tid / 32;       // 0 ~ 7 行
    int load_smem_b_k = (tid % 32) * 4; // 0 ~ 124 列
    
    // 计算 B 的全局和共享内存地址
    // 注意: B 矩阵很小且常驻，这里即使后半部分线程计算出的地址超出 N 范围
    // 在实际矩阵乘中通常 N=8 是固定的。如果 N 可能很大，这里需要边界检查。
    // 但根据题目描述 N=8，所以 tid/32 最大是 7，不会越界。
    int load_gmem_b_addr = load_smem_b_n * INNER_DIM + load_smem_b_k;
    uint32_t load_smem_b_addr = smem_b_base_ptr + load_gmem_b_addr * sizeof(half);
    
    // 使用 CP_ASYNC_CA (Cache All) 因为 size=8
    CP_ASYNC_CA(load_smem_b_addr, &B[load_gmem_b_addr], 8);
    // B 矩阵不需要 commit，因为它只加载一次，只要在计算前 wait 即可

    // ----------------------------------------------------------------
    // 2. Prologue: 加载 A 矩阵
    // ----------------------------------------------------------------
    // 方案: 256 线程，每人加载 16 Bytes (8 halfs)
    // 总量: 256 * 16 = 4096 Bytes = 128行 * 16列 * 2字节 (正好是一个 K_TILE)
    // 映射: 256线程覆盖 128 行，每两行分配给 4 个线程？
    // 最简单的线性映射:
    int load_smem_a_m = tid / 2;      // 0 ~ 127 行
    int load_smem_a_k = (tid % 2) * 8;// 0 或 8 (列偏移)

    int load_gmem_a_m = by * MMA_M * MMA_TILE_M + load_smem_a_m;
    
    // 越界检查
    if (load_gmem_a_m >= M) return;

#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); ++k) {
        int load_gmem_a_k = k * MMA_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * INNER_DIM + load_gmem_a_k;
        
        // 计算带 padding 的 smem 地址
        uint32_t load_smem_a_addr = (smem_a_base_ptr + 
            (k * s_a_stage_offset + load_smem_a_m * (MMA_K + A_PAD) + load_smem_a_k) * sizeof(half));
        
        // A 使用 CG 加载 16 字节
        CP_ASYNC_CG(load_smem_a_addr, &A[load_gmem_a_addr], 16);
        CP_ASYNC_COMMIT_GROUP(); 
    }
    
    // 等待 B 加载完成 (B 在第 0 个 group 之前发射，或者被归入第一个 commit)
    // 上面 B 发射后没调用 commit，它会被归入循环第一次 commit 的组里。
    // 所以 wait(K_STAGE - 2) 是安全的，只要 K_STAGE >= 2。
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    // ----------------------------------------------------------------
    // 3. Main Loop
    // ----------------------------------------------------------------
    
    uint32_t RA[4];
    uint32_t RB[2];
    uint32_t RC[2] = {0, 0};
    
    // 使用 half2 累加器
    half2 RA_NORM_H2[2] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f)};
    half2 RB_NORM_H2 = __float2half2_rn(0.0f);

    int b_stage = 0;

#pragma unroll
    for(int k = (K_STAGE - 1); k < NUM_K_TILES; k++){
        int compute_stage = (k + 1) % K_STAGE;
        int gmem_load_stage = k % K_STAGE;
        
        // 同样，256 线程加载 A，每人 16 字节
        int load_a_gmem_k = k * MMA_K + load_smem_a_k;
        int load_a_gmem_addr = load_gmem_a_m * INNER_DIM + load_a_gmem_k;
        
        uint32_t load_smem_a_addr = (smem_a_base_ptr + 
            (gmem_load_stage * s_a_stage_offset + load_smem_a_m * (MMA_K + A_PAD) + load_smem_a_k) * sizeof(half));
        
        CP_ASYNC_CG(load_smem_a_addr, &A[load_a_gmem_addr], 16);
        CP_ASYNC_COMMIT_GROUP(); 

        // ------------------------------------------------------------
        // LDMATRIX B: 8x16 块
        // ------------------------------------------------------------
        // Lane ID 映射保持不变，Warp 内逻辑
        int lane_smem_b_n = lane_id % 8;
        int lane_smem_b_k = lane_id / 8 * 8;
        // B 在 Smem 中没有 Padding
        uint32_t lane_smem_b_addr = smem_b_base_ptr + 
            (lane_smem_b_n * INNER_DIM + (b_stage * 16 + lane_smem_b_k)) * sizeof(half);
        LDMATRIX_X2(RB[0], RB[1], lane_smem_b_addr);
        b_stage ++;

        // ------------------------------------------------------------
        // LDMATRIX A: 128x16 块 -> 分给 8 个 Warp
        // ------------------------------------------------------------
        // 256 threads = 8 Warps. MMA_M = 16.
        // Warp 0: rows 0-15
        // Warp 7: rows 112-127
        // lane_smem_a_m = warp_id * 16 + (0..15)
        // 这里的 range 是 0 ~ 127，完美覆盖 128 行
        int lane_smem_a_m = warp_id * MMA_M + lane_id % 16;
        int lane_smem_a_k = lane_id / 16 * 8;
        
        uint32_t lane_smem_a_addr = smem_a_base_ptr + 
            (compute_stage * s_a_stage_offset + lane_smem_a_m * (MMA_K + A_PAD) + lane_smem_a_k) * sizeof(half);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], lane_smem_a_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
        
        // 累加 Norm (Half2)
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

    // ----------------------------------------------------------------
    // 4. Epilogue (Wait remaining async)
    // ----------------------------------------------------------------
    if ((K_STAGE - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

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
            (remain_compute_stage * s_a_stage_offset + lane_smem_a_m * (MMA_K + A_PAD) + lane_smem_a_k) * sizeof(half);
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

    // ----------------------------------------------------------------
    // 5. Reduction & Store
    // ----------------------------------------------------------------
    // 将累加的half2转回float以进行规约
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

    // Warp 内规约
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

    // 计算最终结果
    float rsqrt_a_norm0 = rsqrtf(RA_NORM[0]);
    float rsqrt_a_norm1 = rsqrtf(RA_NORM[1]);
    
    // Warp Shuffle 交换 B 的 Norm (逻辑保持不变)
    // 注意: 虽然 Block Size 变了，但是这里是 Warp 内部的 shuffle，
    // 每个 Warp 处理独立的 C 矩阵块 (16x8)，所以逻辑通用。
    
    // 这里的 RC0, RC1 对应当前 Warp 负责的 16 行结果
    // 重新准备用于 shuffle 的寄存器
    uint32_t RC_shfl[2][4]; 
    RC_shfl[0][0] = RC[0];
    RC_shfl[1][0] = RC[1];
    
    // 修正: 原代码这里的 shuffle 逻辑是为了让线程 0 拿到 0,1,2,3 号线程的数据
    // 用于后面的 STMATRIX 或者手动存储。
    // 在 Block=256 下，每个 Warp 依然是处理 16x8 的块，这个结构没变。
    RC_shfl[0][1] = __shfl_sync((0xffffffff), RC[0], lane_id + 1);
    RC_shfl[0][2] = __shfl_sync((0xffffffff), RC[0], lane_id + 2);
    RC_shfl[0][3] = __shfl_sync((0xffffffff), RC[0], lane_id + 3);
    RC_shfl[1][1] = __shfl_sync((0xffffffff), RC[1], lane_id + 1);
    RC_shfl[1][2] = __shfl_sync((0xffffffff), RC[1], lane_id + 2);
    RC_shfl[1][3] = __shfl_sync((0xffffffff), RC[1], lane_id + 3);

#pragma unroll
    for (int k = 0; k < 4; k++) {
        half2 val_h2_0 = reinterpret_cast<half2&>(RC_shfl[0][k]);
        float2 val_f2_0 = __half22float2(val_h2_0);
        
        // RB_NORM 也是在 Warp 内 shuffle
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

    // Store
    if(lane_id % 4 == 0){
        // warp_id 现在是 0~7
        // lane_id/4 覆盖 0~7 行
        // 总共覆盖 8 * 16 = 128 行
        int store_smem_c_m = warp_id * MMA_M + lane_id / 4; 
        int store_smem_c_n = 0;
        int store_gmem_c_m = by * MMA_M * MMA_TILE_M + store_smem_c_m;
        int store_gmem_c_n = store_smem_c_n;

        if (store_gmem_c_m < M) { // 简单边界检查
            int store_gmem_c_addr0 = store_gmem_c_m * MMA_N + store_gmem_c_n;
            int store_gmem_c_addr1 = (store_gmem_c_m + 8) * MMA_N + store_gmem_c_n;
            
            // 下半部分可能越界，需要注意，但通常 M 是 16 的倍数
            LDST128BITS(C[store_gmem_c_addr0]) = LDST128BITS(RC_shfl[0][0]);
            LDST128BITS(C[store_gmem_c_addr1]) = LDST128BITS(RC_shfl[1][0]);
        }
    }
}

template <const int K_STAGE = 2>
void launch_cosine_kernel_tn(half *a, half *b, half *c, int M, int N, int K)
{
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int INNER_DIM = 128;
    constexpr int MMA_TILE_M = 8; // 8 * 16 = 128 rows
    constexpr int A_PAD = 8;      // Padding
    constexpr int B_PAD = 0;
    
    // 重新计算 Smem
    // A: 128行 * (128+8)列 * 2 bytes * K_STAGE
    // B: 8行 * 128列 * 2 bytes
    const int smem_size = MMA_N * INNER_DIM * sizeof(half) + 
                          K_STAGE * MMA_M * MMA_TILE_M * (INNER_DIM + A_PAD) * sizeof(half);
    
    cudaFuncSetAttribute(
        cosine_kernel_tn<MMA_M, MMA_N, MMA_K, INNER_DIM, MMA_TILE_M, A_PAD, B_PAD, K_STAGE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size + 2048);

    dim3 block(256); // 256 threads
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, MMA_M * MMA_TILE_M));

    // 这个代码讲block size改为256，一个block处理128*128的A tile
    cosine_kernel_tn<MMA_M, MMA_N, MMA_K, INNER_DIM, MMA_TILE_M, A_PAD, B_PAD, K_STAGE>
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

    printf("ALGO = COSINE SIMILARITY TN UNROLLED x2\n");

    for (int j = 0; j < test_num; j++)
    {
        int M = M_list[j], N = N_list[j], K = K_list[j];

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int k = 0; k < outer_repeat; k++)
        {
            // Try K_STAGE = 4 for double buffering the double-issue
            double this_sec = perf_gemm<half>(launch_cosine_kernel_tn<3>,
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