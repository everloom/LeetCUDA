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
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                       \
    asm volatile(                                                          \
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                       \
    asm volatile(                                                          \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(bytes))
// smem -> gmem: requires sm_90 or higher.
#define CP_ASYNC_BULK_COMMIT_GROUP() \
    asm volatile("cp.async.bulk.commit_group;\n" ::)
#define CP_ASYNC_BULK_WAIT_ALL() asm volatile("cp.async.bulk.wait_all;\n" ::)
#define CP_ASYNC_BULK_WAIT_GROUP(n) \
    asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_BULK(dst, src, bytes)                                      \
    asm volatile(                                                           \
        "cp.async.bulk.global.shared::cta.bulk_group.L2::128B [%0], [%1], " \
        "%2;\n" ::"r"(dst),                                                 \
        "l"(src), "n"(bytes))
// ldmatrix
#define LDMATRIX_X1(R, addr)                                              \
    asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n" \
                 : "=r"(R)                                                \
                 : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr)                                             \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" \
                 : "=r"(R0), "=r"(R1)                                         \
                 : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                    \
    asm volatile(                                                            \
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                             \
        : "r"(addr))
#define LDMATRIX_X1_T(R, addr)                                                  \
    asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n" \
                 : "=r"(R)                                                      \
                 : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr)                                        \
    asm volatile(                                                          \
        "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n" \
        : "=r"(R0), "=r"(R1)                                               \
        : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr)                                 \
    asm volatile(                                                           \
        "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, " \
        "[%4];\n"                                                           \
        : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                            \
        : "r"(addr))
// stmatrix: requires sm_90 or higher.
#define STMATRIX_X1(addr, R)                                                  \
    asm volatile(                                                             \
        "stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" ::"r"(addr), \
        "r"(R))
#define STMATRIX_X2(addr, R0, R1)                                           \
    asm volatile(                                                           \
        "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"( \
            addr),                                                          \
        "r"(R0), "r"(R1))
#define STMATRIX_X4(addr, R0, R1, R2, R3)                                       \
    asm volatile(                                                               \
        "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" :: \
            "r"(addr),                                                          \
        "r"(R0), "r"(R1), "r"(R2), "r"(R3))
#define STMATRIX_X1_T(addr, R)                                                \
    asm volatile(                                                             \
        "stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n" ::"r"( \
            addr),                                                            \
        "r"(R))
#define STMATRIX_X2_T(addr, R0, R1)                                           \
    asm volatile(                                                             \
        "stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n" :: \
            "r"(addr),                                                        \
        "r"(R0), "r"(R1))
#define STMATRIX_X4_T(addr, R0, R1, R2, R3)                                  \
    asm volatile(                                                            \
        "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, " \
        "%4};\n" ::"r"(addr),                                                \
        "r"(R0), "r"(R1), "r"(R2), "r"(R3))
// mma m16n8k16
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)             \
    asm volatile(                                                               \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, " \
        "%4, %5}, {%6, %7}, {%8, %9};\n"                                        \
        : "=r"(RD0), "=r"(RD1)                                                  \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), \
          "r"(RC1))
// mma m8n8k4
#define HMMA884(RD0, RD1, RD2, RD3, RA0, RA1, RB0, RB1, RC0, RC1, RC2, RC3)     \
    asm volatile(                                                               \
        "mma.sync.aligned..m8n8k4.row.col.f16.f16.f16.f16 {%0, %1, %2, %3}, "   \
        "{%4, %5}, {%6, %7}, {%8, %9, %10, %11};\n"                             \
        : "=r"(RD0), "=r"(RD1), "=r"(RD2), "=r"(RD3)                            \
        : "r"(RA0), "r"(RA1), "r"(RB0), "r"(RB1), "r"(RC0), "r"(RC1), "r"(RC2), \
          "r"(RC3))

HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// 这个kernel的输入A矩阵的shape是250000*128的行主序，B矩阵是8*128的行主序
// 这个函数本质上是想要计算8*128和128*250000这两个矩阵的乘积，得到8*250000的矩阵
// 但由于传入的两个矩阵是8*128的行主序和250000*128的行主序
// 根据C = A * B可以转换成C^T = B^T * A^T
// 这里就转换成了计算B^T和A^T的矩阵乘
// B^T就是最开始说的A矩阵，shape是250000*128的行主序
// 而A^T则是通过在ldmatrix是控制一个warp访问数据位置来实现的，在代码中A^T为传入的B矩阵，shape是8*128的行主序
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
            const int NUM_M_TILES=100, const int INNER_DIM = 128,
          const int A_PAD = 0, const int B_PAD = 0, const int M_STAGE = 3>
__global__ void __launch_bounds__(256)
    cosine_kernel_tn(half *A, half *B,
                  half *C, int M,
                  int N, int K)
{
    // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
    const int bx = blockIdx.x;
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);
    extern __shared__ half smem[];
    half *s_b = smem;
    half *s_a = s_b + MMA_N * INNER_DIM;
    half *s_c = s_a + M_STAGE * MMA_M * (INNER_DIM + B_PAD);
    const int s_b_offset = MMA_N * INNER_DIM;
    const int s_a_stage_offset = MMA_M * (INNER_DIM + B_PAD);
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);
    uint32_t smem_c_base_ptr = __cvta_generic_to_shared(s_c);


    // %号是连续变化，/ 是间断变化
    // 需要注意的是cp async的smem坐标计算方式与ldmatrix的smem坐标计算方式不同
    // cp async的smem计算需要考虑到gmem的合并访存，而ldmatrix的坐标计算主要为了迎合mma对a b矩阵的要求
    int load_smem_a_m = tid / 16; 
    int load_smem_a_k = tid % 16; // 0 ~ 15
    int load_smem_b_n = tid / 32;
    int load_smem_b_k = tid % 32; // 0 ~ 7

    // 之所以这里计算b的gmem addr能直接用smem的n和k来计算，而不用加上global的偏移
    // 是因为对于b而言，gmem中的数据的shape和smem中的shape完全一致
    int load_gmem_b_addr = load_smem_b_n * INNER_DIM + load_smem_b_k * MMA_K;
    int load_smem_b_addr = smem_b_base_ptr + load_gmem_b_addr * sizeof(half);
    // 这里需要注意，cp async传的smem地址需要是uint32_t类型，而这里的load_gmem_b_addr是gmem偏移、是int类型的
    CP_ASYNC_CG(load_smem_b_addr, &B[load_gmem_b_addr], 8); // MMA_K 1
    #pragma unroll
    for (int m = 0; m < (M_STAGE - 1); ++m) {     // 0, 1
        int load_gmem_a_m = by * NUM_M_TILES * MMA_M + load_smem_a_m;
        int load_gmem_a_addr = load_gmem_a_m * INNER_DIM + load_smem_a_k * MMA_K;
        uint32_t load_smem_a_addr = (smem_a_base_ptr + 
            (s_b_offset + m * s_a_stage_offset + load_smem_a_m * INNER_DIM + load_smem_a_k * MMA_K) * sizeof(half));
        CP_ASYNC_CG(load_smem_a_addr, &A[load_gmem_a_addr], 16);
        CP_ASYNC_COMMIT_GROUP();
    }
    // 说一下为什么这里是cp async wait group的参数传入的是stage-2
    // 参考reed大佬的那篇博客，以stage为5为例，前面已经提前发射了stage-1(这里为4)个异步拷贝任务。然后需要等到第一个异步拷贝任务完成之后，才能进入tile的计算
    // 而这里的wait参数表示的意思是，允许还有多少个stage-2个异步组处于未完成状态时就继续执行，也即这段话保证了至少有stage - 1 - (stage - 2)个异步拷贝任务是已经执行完了
    // stage - 1 - (stage - 2) = 1，正好符合reed大佬博客里面说的，在进入tile计算之前需要有一个异步拷贝任务完成的要求
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    uint32_t RA[4];
    uint32_t RB[2];
    float RA_NORM[2] = {0.0f, 0.0f};
    float RB_NORM = {0.0f};
    // ldmatrix载入b矩阵
    // 这里之所以不需要for循环是因为，这里的for循环已经被一个block的不同warp的simt做了
    int lane_smem_b_n = lane_id % 8;
    int load_smem_b_k = lane_id / 8 * 8;
    // 需要注意这里warp_id * MMA_k是k方向上的warp级别的偏移
    uint32_t lane_smem_b_ptr = smem_b_base_ptr + (lane_smem_b_n * INNER_DIM + load_smem_b_k + warp_id * MMA_k) * sizeof(half);
    // 注意这里传入smem地址同样需要uint32_t类型
    LDMATRIX_X2(RB[0], RB[1], lane_smem_b_ptr);

    

    // 下面两个变量用来行主序访问一个8*16的矩阵，连续的两个线程访问相同的行
    // int load_smem_b_m = (lane_id / 2 < 8) ? (lane_id / 2) : (lane_id / 2 - 8); // 0 ~ 7, 0 ~ 7
    // int load_smem_b_k = lane_id % 2 + warp_id * MMA_K; // 0, 1
    // // 下面两个变量用来行主序访问一个16*16的矩阵，连续的两个线程访问相同的行
    // int load_smem_a_m = lane_id / 2; // 0 ~ 15
    // int load_smem_a_k = lane_id % 2 + warp_id * MMA_K; // 0, 1
}

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem, TN
template <const int K_STAGE = 2, const int BLOCK_SWIZZLE_STRIDE = 2048>
void launch_gemm_kernel_tn(half *a, half *b, half *c, int M, int N, int K)
{
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int MMA_TILE_M = 2;
    constexpr int MMA_TILE_N = 4;
    constexpr int WARP_TILE_M = 4;
    constexpr int WARP_TILE_N = 4;
    constexpr int A_PAD = 0;
    constexpr int B_PAD = 0;
    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 128
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 128
    constexpr int BK = MMA_K;                            // 16

    // Shared memory size calculation
    const int smem_max_size = ((K_STAGE)*BM * (BK + A_PAD) * sizeof(half) +
                               (K_STAGE)*BN * (BK + B_PAD) * sizeof(half));

    cudaFuncSetAttribute(
        hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel<MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,
                                                                 WARP_TILE_N, A_PAD, B_PAD, K_STAGE, true>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    const int N_SWIZZLE = (N + BLOCK_SWIZZLE_STRIDE - 1) / BLOCK_SWIZZLE_STRIDE;
    dim3 block(256); // 2 * 4 * 32
    dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),
              N_SWIZZLE);

    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel<MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,
                                                             WARP_TILE_N, A_PAD, B_PAD, K_STAGE, true>
        <<<grid, block, smem_max_size>>>(a, b, c, M, N, K);
}

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem, TN
template <const int K_STAGE = 2>
void launch_cosine_kernel_tn(half *a, half *b, half *c, int M, int N, int K)
{
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int MMA_TILE_M = 2;
    constexpr int MMA_TILE_N = 4;
    constexpr int WARP_TILE_M = 4;
    constexpr int WARP_TILE_N = 4;
    constexpr int A_PAD = 0;
    constexpr int B_PAD = 0;
    constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 128
    constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 128
    constexpr int BK = MMA_K;                            // 16

    // Shared memory size calculation
    const int smem_max_size = ((K_STAGE)*BM * (BK + A_PAD) * sizeof(half) +
                               (K_STAGE)*BN * (BK + B_PAD) * sizeof(half));

    cudaFuncSetAttribute(
        cosine_kernel<MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,
                      WARP_TILE_N, A_PAD, B_PAD, K_STAGE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    const int N_SWIZZLE = (N + BLOCK_SWIZZLE_STRIDE - 1) / BLOCK_SWIZZLE_STRIDE;
    dim3 block(256); // 2 * 4 * 32
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));

    cosine_kernel<MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,
                  WARP_TILE_N, A_PAD, B_PAD, K_STAGE>
        <<<grid, block, smem_max_size>>>(a, b, c, M, N, K);
}

#ifndef BUILD_PYTORCH_EXTENSION
int main(int argc, char *argv[])
{
    const int test_num = 64;
    int M_list[test_num];
    int N_list[test_num];
    int K_list[test_num];

    for (int i = 0; i < test_num; i++)
    {
        M_list[i] = (i + 1) * 256;
        N_list[i] = (i + 1) * 256;
        K_list[i] = (i + 1) * 256;
    }

    if (argc > 1)
        M_list[0] = std::stoi(argv[1]);
    if (argc > 2)
        N_list[0] = std::stoi(argv[2]);
    if (argc > 3)
        K_list[0] = std::stoi(argv[3]);

    int outer_repeat = 10, inner_repeat = 1, warmup = 1;
    if (argc > 4)
        warmup = std::stoi(argv[4]);
    if (argc > 5)
        inner_repeat = std::stoi(argv[5]);

    printf("ALGO = COSINE SIMILARITY TN MMA=2x4 WARP=4x4 STAGES=2 BLOCK SWIZZLE=2048\n");

    // Note: Standard gemm_error_check_tn from utils.h checks C = A * B.
    // Cosine Similarity computes C = (A . B) / (|A| * |B|).
    // So standard GEMM verification will fail.
    // We skip correctness check here and focus on performance profiling.

    for (int j = 0; j < test_num; j++)
    {
        int M = M_list[j], N = N_list[j], K = K_list[j];

        double max_sec = 0.0;
        double min_sec = DBL_MAX;
        double total_sec = 0.0;

        for (int k = 0; k < outer_repeat; k++)
        {
            double this_sec = perf_gemm<half>(launch_cosine_kernel_tn<2>,
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