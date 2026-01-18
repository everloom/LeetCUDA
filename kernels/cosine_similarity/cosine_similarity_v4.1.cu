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
#define CP_ASYNC_CA_256(dst, src, bytes)                                       \
    asm volatile(                                                          \
        "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                       \
    asm volatile(                                                          \
        "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(bytes))
#define CP_ASYNC_CG_64(dst, src, bytes)                                       \
    asm volatile(                                                          \
        "cp.async.cg.shared.global.L2::64B [%0], [%1], %2;\n" ::"r"(dst), \
        "l"(src), "n"(bytes))
#define CP_ASYNC_CG_256(dst, src, bytes)                                       \
    asm volatile(                                                          \
        "cp.async.cg.shared.global.L2::256B [%0], [%1], %2;\n" ::"r"(dst), \
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
            const int INNER_DIM = 128, const int MMA_TILE_M = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 3>
__global__ void __launch_bounds__(128)
    cosine_kernel_tn(half *A, half *B,
                  half *C, int M,
                  int N, int K)
{
    // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
    const int by = blockIdx.y;
    const int NUM_K_TILES = div_ceil(K, MMA_K);
    extern __shared__ half smem[];
    half *s_b = smem;
    // 这里s_a中分为很多个stage，每个stage的大小为64*16
    half *s_a = s_b + MMA_N * INNER_DIM;
    // 这里注意下，虽然这里有s_b_stage_offset这个变量，但实际上b是不分stage的
    // 因为stage一般指的多级流水，但这里b是没有多级流水的
    // 这里有s_b_stage_offset这个变量的存在只是为了mainloop中ldmatrix时方便而已
    const int s_a_stage_offset = MMA_M  * MMA_TILE_M * (INNER_DIM + A_PAD);
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);


    // %号是连续变化，/ 是间断变化
    // 需要注意的是cp async的smem坐标计算方式与ldmatrix的smem坐标计算方式不同
    // cp async的smem计算需要考虑到gmem的合并访存，而ldmatrix的坐标计算主要为了迎合mma对a b矩阵的要求
    /*
    这里说一下代码中的load_smem_a_m / k 和 下面的lane_smem_a_m / k 的区别
    load_smem_a_m / k 是为了cp async载入smem时使用的坐标计算
    而 lane_smem_a_m / k 则是为了ldmatrix载入寄存器时使用的坐标计算
    由于一个block中不同warp执行gmem->smem和smem->reg时，每个线程需要访问的addr不同
    所以这里将cp async和ldmatrix时的smem_addr区分开来
    */
    int load_smem_a_m = tid / 2; 
    int load_smem_a_k = (tid % 2) * 8;
    int load_smem_b_n = tid / 16;
    int load_smem_b_k = (tid % 16) * 8;

    // 这里load_gmem_a_m提前算是因为，对于每个warp中的线程来说,load_gmem_a_m都是固定的，所以提前算就行
    int load_gmem_a_m = by * MMA_M * MMA_TILE_M + load_smem_a_m;

    if (load_gmem_a_m >= M || load_smem_b_n >= N)
        return;

    // 之所以这里计算b的gmem addr能直接用smem的n和k来计算，而不用加上global的偏移
    // 是因为对于b而言，gmem中的数据的shape和smem中的shape完全一致
    int load_gmem_b_addr = load_smem_b_n * INNER_DIM + load_smem_b_k;
    uint32_t load_smem_b_addr = smem_b_base_ptr + load_gmem_b_addr * sizeof(half);
    // 这里需要注意，cp async传的smem地址需要是uint32_t类型，而这里的load_gmem_b_addr是gmem偏移、是int类型的
    // CP_ASYNC_CG(load_smem_b_addr, &B[load_gmem_b_addr], 16);
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); ++k) {     // 0, 1
        int load_gmem_a_k = k * MMA_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * INNER_DIM + load_gmem_a_k;
        uint32_t load_smem_a_addr = (smem_a_base_ptr + 
            (k * s_a_stage_offset + load_smem_a_m * MMA_K + load_smem_a_k) * sizeof(half));
        // 一次cp async载入一个64*16的tile
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
    uint32_t RC[2] = {0, 0};
    
    // 修改变量声明：使用half2累加器，避免转float
    half2 RA_NORM_H2[2] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f)};
    half2 RB_NORM_H2 = __float2half2_rn(0.0f);

    // mainloop部分
    int b_stage = 0;
#pragma unroll
    for(int k = (K_STAGE - 1); k < NUM_K_TILES; k++){
        int compute_stage = (k + 1) % K_STAGE;
        int gmem_load_stage = k % K_STAGE;
        // 下面是发送a的cp async，不需要发送b是因为b在prologue阶段已经完全载入smem了
        // 这里不需要算load_a_gemm_m是因为，mainloop中是对k维度做循环，m维度是固定的
        int load_a_gmem_k = k * MMA_K + load_smem_a_k;
        int load_a_gmem_addr = load_gmem_a_m * INNER_DIM + load_a_gmem_k;
        uint32_t load_smem_a_addr = (smem_a_base_ptr + 
            (gmem_load_stage * s_a_stage_offset + load_smem_a_m * MMA_K + load_smem_a_k) * sizeof(half));
        CP_ASYNC_CG(load_smem_a_addr, &A[load_a_gmem_addr], 16);
        CP_ASYNC_COMMIT_GROUP(); 

        // // ldmatrix载入b矩阵
        int lane_smem_b_n = lane_id % 8;
        int lane_smem_b_k = lane_id / 8 * 8;
        // uint32_t lane_smem_b_addr = smem_b_base_ptr + 
        //     (b_stage * s_b_stage_offset + lane_smem_b_n * MMA_K + lane_smem_b_k * 8) * sizeof(half);
        uint32_t lane_smem_b_addr = smem_b_base_ptr + 
            (lane_smem_b_n * INNER_DIM + (b_stage * 16 + lane_smem_b_k)) * sizeof(half);
        LDMATRIX_X2(RB[0], RB[1], lane_smem_b_addr);
        b_stage ++;

        int lane_smem_a_m = warp_id * MMA_M + lane_id % 16;
        int lane_smem_a_k = lane_id / 16 * 8;
        uint32_t lane_smem_a_addr = smem_a_base_ptr + 
            (compute_stage * s_a_stage_offset + lane_smem_a_m * MMA_K + lane_smem_a_k) * sizeof(half);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], lane_smem_a_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
        
        // a_norm计算 (half2版本)
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

        // b_norm计算 (half2版本)
        {
            half2 val_h2 = reinterpret_cast<half2&>(RB[0]);
            RB_NORM_H2 = __hfma2(val_h2, val_h2, RB_NORM_H2);

            val_h2 = reinterpret_cast<half2&>(RB[1]);
            RB_NORM_H2 = __hfma2(val_h2, val_h2, RB_NORM_H2);
        }

        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }

    // make sure all memory issues ready.
    // 这里判断(K_STAGE - 2) > 0是因为，当k_stage > 2时会出现mainloop完了，但还有cp async在执行的情况
    // 所以这里需要针对k_stage > 2的情况执行一次wait
    if ((K_STAGE - 2) > 0) {
        CP_ASYNC_WAIT_GROUP(0);
        __syncthreads();
    }

    // 上面cp async wait执行完成之后，还有总共k_stage - 1个cp async的数据没有计算
    // 所以这里负责计算剩下的数据
    for(int k = 0; k < (K_STAGE - 1); k++){
        // 这个remain_compute_stage负责计算收尾部分的buffer的index
        // 前面主循环结束后，还有K_STAGE-1个buffer的数据没有计算
        // 这里以 NUM_K_TILES = 10, K_STAGE = 3为例
        // 对于上面的主循环，k的迭代流程如下：
        // k=2: 计算buffer1, 加载数据到buffer2  
        // k=3: 计算buffer2, 加载数据到buffer0
        // k=4: 计算buffer0, 加载数据到buffer1
        // k=5: 计算buffer1, 加载数据到buffer2
        // ...
        // k=8: 计算buffer2, 加载数据到buffer2  
        // k=9: 计算buffer0, 加载数据到buffer0  (最后一轮)
        // 主循环执行完成之后，还有k=8加载的buffer2，k=9加载的buffer0没有计算
        // 这里remain_compute_stage负责计算buffer2和buffer0的index
        // 当k=0时，remain_compute_stage = (10 - (3 - 1) + 0) % 3 = 2
        // 当k=1时，remain_compute_stage = (10 - (3 - 1) + 1) % 3 = 0
        int remain_compute_stage = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);

        // // ldmatrix载入b矩阵
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

        // a_norm计算 (half2版本)
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

        // b_norm计算 (half2版本)
        {
            half2 val_h2 = reinterpret_cast<half2&>(RB[0]);
            RB_NORM_H2 = __hfma2(val_h2, val_h2, RB_NORM_H2);

            val_h2 = reinterpret_cast<half2&>(RB[1]);
            RB_NORM_H2 = __hfma2(val_h2, val_h2, RB_NORM_H2);
        }
    }

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

    // a_norm的规约
    {
        float tmp = RA_NORM[0];
        // tmp += __shfl_sync(0xffffffff, RA_NORM[0], lane_id + 1);
        // tmp += __shfl_sync(0xffffffff, RA_NORM[0], lane_id + 2);
        // tmp += __shfl_sync(0xffffffff, RA_NORM[0], lane_id + 3);
        tmp += __shfl_xor_sync(0xffffffff, tmp, 1);
        tmp += __shfl_xor_sync(0xffffffff, tmp, 2);
        RA_NORM[0] = tmp;

        tmp = RA_NORM[1];
        // tmp += __shfl_sync(0xffffffff, RA_NORM[1], lane_id + 1);
        // tmp += __shfl_sync(0xffffffff, RA_NORM[1], lane_id + 2);
        // tmp += __shfl_sync(0xffffffff, RA_NORM[1], lane_id + 3);
        tmp += __shfl_xor_sync(0xffffffff, tmp, 1);
        tmp += __shfl_xor_sync(0xffffffff, tmp, 2);
        RA_NORM[1] = tmp;
    }

    // b_norm的规约
    {
        float tmp = RB_NORM;
        // tmp += __shfl_sync(0xffffffff, RB_NORM, lane_id + 1);
        // tmp += __shfl_sync(0xffffffff, RB_NORM, lane_id + 2);
        // tmp += __shfl_sync(0xffffffff, RB_NORM, lane_id + 3);
        tmp += __shfl_xor_sync(0xffffffff, tmp, 1);
        tmp += __shfl_xor_sync(0xffffffff, tmp, 2);
        RB_NORM = tmp;
    }

    // epilogue部分
    uint32_t RC0[4];
    uint32_t RC1[4];
    RC0[0] = RC[0];
    RC1[0] = RC[1];
    RC0[1] = __shfl_sync((0xffffffff), RC[0], lane_id + 1);
    RC0[2] = __shfl_sync((0xffffffff), RC[0], lane_id + 2);
    RC0[3] = __shfl_sync((0xffffffff), RC[0], lane_id + 3);
    RC1[1] = __shfl_sync((0xffffffff), RC[1], lane_id + 1);
    RC1[2] = __shfl_sync((0xffffffff), RC[1], lane_id + 2);
    RC1[3] = __shfl_sync((0xffffffff), RC[1], lane_id + 3);


//     /*
//     说一下这里的代码逻辑，主要实现了将矩阵乘的结果RC除以对应的L2范数的乘积
//     即计算了这样的公式: RC / (sqrt(RA_NORM) * sqrt(RB_NORM))
//     首先这里之所以有lane_id % 4 == 0的判断，是因为一个warp中只有0，4，8，。。。线程持有一个16*8的结果（j=0）
//     这里以线程0为例，前面说了，RC0[0，1，2，3]持有第0行的8个half，RC1[0，1，2，3]持有第8行的8个half（看不懂的去看ptx文档上的那个图）
//     然后我们需要将RC0[0，1，2，3]和RC1[0，1，2，3]中的结果除以范数
//     首先对于RC0[0]，RC0[0]实际上是有两个half，对应了16*8矩阵的[0][0]和[0][1]位置的结果，它需要除以第A矩阵第0行的范数和B矩阵第0/1列的范数开根号的乘积
//         对于A矩阵第0行的平方和，其实就是RA_NORM[0]；而对于B矩阵的第0列和第1列的平方和，分别分布在第0号线程和第4号线程的RA_NORM中，所以你可以看到这里使用了warp shuffle来从其他线程拿RB_NORM[j]
//     然后对于RC0[1]，RC0[1]实际上是有两个half，对应了16*8矩阵的[0][2]和[0][3]位置的结果，它需要除以第A矩阵第0行的范数和B矩阵第2/3列的范数开根号的乘积
//         对于A矩阵第0行的平方和，同样是RA_NORM[0]；而对于B矩阵的第2列和第3列的平方和，分别分布在第8号线程和第·1号线程的RA_NORM中，所以你可以看到这里使用了warp shuffle来从其他线程拿RB_NORM[j]
//     以此类推
//     然后就是关于warp shuffle中的这个4 * (2 * k)和4 * (2 * k + 1)咋来的，这里需要你对照着ptx文档中B矩阵和C矩阵的layout图来理解了
//         假设k=0，那么4 * (2 * k)=0，4 * (2 * k + 1)=4，刚好对应了B矩阵的第0列和第1列的平方和所在的线程
//         假设k=1，那么4 * (2 * k)=8，4 * (2 * k + 1)=12，刚好对应了B矩阵的第2列和第3列的平方和所在的线程
//         以此类推
//     然后就计算余弦相似度，然后再将结果存放回RC寄存器中
//     对于RC1[0,1,2,3]的处理同理，只不过处理RC1[0,1,2,3]时，使用的是A矩阵的第8行的平方和，其实就是RA_NORM[1]

//     还有这里之所以全用float类型，我是看gemini 3pro说，sm80没有针对half的开根号指令，直接对half使用开根号的话，会先将half转为float再开根号，所以这里干脆直接全用float了
//     */
    float rsqrt_a_norm0 = rsqrtf(RA_NORM[0]);
    float rsqrt_a_norm1 = rsqrtf(RA_NORM[1]);
#pragma unroll
    for (int k = 0; k < 4; k++) {
        half2 val_h2_0 = reinterpret_cast<half2&>(RC0[k]);
        float2 val_f2_0 = __half22float2(val_h2_0);
        float b_norm0_f = __shfl_sync((0xffffffff), RB_NORM, 4 * (2 * k));
        float b_norm1_f = __shfl_sync((0xffffffff), RB_NORM, 4 * (2 * k + 1));
        float rsqrt_b_norm0 = rsqrtf(b_norm0_f);
        float rsqrt_b_norm1 = rsqrtf(b_norm1_f);
        float res0_f = val_f2_0.x * rsqrt_a_norm0 * rsqrt_b_norm0;
        float res1_f = val_f2_0.y * rsqrt_a_norm0 * rsqrt_b_norm1;
        half2 res_h2 = __float22half2_rn(make_float2(res0_f, res1_f));
        RC0[k] = reinterpret_cast<uint32_t&>(res_h2);

        half2 val_h2_1 = reinterpret_cast<half2&>(RC1[k]);
        float2 val_f2_1 = __half22float2(val_h2_1);
        float res2_f = val_f2_1.x * rsqrt_a_norm1 * rsqrt_b_norm0;
        float res3_f = val_f2_1.y * rsqrt_a_norm1 * rsqrt_b_norm1;
        half2 res_h2_1 = __float22half2_rn(make_float2(res2_f, res3_f));
        RC1[k] = reinterpret_cast<uint32_t&>(res_h2_1);
    }

    /*
    if (lane_id % 4 == 0): 只有线程0,4,8,12,16,20,24,28参与存储
    原因:
      每4个线程的数据被shuffle到第0个线程（线程0,4,8...）
      这些线程现在各自持有4个线程的数据
      减少内存事务数量：从32个线程写入 → 8个线程写入
    */
    // C矩阵的总大小是250000*8
    if(lane_id % 4 == 0){
        int store_smem_c_m = warp_id * MMA_M + lane_id / 4;
        int store_smem_c_n = 0;
        int store_gmem_c_m = by * MMA_M * MMA_TILE_M + store_smem_c_m;
        int store_gmem_c_n = store_smem_c_n;
        // addr0和addr1分别对应了16*8的结果矩阵中的两行，这两行的间隔为8，更具体的可以参考cosine_bk.cu中的注释和ptx文档的layout图
        int store_gmem_c_addr0 = store_gmem_c_m * MMA_N + store_gmem_c_n;
        int store_gmem_c_addr1 = (store_gmem_c_m + 8) * MMA_N + store_gmem_c_n;
        LDST128BITS(C[store_gmem_c_addr0]) = LDST128BITS(RC0[0]);
        LDST128BITS(C[store_gmem_c_addr1]) = LDST128BITS(RC1[0]);
    }
}


// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem, TN
template <const int K_STAGE = 2>
void launch_cosine_kernel_tn(half *a, half *b, half *c, int M, int N, int K)
{
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int INNER_DIM = 128;
    constexpr int MMA_TILE_M = 4;
    constexpr int A_PAD = 0;
    constexpr int B_PAD = 0;
    constexpr int BM = MMA_M * MMA_TILE_M;

    // Shared memory size calculation
    // const int smem_max_size = ((K_STAGE)*BM * (BK + A_PAD) * sizeof(half) +
    //                            (K_STAGE)*BN * (BK + B_PAD) * sizeof(half));
    const int smem_max_size = MMA_N * INNER_DIM * sizeof(half) + 
                            K_STAGE * MMA_M * MMA_TILE_M * INNER_DIM * sizeof(half);
    // cudaFuncSetAttribute的作用是告诉 CUDA 运行时（Runtime），这个 Kernel 函数需要使用超过默认限制（通常是 48KB）的动态共享内存，具体上限设为 98304 字节（即 96KB）。
    // 如果不加上这句话，而kernel使用的smem总数超过48kb，就会报错
    /*
    这里有个奇怪的点，就是这里传入参数为98304，表示一个block使用96KB的smem
    但实际ncu profile发现，每个block可使用最大smem是164KB，不是96KB
    我怀疑是cudaFuncAttributeMaxDynamicSharedMemorySize这个参数导致的, 加了这个参数之后，
    不管传入98304还是其他数值，都默认每个block可使用smem的最大值
    想了解cudaFuncSetAttribute可以直接去看cuda c programming guide
    */
    cudaFuncSetAttribute(
        cosine_kernel_tn<MMA_M, MMA_N, MMA_K, INNER_DIM, MMA_TILE_M, A_PAD, B_PAD, K_STAGE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    // const int N_SWIZZLE = (N + BLOCK_SWIZZLE_STRIDE - 1) / BLOCK_SWIZZLE_STRIDE;
    dim3 block(128); // 2 * 4 * 32
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, BM));

    cosine_kernel_tn<MMA_M, MMA_N, MMA_K, INNER_DIM, MMA_TILE_M, A_PAD, B_PAD, K_STAGE>
        <<<grid, block, smem_max_size>>>(a, b, c, M, N, K);
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