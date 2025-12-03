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
#define CP_ASYNC_WAIT_GROUP(n)                                                 \
  asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
// ca(cache all, L1 + L2): support 4, 8, 16 bytes, cg(cache global, L2): only
// support 16 bytes.
#define CP_ASYNC_CA(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
#define CP_ASYNC_CG(dst, src, bytes)                                           \
  asm volatile(                                                                \
      "cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst),       \
      "l"(src), "n"(bytes))
// smem -> gmem: requires sm_90 or higher.
#define CP_ASYNC_BULK_COMMIT_GROUP()                                           \
  asm volatile("cp.async.bulk.commit_group;\n" ::)
#define CP_ASYNC_BULK_WAIT_ALL() asm volatile("cp.async.bulk.wait_all;\n" ::)
#define CP_ASYNC_BULK_WAIT_GROUP(n)                                            \
  asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_BULK(dst, src, bytes)                                         \
  asm volatile(                                                                \
      "cp.async.bulk.global.shared::cta.bulk_group.L2::128B [%0], [%1], "      \
      "%2;\n" ::"r"(dst),                                                      \
      "l"(src), "n"(bytes))
// ldmatrix
#define LDMATRIX_X1(R, addr)                                                   \
  asm volatile("ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"        \
               : "=r"(R)                                                       \
               : "r"(addr))
#define LDMATRIX_X2(R0, R1, addr)                                              \
  asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"    \
               : "=r"(R0), "=r"(R1)                                            \
               : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr)                                      \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n"     \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))
#define LDMATRIX_X1_T(R, addr)                                                 \
  asm volatile("ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"  \
               : "=r"(R)                                                       \
               : "r"(addr))
#define LDMATRIX_X2_T(R0, R1, addr)                                            \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, [%2];\n"       \
      : "=r"(R0), "=r"(R1)                                                     \
      : "r"(addr))
#define LDMATRIX_X4_T(R0, R1, R2, R3, addr)                                    \
  asm volatile(                                                                \
      "ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, %3}, "      \
      "[%4];\n"                                                                \
      : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3)                                 \
      : "r"(addr))
// stmatrix: requires sm_90 or higher.
#define STMATRIX_X1(addr, R)                                                   \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x1.m8n8.shared.b16 [%0], {%1};\n" ::"r"(addr),    \
      "r"(R))
#define STMATRIX_X2(addr, R0, R1)                                              \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x2.m8n8.shared.b16 [%0], {%1, %2};\n" ::"r"(      \
          addr),                                                               \
      "r"(R0), "r"(R1))
#define STMATRIX_X4(addr, R0, R1, R2, R3)                                      \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x4.m8n8.shared.b16 [%0], {%1, %2, %3, %4};\n" ::  \
          "r"(addr),                                                           \
      "r"(R0), "r"(R1), "r"(R2), "r"(R3))
#define STMATRIX_X1_T(addr, R)                                                 \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x1.trans.m8n8.shared.b16 [%0], {%1};\n" ::"r"(    \
          addr),                                                               \
      "r"(R))
#define STMATRIX_X2_T(addr, R0, R1)                                            \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x2.trans.m8n8.shared.b16 [%0], {%1, %2};\n" ::    \
          "r"(addr),                                                           \
      "r"(R0), "r"(R1))
#define STMATRIX_X4_T(addr, R0, R1, R2, R3)                                    \
  asm volatile(                                                                \
      "stmatrix.sync.aligned.x4.trans.m8n8.shared.b16 [%0], {%1, %2, %3, "     \
      "%4};\n" ::"r"(addr),                                                    \
      "r"(R0), "r"(R1), "r"(R2), "r"(R3))
// mma m16n8k16
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)            \
  asm volatile(                                                                \
      "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, "  \
      "%4, %5}, {%6, %7}, {%8, %9};\n"                                         \
      : "=r"(RD0), "=r"(RD1)                                                   \
      : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0),  \
        "r"(RC1))

HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }



/*
这个是reg double buffer，对应了reed大佬的multistage中的tile内流水线https://zhuanlan.zhihu.com/p/665082713
这个代码属于是multistage的tile间和tile内流水线都实现了，这里的reg double buffer相当于两级的tile内流水
看代码可以发现，在大K tile的for循环中，tile内流水的顺序是这样的（假设32的大K分为了K1和K2的两个小k tile）：
  loadmatrix k2 -> mma k1 -> mma k2 -> loadmatrix k1
这里loadmatrix和mma都是同步指令，所以直观看上去上面的这种顺序是没办法形成流水线的
我理解这里可能因为loadmatrix k2和mma k1在数据上没有依赖，所以编译器在编译时可以将loadmatrix k2和mma k1同时发射，这样有可能形成流水线（未完持续）

根据注释说的，这里k变成double了(从16变成32)，为了减少bank冲突，不能一次访问32，需要一次访问16
所以把原本的[stages][BM][BK * WARP_TILE_K]的布局的smem变成了[stages * WARP_TILE_K][BM][BK]的布局
目的就是把32拆成两个16去访问。直观上看是stage数翻倍了，实际上smem的大小仍然不变
然后这里为什么改变smem布局、把k从32拆成两个16能减少bank冲突，这块虽然问了cursor但没怎么仔细看，以后如果有需要的话再研究这块
*/
// In order to reduce bank conflicts, we will save the K(16x2=32)
// dimension by half according to the stage dimension. For example,
// stages=3, warp_tile_k=2, it will be saved as [3*2][BM][16].
// 128x128, mma2x4, warp4x4(64,32,32), stages, block swizzle, dsmem,
// k32 with reg double buffers
// 为了减少存储体冲突，我们将根据stage dimension将K(16x2=32)维度减半保存。
// 例如，stages=3, warp_tile_k=2时，它将被保存为[3*2][BM][16]。
// 128x128, mma2x4, warp4x4(64,32,32), 多阶段流水线, 块交换, 动态共享内存,
// k32配合寄存器双缓冲
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int WARP_TILE_K = 2, const int A_PAD = 0, const int B_PAD = 0,
          const int K_STAGE = 2, const bool BLOCK_SWIZZLE = true,
          const bool WARP_SWIZZLE = true>
__global__ void __launch_bounds__(256)
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel(
        const half *__restrict__ A, const half *__restrict__ B,
        half *__restrict__ C, int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K * WARP_TILE_K); // 这里因为tile内分了两个ktile，所以MMA_K乘上了WARP_TILE_K
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K;                            // 16x2=32 ，看了下代码，这里MMA_K还是16

  extern __shared__ half smem[];

    // 参考hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel这个kernel
    // //   A矩阵和B矩阵的共享内存布局是这样的：
    // // ┌─────────────────────────────────────────────────────────────┐
    // // │              A矩阵区域                     │    B矩阵区域    │
    // // │  Stage0   Stage1   ...   Stage(K-1)       │                 │
    // // ├─────────┬─────────┬─────┬─────────────────┼─────────────────┤
    // // │ BM×(BK  │ BM×(BK  │ ... │ BM×(BK+A_PAD)   │ K_STAGE×BK×     │
    // // │ +A_PAD) │ +A_PAD) │     │                 │ (BN+B_PAD)      │
    // // └─────────┴─────────┴─────┴─────────────────┴─────────────────┘
    // // ^                                           ^
    // // s_a                                         s_b
    // half *s_a = smem;
    // half *s_b = smem + K_STAGE * BM * (BK + A_PAD); // 这里加的K_STAGE * BM * (BK + A_PAD)的offset表示为A矩阵预留的smem的空间

  half *s_a = smem;
  half *s_b = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;
  half *s_a_norm = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K + BM;
  half *s_b_norm = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K + BM + BN;

  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // 128x16
  constexpr int s_b_stage_offset = BK * (BN + B_PAD); // 16x128
  constexpr int s_a_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
  constexpr int s_b_mma_k_store_offset = K_STAGE * BK * (BN + B_PAD);
// 这里threadIdx.y恒为0
  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  // 下面这两个变量的作用
  // 一个block有8个warp，按照2*4的二维排列，这里warp_m和warp_n相当于warp在block中的二维坐标
  const int warp_m = warp_id % 2;      // 0,1
  const int warp_n = warp_id / 2;      // 0,1,2,3
  // 先计算shared memory中的索引
  // tid和需要加载的smem s_a[BM][BK] 之间的索引关系 BM=128 BK=16 按行读取
  // A行主序
  // 对于s_a每行16个数据，每个线程读取8个，需要2个线程；总共128行，需要128x2刚好256线程
  int load_smem_a_m = tid / 2;                 // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // col 0,8
// tid和需要加载的smem s_b[BK][BN] 之间的索引关系 BK=16 BN=128 按行读取
  // B行主序
  // 对于s_b每行128个数据，每个线程读8个数据，需要16个线程；总共16行，需要16x16=256个线程
  int load_smem_b_k = tid / 16;                // row 0~15
  int load_smem_b_n = (tid % 16) * 8;          // col 0,8,16,...
    // 再计算全局内存中的索引
  // 要加载到s_a中的元素对应到A全局内存中的行数
  // 每个block负责出C中大小为BM*BN的块
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2];
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      RC[i][j][0] = 0;
      RC[i][j][1] = 0;
    }
  }

  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) { // 0, 1
    // k * WMMA_K, WMMA_K=16 -> (k << 4)
    int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    /*
    这里说一下为什么预加载K_STAGE-1个gmem的数据
    参考reed大佬的讲解multistage的博客https://zhuanlan.zhihu.com/p/665082713
    即在进入tile间循环计算之前，需要将stage-1个异步的gmem到smem的数据加载任务发射出去
    */

    /*
    这里说一下这里为什么A分两次加载，直观上是因为32的K被分为了两份16的K，所以分两次加载
    实际上，加载到smem中的数据分布如下

    Smem Base Address (s_a)
    |
    v
    +-------------------------------------------------------+
    | Stage 0, K0 (0~15)                                    |  <-- 区域 A 开始
    | Stage 1, K0 (0~15)                                    |
    | Stage 2, K0 (0~15)                                    |
    +-------------------------------------------------------+  <-- s_a_mma_k_store_offset 指向这里
    | Stage 0, K1 (16~31)                                   |  <-- 区域 B 开始
    | Stage 1, K1 (16~31)                                   |
    | Stage 2, K1 (16~31)                                   |
    +-------------------------------------------------------+
    可以看到不同stage的前一个16k和后一个16k并不是连续的，而是分为了两块区域，刚好对应了最开始申请s_a内存时布局[stages * WARP_TILE_K][BM][BK]
    之所以使用这样的smem布局，看上面作者的注释说明，是因为这样能减少bank冲突，但这块我还没弄清楚为什么能减少bank冲突
    */
    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
             sizeof(half));
    // 16表示每个线程搬多少字节的个数
    // 这里A tile的大小是BM×BK ，在这里为128*16
    // 128*16/256=8,即一个block中的线程,每个搬8个half才能把128*16个half从gmem搬到smem,8个half就是16字节
    // 这里忽略A_PAD参数，因为A_PAD只是用来改变smem中数据的填充顺序的，与从gmem中搬多少个数据无关，下同
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16); // MMA_K 0


    uint32_t load_smem_a_mma_k_ptr =
        (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
             sizeof(half));
    CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16],
                16); // MMA_K 1

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr +
         (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
             sizeof(half));
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    int load_gmem_b_k_mma_k = k * BK * WARP_TILE_K + MMA_K + load_smem_b_k;
    int load_gmem_b_addr_mma_k = load_gmem_b_k_mma_k * N + load_gmem_b_n;
    uint32_t load_smem_b_mma_k_ptr =
        (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) +
         (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
             sizeof(half));
    CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr_mma_k], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2); // s2->0, s3->1, s4->2
  __syncthreads();

  // 这里因为使用了tile内流水（32的大k tile分成了两个16的小k tile），所以需要的RA和RB寄存器翻倍了
  uint32_t RA[2][WARP_TILE_M][4];
  uint32_t RB[2][WARP_TILE_N][2];

  // 这两个参数用于控制tile内流水，因为使用的双缓冲，即大k tile分成了两个小k tile
  // 所以这里用了两个变量控制寄存器
  int reg_store_idx = 0;
  int reg_load_idx = 1;

  // 这里在进入tile内的k tile for循环之前，需要提前加载了tile内的小k tile的数据
  // 这里是将每个k tile分成了两个小k tile，所以这里是加载的第一个小k tile的数据
  // 这个步骤就是reed大佬的gemm流水线博客里面的那个图，第一个黑色虚线和黑色实线部分
  {
// ldmatrix for s_a, ldmatrix.trans for s_b.
// smem -> reg buffers 0, first MMA_K, 0~15
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      // warp_smem_a_m是warp level的index，lane_smem_a_m是warp内thread level的index
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
      // 0 * s_a_stage_offset表示stage偏移
      // 这里为什么是 0：这段代码通常出现在 Kernel 的序幕（Prologue）阶段。它正在加载第一批数据，所以硬编码为读取第 0 个 Stage 的 Buffer。在主循环中，这个 0 会变成变量 smem_sel
      uint32_t lane_smem_a_ptr =
          (smem_a_base_ptr + (0 * s_a_stage_offset +
                              lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) *
                                 sizeof(half));
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                  lane_smem_a_ptr);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
      // 这里说一下，为什么这里lane_smem_b_n = warp_smem_b_n，而不是lane_smem_b_n = warp_smem_b_n + (lane_id / 16) * 8
      // 上面ldmatrix载入a时，lane_smem_a_k = (lane_id / 16) * 8，是因为那里使用的x4的指令，有16列
      // 而这里载入b时，使用的x2的方法，载入的是16*8的矩阵，所以lane_smem_b_n这个列坐标就不需要(lane_id / 16) * 8这个列偏移
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      // 这里说一下为什么下面这个lane_smem_b_ptr需要乘以sizeof(half)，主要是因为类型是uint32_t的
      // 使用uint32_t类型表示代表 Shared Memory 中的字节地址（Byte Address），所以在算出元素偏移量之后，还要乘上half的字节个数
      // 只有当指针是T*(指针类型)类型时，才能只使用ptr + offset而不需要乘以sizeof(T)
      uint32_t lane_smem_b_ptr =
          (smem_b_base_ptr + (0 * s_b_stage_offset +
                              lane_smem_b_k * (BN + B_PAD) + lane_smem_b_n) *
                                 sizeof(half));
      // may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                    lane_smem_b_ptr);
    }
  }

  // 这里进入tile内的k tile for循环
  // 需要注意的是，因为k tile的循环次数按照k tile大小为32计算的
#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    reg_store_idx ^= 1;               // 0->1
    reg_load_idx ^= 1;                // 1->0
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // stage gmem -> smem
    int load_gmem_a_k = k * BK * WARP_TILE_K + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK * WARP_TILE_K + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;
    // tile间流水，发射下一个stage的cp.async
    // 这里因为把32的k tile拆成了两个16的小k tile，所以对于A和B，都需要发射两次cp.async
    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset +
                            load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
                               sizeof(half));
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr =
        (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) +
         (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) +
          load_smem_a_k) *
             sizeof(half));
    CP_ASYNC_CG(load_smem_a_mma_k_ptr, &A[load_gmem_a_addr + 16],
                16); // MMA_K 1

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset +
                            load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
                               sizeof(half));
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    int load_gmem_b_k_mma_k = k * BK * WARP_TILE_K + MMA_K + load_smem_b_k;
    int load_gmem_b_addr_mma_k = load_gmem_b_k_mma_k * N + load_gmem_b_n;
    uint32_t load_smem_b_mma_k_ptr =
        (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) +
         (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) +
          load_smem_b_n) *
             sizeof(half));
    CP_ASYNC_CG(load_smem_b_mma_k_ptr, &B[load_gmem_b_addr_mma_k], 16);
    CP_ASYNC_COMMIT_GROUP();

// ldmatrix for s_a, ldmatrix.trans for s_b.
// smem -> reg buffers 1, second MMA_K, 16~31
// 上面发射完了cp.async之后，这里需要将第二个小ktile的数据加载到寄存器中
// 可以看到这里的reg_store_idx，在进入for循环之后，从0变为了1
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
      // smem_sel * s_a_stage_offset表示stage偏移
      // 需要注意的是这里的偏移量加上了s_a_mma_k_store_offset，这是因为这里加载的是第二个小ktile的数据
      uint32_t lane_smem_a_ptr =
          (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) +
           (smem_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) +
            lane_smem_a_k) *
               sizeof(half));
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                  lane_smem_a_ptr);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr =
          (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) +
           (smem_sel * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) +
            lane_smem_b_n) *
               sizeof(half));
      // may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                    lane_smem_b_ptr);
    }

// MMA compute, first MMA_K
// 加载完第二个小k tile的数据到寄存器之后，开始计算第一个小k tile的mma
// 这里的reg_load_idx，在进入for循环之后，从1变为了0
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        /*
        说一下这里的warp swizzle，这里只是对RC和RB里面数据的访问顺序进行了swizzle，swizzle与否都不影响计算结果
        cursor说这里的warp swizzle的目的是为了减少reg的bank冲突
        当没有swizzle时，i j的变化顺序如下：
        i=0: j_s = 0,1,2,3  (左到右)
        i=1: j_s = 0,1,2,3  (左到右)  
        i=2: j_s = 0,1,2,3  (左到右)
        i=3: j_s = 0,1,2,3  (左到右)
        当有了swizzle时，i j的变化顺序如下
        i=0: j_s = 0,1,2,3     (左到右) ← i%2=0，偶数行
        i=1: j_s = 3,2,1,0     (右到左) ← i%2=1，奇数行反转
        i=2: j_s = 0,1,2,3     (左到右) ← i%2=0，偶数行  
        i=3: j_s = 3,2,1,0     (右到左) ← i%2=1，奇数行反转
        我理解是，由于这里的循环展开优化，外加不同线程本来运行速度不一样
        有的线程此时可能才运行到i=0，有的可能已经运行到了i=1
        然后这两个线程在j上访问的顺序就会不一样（i=0的线程j是正序访问，i=1的线程j是逆序访问）
        所以就能在一定程度上减少reg的bank冲突
        更多具体的细节我没有继续深究了，后面有需要的话再来继续研究这块
        */
        int j_s = ((i % 2) && WARP_SWIZZLE) ? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0],
                  RA[reg_load_idx][i][1], RA[reg_load_idx][i][2],
                  RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                  RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
      }
    }

    reg_store_idx ^= 1; // 1 -> 0
    reg_load_idx ^= 1;  // 0 -> 1
// MMA compute, second MMA_K
// 这里计算第二个mma，对应第二个小k tile
// 这的的reg load idx在经过上面的反转之后，又从0变为了1
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // Warp swizzle: Right -> Left -> Right -> Left
        int j_s = ((i % 2) && WARP_SWIZZLE) ? (WARP_TILE_N - j - 1) : j;
        HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0],
                  RA[reg_load_idx][i][1], RA[reg_load_idx][i][2],
                  RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                  RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
      }
    }
    // 前面已经把所有的tile内的mma计算完了，所以这里需要等待下一次大K迭代所需要的结果已经加载完成，所以这里需要再一次wait
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    /*
    上面执行wait之后，下一次for循环迭代所需要的数据已经从gmem拷贝到了smem，这里需要在进入下一个
    大K的for循环之前，提前将一个大K的迭代的第一个小k tile的数据拷贝到reg中
    */
    // load next k iters to reg buffers.
    // smem -> reg buffers 0, first MMA_K, 0~15
    // int smem_sel_reg = (k + 2) % K_STAGE; // vs smem_sel k=2->(0)1, k=3->(1)2
    // 正在为下一轮大K维度的迭代加载数据到寄存器缓冲区, 为了提前准备下一次小k的MMA计算所需的数据
    // 从smem加载数据到寄存器缓冲区0
    // smem_sel_reg = (k + 2) % K_STAGE与这里的smem_sel_reg = (smem_sel + 1) % K_STAGE是等价的
    // 实现了这样的效果，以stage=3为例，k=2时，smem_sel=0, smem_sel_reg=1
    // 当k=3时，smem_sel=1, smem_sel_reg=2
    // smem_sel代表了当前for循环迭代计算的buffer的index，smem_sel_reg代表了下一轮for循环计算的buffer的index
    // 这里说一下这个smem_sel_reg和上面的smem_sel_next的区别
    // smem_sel_reg算的是smem_sel的下一个stage，smem_sel_reg = smem_sel + 1，而且smem_sel_reg只用在加载下一个小k tile的ldmatrix的逻辑里面
    // 但smem_sel_next算的不是smem_sel的下一个stage，算的是异步的gmem->smem的stage的index，是用在main loop中的cp async逻辑里的
    int smem_sel_reg =
        (smem_sel + 1) % K_STAGE; // vs smem_sel k=2->(0)1, k=3->(1)2
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
      uint32_t lane_smem_a_ptr =
          (smem_a_base_ptr + (smem_sel_reg * s_a_stage_offset +
                              lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) *
                                 sizeof(half));
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                  lane_smem_a_ptr);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr =
          (smem_b_base_ptr + (smem_sel_reg * s_b_stage_offset +
                              lane_smem_b_k * (BN + B_PAD) + lane_smem_b_n) *
                                 sizeof(half));
      // may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                    lane_smem_b_ptr);
    }
  }

  // make sure all memory issues ready.
  if constexpr ((K_STAGE - 2) > 0) {
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();
  }

  // processing last (K_STAGE-1) k iters.
  {
#pragma unroll
    for (int k = 0; k < (K_STAGE - 1); k++) {
      // 这个stage_sel负责计算收尾部分的buffer的index
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
      // 这里stage_sel负责计算buffer2和buffer0的index
      // 当k=0时，stage_sel = (10 - (3 - 1) + 0) % 3 = 2
      // 当k=1时，stage_sel = (10 - (3 - 1) + 1) % 3 = 0
      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
// ldmatrix for s_a, ldmatrix.trans for s_b.
// smem -> reg buffers 1, second MMA_K
      // 这里仍然是加载第二个小k tile的数据(第一个小ktile的数据加载已经在上面主循环的最后那块执行了)
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
        uint32_t lane_smem_a_ptr =
            (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) +
             (stage_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) +
              lane_smem_a_k) *
                 sizeof(half));
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                    lane_smem_a_ptr);
      }

#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16;  // 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr =
            (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) +
             (stage_sel * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) +
              lane_smem_b_n) *
                 sizeof(half));
        LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                      lane_smem_b_ptr);
      }

// MMA compute, first MMA_K
      // 计算第一个小k tile的mma
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          int j_s = ((i % 2) && WARP_SWIZZLE) ? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0],
                    RA[reg_load_idx][i][1], RA[reg_load_idx][i][2],
                    RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                    RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
        }
      }

      reg_store_idx ^= 1; // 1 -> 0
      reg_load_idx ^= 1;  // 0 -> 1

// MMA compute, second MMA_K
      // 计算第二个小k tile的mma
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          // Warp swizzle: Right -> Left -> Right -> Left
          int j_s = ((i % 2) && WARP_SWIZZLE) ? (WARP_TILE_N - j - 1) : j;
          HMMA16816(RC[i][j_s][0], RC[i][j_s][1], RA[reg_load_idx][i][0],
                    RA[reg_load_idx][i][1], RA[reg_load_idx][i][2],
                    RA[reg_load_idx][i][3], RB[reg_load_idx][j_s][0],
                    RB[reg_load_idx][j_s][1], RC[i][j_s][0], RC[i][j_s][1]);
        }
      }

      // load next k iters to reg buffers.
      // smem -> reg buffers 0, first MMA_K, 0~15
      // int stage_sel_reg = ((NUM_K_TILES - K_STAGE + k) % K_STAGE);
      // 加载下一轮大k tile的第一个小k tile的数据到reg
      int stage_sel_reg = (stage_sel + 1) % K_STAGE;
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
        uint32_t lane_smem_a_ptr =
            (smem_a_base_ptr + (stage_sel_reg * s_a_stage_offset +
                                lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) *
                                   sizeof(half));
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                    lane_smem_a_ptr);
      }

#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16;  // 0~15, 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr =
            (smem_b_base_ptr + (stage_sel_reg * s_b_stage_offset +
                                lane_smem_b_k * (BN + B_PAD) + lane_smem_b_n) *
                                   sizeof(half));
        LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                      lane_smem_b_ptr);
      }
    }
  }

  // collective store with reg reuse & warp shuffle
  for (int i = 0; i < WARP_TILE_M; ++i) {
// reuse RA[2][4][4] reg here, this may boost 0.3~0.5 TFLOPS up.
// may not put 'if' in N loop, it will crash the 'pragma unroll' hint ?
/*
这里原本RC[i][j]: 存储MMA计算的结果
对于RA[0/1][j]: 原本用于输入数据A的寄存器，现在复用来存储输出数据
这样节省寄存器使用，提升0.3~0.5 TFLOPS
*/
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      // 这里为了使用LDST128BITS存储结果，使用了warp shuffle，将数据汇聚到特定线程（一个warp32个线程的数据放到了其中8个线程中存储）
      // 这样做的好处是，为shuffle之前，32个线程 × 32位存储 = 32个内存事务
      // shuffle之后8个线程 × 128位存储 = 8个内存事务。可以看到shuffle之后内存事务减少
      /*
      这里的warp shuffle的目的: 将每个线程的数据通过warp shuffle收集到特定线程上，为合并写入做准备
      数据流向:
        线程0收集：自己的数据 + 线程1,2,3的数据
        线程4收集：自己的数据 + 线程5,6,7的数据
      */
      /*
      // 这里i表示当前正在处理的warp负责区域中的哪一个 16 行的水平条带
      // A的threadblock tile的大小是128*16，由于MMA_TILE_M = 2，所以实际上A的warp tile大小是64*16
      // 同时由于WARP_TILE_M为4，所以A的warp tile被横向划分为了4个16*16的矩阵，这里的i就表示当前16*16矩阵是第几个
      // 这里threadblock tile大小是128*128，被分成了2*4份（二维），每份64*32，相当于一共8个warp tile
      一个warp的RC[i][j]存储了一个16*16的结果，这16*16结果在一个warp中的排布见ptx文件
      一个warp tile的大小是64*32，这里i固定然后对j做for循环（WARP_TILE_N=4）
      所以当j的循环结束时，相当于将一个16*32的结果放到了一个warp的RA中(B warp tile的大小是16*32,被分成了4个16*8，对应了WARP_TILE_N=4)
      具体是这样的，一个warp有32个线程，其中只有0，4，8，12，16，20，24，28这8个线程存储了结果，以0号线程为例
      0号线程的RA的大小是2*4*4个uint32，可以存32*2个half
      所以一个warp中总共8个线程共存放了16*32的结果
      */
      RA[0][j][0] = RC[i][j][0];
      RA[1][j][0] = RC[i][j][1];
      RA[0][j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
      RA[0][j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
      RA[0][j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
      RA[1][j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
      RA[1][j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
      RA[1][j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
    }
    /*
    if (lane_id % 4 == 0): 只有线程0,4,8,12,16,20,24,28参与存储
    原因:
      每4个线程的数据被shuffle到第0个线程（线程0,4,8...）
      这些线程现在各自持有4个线程的数据
      减少内存事务数量：从32个线程写入 → 8个线程写入
    */
    if (lane_id % 4 == 0) {
      // 这里i表示当前正在处理的warp负责区域中的哪一个 16 行的水平条带
      // A的threadblock tile的大小是128*16，由于MMA_TILE_M = 2，所以实际上A的warp tile大小是64*16
      // 同时由于WARP_TILE_M为4，所以A的warp tile被横向划分为了4个16*16的矩阵，这里的i就表示当前16*16矩阵是第几个
      // 这里threadblock tile大小是128*128，被分成了2*4份（二维），每份64*32，相当于一共8个warp tile
      // 注意这里的warp_m的定义，表示当前线程在的warp tile在整个threadblock tile中的行偏移
      // warp_m * (MMA_M * WARP_TILE_M)就表示当前线程所在位置在A threadblock tile中的行偏移 warp_m * ( 16 * 4 )
      // 而i * MMA_M则表示在A warp tile中的行偏移，i * 16
      int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n;
        int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
        int store_gmem_c_addr_1 =
            (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
        LDST128BITS(C[store_gmem_c_addr_0]) = LDST128BITS(RA[0][j][0]);
        LDST128BITS(C[store_gmem_c_addr_1]) = LDST128BITS(RA[1][j][0]);
      }
    }
  }
}


#include "utils.h"

// 128x128, mma2x4, warp4x4x2(64,32,32), stages, block swizzle, dsmem, reg
// double buffers
#define LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(stages,       \
                                                                 stride)       \
  {                                                                            \
    const int smem_max_size =                                                  \
        ((stages) * BM * (BK + A_PAD) * WARP_TILE_K * sizeof(half) +           \
         (stages) * BK * (BN + B_PAD) * WARP_TILE_K * sizeof(half));           \
    cudaFuncSetAttribute(                                                      \
        hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel<               \
            MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,          \
            WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), true>,           \
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);                   \
    const int N_SWIZZLE = (N + (stride) - 1) / (stride);                       \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),  \
              N_SWIZZLE);                                                      \
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel<                   \
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M, WARP_TILE_N, \
        WARP_TILE_K, A_PAD, B_PAD, (stages), true>                             \
        <<<grid, block, smem_max_size>>>(a, b, c, M, N, K);                    \
  }

  #define LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(stages)    \
  {                                                                            \
    const int smem_max_size =                                                  \
        ((stages) * BM * (BK + A_PAD) * WARP_TILE_K * sizeof(half) +           \
         (stages) * BK * (BN + B_PAD) * WARP_TILE_K * sizeof(half));           \
    cudaFuncSetAttribute(                                                      \
        hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel<               \
            MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,          \
            WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), false>,          \
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);                   \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                               \
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_kernel<                   \
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M, WARP_TILE_N, \
        WARP_TILE_K, A_PAD, B_PAD, (stages), false>                            \
        <<<grid, block, smem_max_size>>>(                                      \
            reinterpret_cast<half *>(a.data_ptr()),                            \
            reinterpret_cast<half *>(b.data_ptr()),                            \
            reinterpret_cast<half *>(c.data_ptr()), M, N, K);                  \
  }


// 128x128, mma2x4, warp4x4x2(64,32,32), stages, block swizzle, dsmem, reg
// double buffers
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem(torch::Tensor a,
                                                      torch::Tensor b,
                                                      torch::Tensor c,
                                                      int stages, bool swizzle,
                                                      int swizzle_stride) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(b, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(c, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(b, K, N)
  CHECK_TORCH_TENSOR_SHAPE(c, M, N)
  constexpr int MMA_M = 16;
  constexpr int MMA_N = 8;
  constexpr int MMA_K = 16;
  constexpr int MMA_TILE_M = 2;
  constexpr int MMA_TILE_N = 4;
  constexpr int WARP_TILE_M = 4;
  constexpr int WARP_TILE_N = 4;
  constexpr int WARP_TILE_K = 2;
  // bank conflicts free via pad = 8, reject fantasy, trust the profile.
  // ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld
  // ./hgemm_mma_stage.89.debug.bin ncu --metrics
  // sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm
  // ./hgemm_mma_stage.89.debug.bin
  constexpr int A_PAD = 8; // 0,8,16
  constexpr int B_PAD = 8; // 0,8,16
  constexpr int NUM_THREADS =
      (MMA_TILE_M * MMA_TILE_N * WARP_SIZE); // 2 * 4 * 32 = 256
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M;
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N;
  constexpr int BK = MMA_K;
  // s2: 2*128*(32)*2=16KB, 2*32*(128+16)*2=18KB, ~35KB
  // s3: 3*128*(32)*2=24KB, 3*32*(128+16)*2=27KB, ~51KB
  // s4: 4*128*(32)*2=32KB, 4*32*(128+16)*2=36KB, ~68KB
  // s5: 5*128*(32)*2=40KB, 5*32*(128+16)*2=45KB, ~85KB
  if (swizzle) {
    // assert(swizzle_stride % 256 == 0);
    switch (stages) {
    case 2: // ~35KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(2,
                                                               swizzle_stride);
      break;
    case 3: // ~51KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(3,
                                                               swizzle_stride);
      break;
    case 4: // ~68KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(4,
                                                               swizzle_stride);
      break;
    case 5: // ~85KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(5,
                                                               swizzle_stride);
      break;
    default:
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(2,
                                                               swizzle_stride);
      break;
    }
  } else {
    switch (stages) {
    case 2:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(2);
      break;
    case 3:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(3);
      break;
    case 4:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(4);
      break;
    case 5:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(5);
      break;
    default:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(2);
      break;
    }
  }
}
