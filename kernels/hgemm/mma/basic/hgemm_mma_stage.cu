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
下面这个mma的多级流水和block swizzle就不讲了，和hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel是一模一样的原理
*/
// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = true>
__global__ void __launch_bounds__(256)
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel(half *A, half *B, half *C,
                                                    int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K;                            // 16

  __shared__ half s_a[K_STAGE][BM][BK + A_PAD]; // 128*16*2=4KB
  __shared__ half
      s_b[K_STAGE][BK][BN + B_PAD]; // 16*128*2=4KB, 16*(128+16)*2=4.5KB
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2;      // 0,1
  const int warp_n = warp_id / 2;      // 0,1,2,3

  int load_smem_a_m = tid / 2;                 // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // col 0,8
  int load_smem_b_k = tid / 16;                // row 0~15
  int load_smem_b_n = (tid % 16) * 8;          // col 0,8,...,120
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
    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a, 这个的意思是说load_gmem_a_k是A的gmem的列坐标
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
             sizeof(half));
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr +
         (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
             sizeof(half));
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2); // s2->0, s3->1, s4->2
  __syncthreads();

#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    // gmem -> smem
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset +
                            load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
                               sizeof(half));
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset +
                            load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
                               sizeof(half));
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

    // ldmatrix for s_a, ldmatrix.trans for s_b.
    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];

// smem -> reg
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
      uint32_t lane_smem_a_ptr = __cvta_generic_to_shared(
          &s_a[smem_sel][lane_smem_a_m][lane_smem_a_k]);
      LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr = __cvta_generic_to_shared(
          &s_b[smem_sel][lane_smem_b_k][lane_smem_b_n]);
      LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
    }

// MMA compute
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1], RA[i][2],
                  RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
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
      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      // ldmatrix for s_a, ldmatrix.trans for s_b.
      uint32_t RA[WARP_TILE_M][4];
      uint32_t RB[WARP_TILE_N][2];

// smem -> reg
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
        uint32_t lane_smem_a_ptr = __cvta_generic_to_shared(
            &s_a[stage_sel][lane_smem_a_m][lane_smem_a_k]);
        LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
      }

#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16;  // 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr = __cvta_generic_to_shared(
            &s_b[stage_sel][lane_smem_b_k][lane_smem_b_n]);
        LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
      }

// MMA compute
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1], RA[i][2],
                    RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1]);
        }
      }
    }
  }

// reg -> gmem, MMA_MxMMA_N=16x8
#pragma unroll
  for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      // mapping lane smem index -> global index.
      // [16][8],
      // https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
      // #matrix-fragments-for-mma-m16n8k16-with-floating-point-type
      // [0~7][0~3 u32 -> 0~7 f16], [8~15][0~3 u32 -> 0~7 f16]
      int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
      int store_lane_gmem_c_n =
          bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
      int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
      int store_gmem_c_addr_1 =
          (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
      LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]);
      LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]);
    }
  }
}

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = true, const bool COLLECTIVE_STORE = false>
__global__ void __launch_bounds__(256)
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel(half *A, half *B,
                                                          half *C, int M, int N,
                                                          int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  // COLLECTIVE_STORE true/false control use stmatrix or not.
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K;                            // 16

  extern __shared__ half smem[];
  half *s_a = smem;
  half *s_b = smem + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD);
  constexpr int s_b_stage_offset = BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2;      // 0,1
  const int warp_n = warp_id / 2;      // 0,1,2,3

  int load_smem_a_m = tid / 2;                 // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // col 0,8
  int load_smem_b_k = tid / 16;                // row 0~15
  int load_smem_b_n = (tid % 16) * 8;          // col 0,8,...,120
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
    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
             sizeof(half));
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr +
         (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
             sizeof(half));
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2); // s2->0, s3->1, s4->2
  __syncthreads();

#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    // gmem -> smem
    // s2/4 can use bitwise ops but s3 can not, so, we use mod
    // ops for all stages kernel. s2: (k + 1)&1, s4: (k + 1)&3
    // s3: (k + 1) % 3
    int smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    int smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global row of b
    int load_gmem_b_addr = load_gmem_b_k * N + load_gmem_b_n;

    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset +
                            load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
                               sizeof(half));
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset +
                            load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
                               sizeof(half));
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);
    CP_ASYNC_COMMIT_GROUP();

    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];
// ldmatrix for s_a, ldmatrix.trans for s_b.
// smem -> reg
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
      uint32_t lane_smem_a_ptr =
          (smem_a_base_ptr + (smem_sel * s_a_stage_offset +
                              lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) *
                                 sizeof(half));
      LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
      int lane_smem_b_k = lane_id % 16;  // 0~15
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr =
          (smem_b_base_ptr + (smem_sel * s_b_stage_offset +
                              lane_smem_b_k * (BN + B_PAD) + lane_smem_b_n) *
                                 sizeof(half));
      LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
    }

// MMA compute
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1], RA[i][2],
                  RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1]);
      }
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
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
      uint32_t RA[WARP_TILE_M][4];
      uint32_t RB[WARP_TILE_N][2];

      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
// ldmatrix for s_a, ldmatrix.trans for s_b.
// smem -> reg
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
        int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
        uint32_t lane_smem_a_ptr =
            (smem_a_base_ptr + (stage_sel * s_a_stage_offset +
                                lane_smem_a_m * (BK + A_PAD) + lane_smem_a_k) *
                                   sizeof(half));
        LDMATRIX_X4(RA[i][0], RA[i][1], RA[i][2], RA[i][3], lane_smem_a_ptr);
      }

#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int warp_smem_b_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int lane_smem_b_k = lane_id % 16;  // 0~15
        int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
        uint32_t lane_smem_b_ptr =
            (smem_b_base_ptr + (stage_sel * s_b_stage_offset +
                                lane_smem_b_k * (BN + B_PAD) + lane_smem_b_n) *
                                   sizeof(half));
        LDMATRIX_X2_T(RB[j][0], RB[j][1], lane_smem_b_ptr);
      }

// MMA compute
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          HMMA16816(RC[i][j][0], RC[i][j][1], RA[i][0], RA[i][1], RA[i][2],
                    RA[i][3], RB[j][0], RB[j][1], RC[i][j][0], RC[i][j][1]);
        }
      }
    }
  }

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 90)
  if (COLLECTIVE_STORE) {
    // The following code has not been tested because I do not have a GPU with
    // sm>=90 reg -> smem(stmatrix) -> gmem(cp.async.bulk), MMA_MxMMA_N=16x8
    // NOTE: need [MMA_M][MMA_N] per warp to avoid overlap between warps.
    __shared__ half
        s_c[MMA_TILE_M][MMA_TILE_N][MMA_M][MMA_N]; // (2*4)*16*8*2=2KB
    uint32_t smem_c_base_ptr = __cvta_generic_to_shared(&s_c[warp_m][warp_n]);
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // store (i,j) warp tile -> smem c, 16x8
        uint32_t lane_smem_c_ptr =
            (smem_c_base_ptr +
             (lane_id % 16) * MMA_N * sizeof(half)); // (0~15)*8
        STMATRIX_X2(lane_smem_c_ptr, RC[i][j][0], RC[i][j][1]);
        // smem -> gmem, may use cp.async.bulk.global.share::cta?
        int store_warp_gmem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int store_warp_gmem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int store_lane_gmem_c_m = by * BM + store_warp_gmem_c_m;
        int store_lane_gmem_c_n = bx * BN + store_warp_gmem_c_n;
        // send 16 memory issues with 128 bits within lower half lanes.
        // TODO: use cp.async.bulk and wait outside the inner loop.
        if (lane_id < 16) {
          int store_gmem_c_addr =
              (store_lane_gmem_c_m + lane_id) * N + store_lane_gmem_c_n;
          LDST128BITS(C[store_gmem_c_addr]) =
              LDST128BITS(s_c[warp_m][warp_n][lane_id][0]);
        }
        __syncwarp();
      }
    }
  } else {
// reg -> gmem, MMA_MxMMA_N=16x8
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
        int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
        int store_lane_gmem_c_n =
            bx * BN + store_warp_smem_c_n + (lane_id % 4) * 2;
        int store_gmem_c_addr_0 = store_lane_gmem_c_m * N + store_lane_gmem_c_n;
        int store_gmem_c_addr_1 =
            (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
        LDST32BITS(C[store_gmem_c_addr_0]) = LDST32BITS(RC[i][j][0]);
        LDST32BITS(C[store_gmem_c_addr_1]) = LDST32BITS(RC[i][j][1]);
      }
    }
  }
#else
  // #warning "stmatrix need sm>=90, force use __shfl_sync for collective
  // store!"
  {
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      // may reuse RA[4][4] as RC0 ? only new RC1[4][4].
      uint32_t RC0[WARP_TILE_N][4];
      uint32_t RC1[WARP_TILE_N][4];
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
        // thus, we only need 8 memory issues with 128 bits after shfl_sync.
        RC0[j][0] = RC[i][j][0];
        RC1[j][0] = RC[i][j][1];
        RC0[j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
        RC0[j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
        RC0[j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
        RC1[j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
        RC1[j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
        RC1[j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
      }

      if (lane_id % 4 == 0) {
        int store_warp_smem_c_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
        int store_lane_gmem_c_m = by * BM + store_warp_smem_c_m + lane_id / 4;
#pragma unroll
        for (int j = 0; j < WARP_TILE_N; ++j) {
          int store_warp_smem_c_n = warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N;
          int store_lane_gmem_c_n = bx * BN + store_warp_smem_c_n;
          int store_gmem_c_addr_0 =
              store_lane_gmem_c_m * N + store_lane_gmem_c_n;
          int store_gmem_c_addr_1 =
              (store_lane_gmem_c_m + 8) * N + store_lane_gmem_c_n;
          LDST128BITS(C[store_gmem_c_addr_0]) = LDST128BITS(RC0[j][0]);
          LDST128BITS(C[store_gmem_c_addr_1]) = LDST128BITS(RC1[j][0]);
        }
      }
    }
  }
#endif
}

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
  half *s_a = smem;
  half *s_b = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // 128x16
  constexpr int s_b_stage_offset = BK * (BN + B_PAD); // 16x128
  constexpr int s_a_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
  constexpr int s_b_mma_k_store_offset = K_STAGE * BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2;      // 0,1
  const int warp_n = warp_id / 2;      // 0,1,2,3

  int load_smem_a_m = tid / 2;                 // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // col 0,8
  int load_smem_b_k = tid / 16;                // row 0~15
  int load_smem_b_n = (tid % 16) * 8;          // col 0,8,16,...
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

    // 可以看到这里关于a矩阵加载了两次，是因为k维度翻倍变成了32
    // 但这里将32的k拆成了两个16，所以进行了两次加载，第一次加载前16的k，第二次加载后16的k
    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
             sizeof(half));
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
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
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
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
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
      reg_store_idx ^= 1; // 0->1
      reg_load_idx ^= 1;  // 1->0
      // 这个stage_sel负责计算收尾部分的buffer的index
      // 前面主循环结束后，还有K_STAGE-1个buffer的数据没有计算
      // 还不懂可以看hgemm_wmma_m16n16k16_mma4x2_warp2x4_stages_dsmem_kernel里面的注释
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

// NOTE: use ldmatrix.x4.trans for matrix B smem -> reg
// In order to reduce bank conflicts, we will save the K(16x2=32)
// dimension by half according to the stage dimension. For example,
// stages=3, warp_tile_k=2, it will be saved as [3*2][BM][16].
// 128x128, mma2x4, warp4x4(64,32,32), stages, block swizzle, dsmem,
// k32 with reg double buffers
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int WARP_TILE_K = 2, const int A_PAD = 0, const int B_PAD = 0,
          const int K_STAGE = 2, const bool BLOCK_SWIZZLE = true,
          const bool WARP_SWIZZLE = true>
__global__ void __launch_bounds__(256)
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel(
        const half *__restrict__ A, const half *__restrict__ B,
        half *__restrict__ C, int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K * WARP_TILE_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K;                            // 16x2=32

  extern __shared__ half smem[];
  half *s_a = smem;
  half *s_b = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // 128x16
  constexpr int s_b_stage_offset = BK * (BN + B_PAD); // 16x128
  constexpr int s_a_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
  constexpr int s_b_mma_k_store_offset = K_STAGE * BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2;      // 0,1
  const int warp_n = warp_id / 2;      // 0,1,2,3

  int load_smem_a_m = tid / 2;                 // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // col 0,8
  int load_smem_b_k = tid / 16;                // row 0~15
  int load_smem_b_n = (tid % 16) * 8;          // col 0,8,16,...
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

    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
             sizeof(half));
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

  uint32_t RA[2][WARP_TILE_M][4];
  uint32_t RB[2][WARP_TILE_N][2];

  int reg_store_idx = 0;
  int reg_load_idx = 1;

  {
// ldmatrix for s_a, ldmatrix.trans for s_b.
// smem -> reg buffers 0, first MMA_K, 0~15
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
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
      int lane_smem_b_n = warp_smem_b_n; // 0, MMA_N=8
      uint32_t lane_smem_b_ptr =
          (smem_b_base_ptr +
           s_b_mma_k_store_offset * sizeof(half) * (lane_id / 16) +
           (0 * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) +
            lane_smem_b_n) *
               sizeof(half));
      // TRICK: I use .x4.trans to load 4 matrix for reg double buffers at once.
      LDMATRIX_X4_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                    RB[reg_load_idx][j][0], RB[reg_load_idx][j][1],
                    lane_smem_b_ptr);
    }
  }

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
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      int warp_smem_a_m = warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M;
      int lane_smem_a_m = warp_smem_a_m + lane_id % 16; // 0~15
      int lane_smem_a_k = (lane_id / 16) * 8;           // 0,8
      uint32_t lane_smem_a_ptr =
          (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) +
           (smem_sel * s_a_stage_offset + lane_smem_a_m * (BK + A_PAD) +
            lane_smem_a_k) *
               sizeof(half));
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                  lane_smem_a_ptr);
    }

// MMA compute, first MMA_K
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

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    // load next k iters to reg buffers.
    // smem -> reg buffers 0, first MMA_K, 0~15
    // int smem_sel_reg = (k + 2) % K_STAGE; // vs smem_sel k=2->(0)1, k=3->(1)2
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
          (smem_b_base_ptr +
           s_b_mma_k_store_offset * sizeof(half) * (lane_id / 16) +
           (smem_sel_reg * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) +
            lane_smem_b_n) *
               sizeof(half));
      // may use .x4.trans to load 4 matrix for reg double buffers at once?
      LDMATRIX_X4_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                    RB[reg_load_idx][j][0], RB[reg_load_idx][j][1],
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
      reg_store_idx ^= 1; // 0->1
      reg_load_idx ^= 1;  // 1->0

      int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
// ldmatrix for s_a, ldmatrix.trans for s_b.
// smem -> reg buffers 1, second MMA_K
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

// MMA compute, first MMA_K
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
            (smem_b_base_ptr +
             s_b_mma_k_store_offset * sizeof(half) * (lane_id / 16) +
             (stage_sel_reg * s_b_stage_offset + lane_smem_b_k * (BN + B_PAD) +
              lane_smem_b_n) *
                 sizeof(half));
        // may use .x4.trans to load 4 matrix for reg double buffers at once?
        LDMATRIX_X4_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                      RB[reg_load_idx][j][0], RB[reg_load_idx][j][1],
                      lane_smem_b_ptr);
      }
    }
  }

  // collective store with reg reuse & warp shuffle
  for (int i = 0; i < WARP_TILE_M; ++i) {
// reuse RA[2][4][4] reg here, this may boost 0.3~0.5 TFLOPS up.
// may not put 'if' in N loop, it will crash the 'pragma unroll' hint ?
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      RA[0][j][0] = RC[i][j][0];
      RA[1][j][0] = RC[i][j][1];
      RA[0][j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
      RA[0][j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
      RA[0][j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
      RA[1][j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
      RA[1][j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
      RA[1][j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
    }

    if (lane_id % 4 == 0) {
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

// NOTE: reduce registers usage.
// In order to reduce bank conflicts, we will save the K(16x2=32)
// dimension by half according to the stage dimension. For example,
// stages=3, warp_tile_k=2, it will be saved as [3*2][BM][16].
// 128x128, mma2x4, warp4x4(64,32,32), stages, block swizzle, dsmem,
// k32 with reg double buffers
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int WARP_TILE_K = 2, const int A_PAD = 0, const int B_PAD = 0,
          const int K_STAGE = 2, const bool BLOCK_SWIZZLE = true,
          const bool WARP_SWIZZLE = true>
__global__ void __launch_bounds__(256)
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel(
        const half *__restrict__ A, const half *__restrict__ B,
        half *__restrict__ C, int M, int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K * WARP_TILE_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K;                            // 16x2=32

  extern __shared__ half smem[];
  half *s_a = smem;
  half *s_b = smem + K_STAGE * BM * (BK + A_PAD) * WARP_TILE_K;
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // 128x16
  constexpr int s_b_stage_offset = BK * (BN + B_PAD); // 16x128
  constexpr int s_a_mma_k_store_offset = K_STAGE * BM * (BK + A_PAD);
  constexpr int s_b_mma_k_store_offset = K_STAGE * BK * (BN + B_PAD);

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2;      // 0,1
  const int warp_n = warp_id / 2;      // 0,1,2,3

  const int load_smem_a_m = tid / 2;                 // row 0~127
  const int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // col 0,8
  const int load_smem_b_k = tid / 16;                // row 0~15
  const int load_smem_b_n = (tid % 16) * 8;          // col 0,8,16,...
  const int load_gmem_a_m = by * BM + load_smem_a_m; // global row of a and c
  const int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of b and c
  if (load_gmem_a_m >= M || load_gmem_b_n >= N)
    return;
  // 16 reg for pre-defined vars.

  uint32_t RC[WARP_TILE_M][WARP_TILE_N][2]; // 4*4*2=32 reg
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
  for (int k = 0; k < (K_STAGE - 1); ++k) {
    // reduce 9 registers -> 4 registers.
    // a gmem -> a smem
    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
             sizeof(half));
    CP_ASYNC_CG(load_smem_a_ptr,
                &A[load_gmem_a_m * K + k * BK * WARP_TILE_K + load_smem_a_k],
                16); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr =
        (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
             sizeof(half));
    CP_ASYNC_CG(
        load_smem_a_mma_k_ptr,
        &A[load_gmem_a_m * K + k * BK * WARP_TILE_K + load_smem_a_k + 16],
        16); // MMA_K 1

    // b gmem -> b smem
    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr +
         (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
             sizeof(half));
    CP_ASYNC_CG(load_smem_b_ptr,
                &B[(k * BK * WARP_TILE_K + load_smem_b_k) * N + load_gmem_b_n],
                16); // MMA_K 0
    uint32_t load_smem_b_mma_k_ptr =
        (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) +
         (k * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
             sizeof(half));
    CP_ASYNC_CG(
        load_smem_b_mma_k_ptr,
        &B[(k * BK * WARP_TILE_K + MMA_K + load_smem_b_k) * N + load_gmem_b_n],
        16); // MMA_K 1

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2); // s2->0, s3->1, s4->2
  __syncthreads();

  uint32_t RA[2][WARP_TILE_M][4]; // 2*4*4=32 reg
  uint32_t RB[2][WARP_TILE_N][2]; // 2*4*2=16 reg

  // 16+32+32+16=96 reg

  int reg_store_idx = 0;
  int reg_load_idx = 1;

  {
// ldmatrix for s_a, ldmatrix.trans for s_b.
// smem -> reg buffers 0, first MMA_K, 0~15
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_a_ptr =
          (smem_a_base_ptr +
           ((warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16) *
                (BK + A_PAD) +
            (lane_id / 16) * 8) *
               sizeof(half));
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                  lane_smem_a_ptr);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_b_ptr =
          (smem_b_base_ptr + ((lane_id % 16) * (BN + B_PAD) +
                              (warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)) *
                                 sizeof(half));
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                    lane_smem_b_ptr);
    }
  }

  int smem_sel = 0;                            // s3 k 2->0, k 3->1, k 4->2...
  int smem_sel_next = (K_STAGE - 1) % K_STAGE; // s3 k 2->2, k 3->0, k 4->1...
#pragma unroll
  for (int k = (K_STAGE - 1); k < NUM_K_TILES; ++k) {
    reg_store_idx ^= 1;           // 0->1
    reg_load_idx ^= 1;            // 1->0
    smem_sel = (k + 1) % K_STAGE; // s3 k 2->0, k 3->1, k 4->2...
    smem_sel_next = k % K_STAGE;  // s3 k 2->2, k 3->0, k 4->1...

    // reduce 9 registers -> 4 registers.
    // a gmem -> a smem
    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset +
                            load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
                               sizeof(half));
    CP_ASYNC_CG(load_smem_a_ptr,
                &A[load_gmem_a_m * K + k * BK * WARP_TILE_K + load_smem_a_k],
                16); // MMA_K 0
    uint32_t load_smem_a_mma_k_ptr =
        (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) +
         (smem_sel_next * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) +
          load_smem_a_k) *
             sizeof(half));
    CP_ASYNC_CG(
        load_smem_a_mma_k_ptr,
        &A[load_gmem_a_m * K + k * BK * WARP_TILE_K + load_smem_a_k + 16],
        16); // MMA_K 1

    // b gmem -> b smem
    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset +
                            load_smem_b_k * (BN + B_PAD) + load_smem_b_n) *
                               sizeof(half));
    CP_ASYNC_CG(load_smem_b_ptr,
                &B[(k * BK * WARP_TILE_K + load_smem_b_k) * N + load_gmem_b_n],
                16); // MMA_K 0
    uint32_t load_smem_b_mma_k_ptr =
        (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) +
         (smem_sel_next * s_b_stage_offset + load_smem_b_k * (BN + B_PAD) +
          load_smem_b_n) *
             sizeof(half));
    CP_ASYNC_CG(
        load_smem_b_mma_k_ptr,
        &B[(k * BK * WARP_TILE_K + MMA_K + load_smem_b_k) * N + load_gmem_b_n],
        16); // MMA_K 1

    CP_ASYNC_COMMIT_GROUP();

// ldmatrix for s_a, ldmatrix.trans for s_b.
// smem -> reg buffers 1, second MMA_K, 16~31
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_a_ptr =
          (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) +
           (smem_sel * s_a_stage_offset +
            (warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16) *
                (BK + A_PAD) +
            (lane_id / 16) * 8) *
               sizeof(half));
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                  lane_smem_a_ptr);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_b_ptr =
          (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) +
           (smem_sel * s_b_stage_offset + (lane_id % 16) * (BN + B_PAD) +
            (warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)) *
               sizeof(half));
      LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                    lane_smem_b_ptr);
    }

// MMA compute, first MMA_K
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

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    // load next k iters to reg buffers.
    // smem -> reg buffers 0, first MMA_K, 0~15
    // int smem_sel_reg = (smem_sel + 1) % K_STAGE; // vs smem_sel k=2->(0)1,
    // k=3->(1)2
    smem_sel = (smem_sel + 1) % K_STAGE;
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_a_ptr =
          (smem_a_base_ptr +
           (smem_sel * s_a_stage_offset +
            (warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16) *
                (BK + A_PAD) +
            (lane_id / 16) * 8) *
               sizeof(half));
      LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                  RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                  lane_smem_a_ptr);
    }

#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // reduce 4 registers -> 1 registers.
      uint32_t lane_smem_b_ptr =
          (smem_b_base_ptr +
           (smem_sel * s_b_stage_offset + (lane_id % 16) * (BN + B_PAD) +
            (warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)) *
               sizeof(half));
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
      reg_store_idx ^= 1; // 0->1
      reg_load_idx ^= 1;  // 1->0

      // int stage_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
      smem_sel = ((NUM_K_TILES - (K_STAGE - 1) + k) % K_STAGE);
// ldmatrix for s_a, ldmatrix.trans for s_b.
// smem -> reg buffers 1, second MMA_K
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // reduce 4 registers -> 1 registers.
        uint32_t lane_smem_a_ptr =
            (smem_a_base_ptr + s_a_mma_k_store_offset * sizeof(half) +
             (smem_sel * s_a_stage_offset +
              (warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16) *
                  (BK + A_PAD) +
              (lane_id / 16) * 8) *
                 sizeof(half));
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                    lane_smem_a_ptr);
      }

#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // reduce 4 registers -> 1 registers.
        uint32_t lane_smem_b_ptr =
            (smem_b_base_ptr + s_b_mma_k_store_offset * sizeof(half) +
             (smem_sel * s_b_stage_offset + (lane_id % 16) * (BN + B_PAD) +
              (warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)) *
                 sizeof(half));
        LDMATRIX_X2_T(RB[reg_store_idx][j][0], RB[reg_store_idx][j][1],
                      lane_smem_b_ptr);
      }

// MMA compute, first MMA_K
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
      // int stage_sel_reg = (stage_sel + 1) % K_STAGE;
      smem_sel = (smem_sel + 1) % K_STAGE;
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // reduce 4 registers -> 1 registers.
        uint32_t lane_smem_a_ptr =
            (smem_a_base_ptr +
             (smem_sel * s_a_stage_offset +
              (warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M + lane_id % 16) *
                  (BK + A_PAD) +
              (lane_id / 16) * 8) *
                 sizeof(half));
        LDMATRIX_X4(RA[reg_store_idx][i][0], RA[reg_store_idx][i][1],
                    RA[reg_store_idx][i][2], RA[reg_store_idx][i][3],
                    lane_smem_a_ptr);
      }

#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // reduce 4 registers -> 1 registers.
        uint32_t lane_smem_b_ptr =
            (smem_b_base_ptr +
             (smem_sel * s_b_stage_offset + (lane_id % 16) * (BN + B_PAD) +
              (warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)) *
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
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
      // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      RA[0][j][0] = RC[i][j][0];
      RA[1][j][0] = RC[i][j][1];
      RA[0][j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
      RA[0][j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
      RA[0][j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
      RA[1][j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
      RA[1][j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
      RA[1][j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
    }

    if (lane_id % 4 == 0) {
// reduce 6 registers -> 0 registers.
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        LDST128BITS(C[(by * BM + warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M +
                       lane_id / 4) *
                          N +
                      (bx * BN + warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)]) =
            (LDST128BITS(RA[0][j][0]));
        LDST128BITS(C[(by * BM + warp_m * (MMA_M * WARP_TILE_M) + i * MMA_M +
                       lane_id / 4 + 8) *
                          N +
                      (bx * BN + warp_n * (MMA_N * WARP_TILE_N) + j * MMA_N)]) =
            (LDST128BITS(RA[1][j][0]));
      }
    }
  }
}

// TODO: smem swizzle per 4x8 submatrix(not per line, 8 half values per line)
// In CUTLASS, each group of four threads is assigned a specific address in
// shared memory. This approach allows avoiding conflicts when reading from
// and writing to shared memory without increasing the shared memory usage.
// https://developer.download.nvidia.cn/video/gputechconf/gtc/2019/presentation
// /s9593-cutensor-high-performance-tensor-operations-in-cuda-v2.pdf
// A matrix smem, MMA_MxMMA_K=16x16; B matrix smem, MMA_KxMMA_N=16x8; PTX
// layout. reference: https://zhuanlan.zhihu.com/p/638522893
// https://zhuanlan.zhihu.com/p/696231622
// https://www.zhihu.com/question/600927104/answer/3029266372
// https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
// #warp-level-matrix-load-instruction-ldmatrix

// build cpp binary
#ifndef NO_MMA_HGEMM_BIN

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

// 128x128, mma2x4, warp4x4x2(64,32,32), stages, block swizzle, dsmem, reg
// double buffers
template <const int K_STAGE = 2, const int BLOCK_SWIZZLE_STRIDE = 2048>
void lanunch_hgemm_mma_m16n8k16_nn(half *a, half *b, half *c, int M, int N,
                                   int K) {
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
  // ./hgemm_mma_stage.debug.89.bin ncu --metrics
  // sm__sass_l1tex_data_bank_conflicts_pipe_lsu_mem_shared_op_ldsm
  // ./hgemm_mma_stage.debug.89.bin
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
  LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_KERNEL(
      K_STAGE, BLOCK_SWIZZLE_STRIDE);
}

#ifdef HGEMM_MMA_DEBUG
#include <iostream>
#endif

int main(int argc, char *argv[]) {
#ifdef HGEMM_MMA_DEBUG
  const int test_num = 1;
#else
  const int test_num = 64;
#endif
  int M_list[test_num];
  int N_list[test_num];
  int K_list[test_num];

  for (int i = 0; i < test_num; i++) {
    M_list[i] = (i + 1) * 256;
    N_list[i] = (i + 1) * 256;
    K_list[i] = (i + 1) * 256;
  }

#ifdef HGEMM_MMA_DEBUG
  if (argc > 1)
    M_list[0] = std::stoi(argv[1]);
  if (argc > 2)
    N_list[0] = std::stoi(argv[2]);
  if (argc > 3)
    K_list[0] = std::stoi(argv[3]);
#endif

#ifdef HGEMM_MMA_DEBUG
  int outer_repeat = 1, inner_repeat = 1, warmup = 1;
  if (argc > 4)
    warmup = std::stoi(argv[4]);
  if (argc > 5)
    inner_repeat = std::stoi(argv[5]);
#else
  int outer_repeat = 10, inner_repeat = 1, warmup = 1;
#endif

  printf("ALGO = MMA16816 HGEMM NN MMA=2x4 WARP=4x4x2 STAGES=2 BLOCK "
         "SWIZZLE=2048\n");
#ifndef HGEMM_MMA_DEBUG
  for (int j = 0; j < 5; j++) {
    int M = M_list[j], N = N_list[j], K = K_list[j];
    float max_error = gemm_error_check_nn<half>(
        lanunch_hgemm_mma_m16n8k16_nn<2, 2048>, M, N, K);
    printf("M N K = %6d %6d %6d, ", M, N, K);
    printf("Max Error = %f\n", max_error);
  }
#endif

  for (int j = 0; j < test_num; j++) {
    int M = M_list[j], N = N_list[j], K = K_list[j];

    double max_sec = 0.0;
    double min_sec = DBL_MAX;
    double total_sec = 0.0;

    for (int k = 0; k < outer_repeat; k++) {
      double this_sec = perf_gemm<half>(lanunch_hgemm_mma_m16n8k16_nn<2, 2048>,
                                        M, N, K, inner_repeat, warmup);
      max_sec = max(max_sec, this_sec);
      min_sec = min(min_sec, this_sec);
      total_sec += this_sec;
    }

    // 1 TFLOPS = 10^12 FLOPS
    // ref: https://imgtec.eetrend.com/blog/2021/100062210.html.
    double avg_sec = total_sec / outer_repeat;
    double avg_Tflops = ((double)M) * N * K * 2 * 1e-12 / avg_sec;

    printf("M N K = %6d %6d %6d, ", M, N, K);
    printf("Time = %12.8lf %12.8lf %12.8lf s, ", min_sec, avg_sec, max_sec);
    printf("AVG Performance = %10.4lf Tflops\n", avg_Tflops);
  }

  return 0;
}

#else

#include <torch/extension.h>
#include <torch/types.h>
#define STRINGFY(str) #str
#define TORCH_BINDING_COMMON_EXTENSION(func)                                   \
  m.def(STRINGFY(func), &func, STRINGFY(func));

#define CHECK_TORCH_TENSOR_DTYPE(T, th_type)                                   \
  if (((T).options().dtype() != (th_type))) {                                  \
    std::cout << "Tensor Info:" << (T).options() << std::endl;                 \
    throw std::runtime_error("values must be " #th_type);                      \
  }

#define CHECK_TORCH_TENSOR_SHAPE(T, S0, S1)                                    \
  if (((T).size(0) != (S0)) || ((T).size(1) != (S1))) {                        \
    throw std::runtime_error("Tensor size mismatch!");                         \
  }

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle
#define LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_KERNEL(stages, stride)       \
  {                                                                            \
    const int N_SWIZZLE = (N + (stride) - 1) / (stride);                       \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),  \
              N_SWIZZLE);                                                      \
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel<                           \
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M, WARP_TILE_N, \
        A_PAD, B_PAD, (stages), true>                                          \
        <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),              \
                          reinterpret_cast<half *>(b.data_ptr()),              \
                          reinterpret_cast<half *>(c.data_ptr()), M, N, K);    \
  }

#define LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_KERNEL(stages)            \
  {                                                                            \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                               \
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_kernel<                           \
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M, WARP_TILE_N, \
        A_PAD, B_PAD, (stages), false>                                         \
        <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),              \
                          reinterpret_cast<half *>(b.data_ptr()),              \
                          reinterpret_cast<half *>(c.data_ptr()), M, N, K);    \
  }

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle
void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages(torch::Tensor a, torch::Tensor b,
                                              torch::Tensor c, int stages,
                                              bool swizzle,
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
  // constexpr int BK = MMA_K;
  // s2: 2*128*(16)*2=8KB,  2*16*(128+16)*2=9KB,    ~17KB
  // s3: 3*128*(16)*2=12KB, 3*16*(128+16)*2=13.5KB, ~26KB
  // s4: 4*128*(16)*2=16KB, 4*16*(128+16)*2=18KB,   ~34KB
  // s5: 5*128*(16)*2=20KB, 5*16*(128+16)*2=22.5KB, ~43KB
  if (swizzle) {
    // assert(swizzle_stride % 256 == 0);
    switch (stages) {
    case 2: // ~17KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_KERNEL(2, swizzle_stride);
      break;
    case 3: // ~26KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_KERNEL(3, swizzle_stride);
      break;
    case 4: // ~34KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_KERNEL(4, swizzle_stride);
      break;
    // case 5: // ~43KB
    //   LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_KERNEL(5, swizzle_stride);
    //   break;
    default:
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages) {
    case 2:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_KERNEL(2);
      break;
    case 3:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_KERNEL(3);
      break;
    case 4:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_KERNEL(4);
      break;
    // case 5:
    //   LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_KERNEL(5);
    //   break;
    default:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_KERNEL(2);
      break;
    }
  }
}

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem
#define LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(stages, stride) \
  {                                                                            \
    const int smem_max_size = ((stages) * BM * (BK + A_PAD) * sizeof(half) +   \
                               (stages) * BK * (BN + B_PAD) * sizeof(half));   \
    cudaFuncSetAttribute(                                                      \
        hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel<                 \
            MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,          \
            WARP_TILE_N, A_PAD, B_PAD, (stages), true>,                        \
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);                   \
    const int N_SWIZZLE = (N + (stride) - 1) / (stride);                       \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),  \
              N_SWIZZLE);                                                      \
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel<                     \
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M, WARP_TILE_N, \
        A_PAD, B_PAD, (stages), true><<<grid, block, smem_max_size>>>(         \
        reinterpret_cast<half *>(a.data_ptr()),                                \
        reinterpret_cast<half *>(b.data_ptr()),                                \
        reinterpret_cast<half *>(c.data_ptr()), M, N, K);                      \
  }

#define LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(stages)      \
  {                                                                            \
    const int smem_max_size = ((stages) * BM * (BK + A_PAD) * sizeof(half) +   \
                               (stages) * BK * (BN + B_PAD) * sizeof(half));   \
    cudaFuncSetAttribute(                                                      \
        hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel<                 \
            MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,          \
            WARP_TILE_N, A_PAD, B_PAD, (stages), false>,                       \
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);                   \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                               \
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_kernel<                     \
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M, WARP_TILE_N, \
        A_PAD, B_PAD, (stages), false><<<grid, block, smem_max_size>>>(        \
        reinterpret_cast<half *>(a.data_ptr()),                                \
        reinterpret_cast<half *>(b.data_ptr()),                                \
        reinterpret_cast<half *>(c.data_ptr()), M, N, K);                      \
  }

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem
void hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem(torch::Tensor a,
                                                    torch::Tensor b,
                                                    torch::Tensor c, int stages,
                                                    bool swizzle,
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
  // s2: 2*128*(16)*2=8KB,  2*16*(128+16)*2=9KB,    ~17KB
  // s3: 3*128*(16)*2=12KB, 3*16*(128+16)*2=13.5KB, ~26KB
  // s4: 4*128*(16)*2=16KB, 4*16*(128+16)*2=18KB,   ~34KB
  // s5: 5*128*(16)*2=20KB, 5*16*(128+16)*2=22.5KB, ~43KB
  if (swizzle) {
    // assert(swizzle_stride % 256 == 0);
    switch (stages) {
    case 2: // ~17KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(2, swizzle_stride);
      break;
    case 3: // ~26KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(3, swizzle_stride);
      break;
    case 4: // ~34KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(4, swizzle_stride);
      break;
    case 5: // ~43KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(5, swizzle_stride);
      break;
    default:
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(2, swizzle_stride);
      break;
    }
  } else {
    switch (stages) {
    case 2:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(2);
      break;
    case 3:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(3);
      break;
    case 4:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(4);
      break;
    case 5:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(5);
      break;
    default:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4_DSMEM_KERNEL(2);
      break;
    }
  }
}

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
        <<<grid, block, smem_max_size>>>(                                      \
            reinterpret_cast<half *>(a.data_ptr()),                            \
            reinterpret_cast<half *>(b.data_ptr()),                            \
            reinterpret_cast<half *>(c.data_ptr()), M, N, K);                  \
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

// use ldmatrix.x4.trans for matrix B smem -> reg
// 128x128, mma2x4, warp4x4x2(64,32,32), stages, block swizzle, dsmem, reg
// double buffers
#define LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_X4_KERNEL(stages,    \
                                                                    stride)    \
  {                                                                            \
    const int smem_max_size =                                                  \
        ((stages) * BM * (BK + A_PAD) * WARP_TILE_K * sizeof(half) +           \
         (stages) * BK * (BN + B_PAD) * WARP_TILE_K * sizeof(half));           \
    cudaFuncSetAttribute(                                                      \
        hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel<            \
            MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,          \
            WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), true>,           \
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);                   \
    const int N_SWIZZLE = (N + (stride) - 1) / (stride);                       \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),  \
              N_SWIZZLE);                                                      \
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel<                \
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M, WARP_TILE_N, \
        WARP_TILE_K, A_PAD, B_PAD, (stages), true>                             \
        <<<grid, block, smem_max_size>>>(                                      \
            reinterpret_cast<half *>(a.data_ptr()),                            \
            reinterpret_cast<half *>(b.data_ptr()),                            \
            reinterpret_cast<half *>(c.data_ptr()), M, N, K);                  \
  }

#define LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_X4_KERNEL(stages) \
  {                                                                            \
    const int smem_max_size =                                                  \
        ((stages) * BM * (BK + A_PAD) * WARP_TILE_K * sizeof(half) +           \
         (stages) * BK * (BN + B_PAD) * WARP_TILE_K * sizeof(half));           \
    cudaFuncSetAttribute(                                                      \
        hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel<            \
            MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,          \
            WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), false>,          \
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);                   \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                               \
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4_kernel<                \
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M, WARP_TILE_N, \
        WARP_TILE_K, A_PAD, B_PAD, (stages), false>                            \
        <<<grid, block, smem_max_size>>>(                                      \
            reinterpret_cast<half *>(a.data_ptr()),                            \
            reinterpret_cast<half *>(b.data_ptr()),                            \
            reinterpret_cast<half *>(c.data_ptr()), M, N, K);                  \
  }

// 128x128, mma2x4, warp4x4x2(64,32,32), stages, block swizzle, dsmem, reg
// double buffers
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_x4(
    torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle,
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
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_X4_KERNEL(
          2, swizzle_stride);
      break;
    case 3: // ~51KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_X4_KERNEL(
          3, swizzle_stride);
      break;
    case 4: // ~68KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_X4_KERNEL(
          4, swizzle_stride);
      break;
    case 5: // ~85KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_X4_KERNEL(
          5, swizzle_stride);
      break;
    default:
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_X4_KERNEL(
          2, swizzle_stride);
      break;
    }
  } else {
    switch (stages) {
    case 2:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_X4_KERNEL(2);
      break;
    case 3:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_X4_KERNEL(3);
      break;
    case 4:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_X4_KERNEL(4);
      break;
    case 5:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_X4_KERNEL(5);
      break;
    default:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_X4_KERNEL(2);
      break;
    }
  }
}

// use ldmatrix.x4.trans for matrix B smem -> reg
// 128x128, mma2x4, warp4x4x2(64,32,32), stages, block swizzle, dsmem, reg
// double buffers
#define LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_RR_KERNEL(stages,    \
                                                                    stride)    \
  {                                                                            \
    const int smem_max_size =                                                  \
        ((stages) * BM * (BK + A_PAD) * WARP_TILE_K * sizeof(half) +           \
         (stages) * BK * (BN + B_PAD) * WARP_TILE_K * sizeof(half));           \
    cudaFuncSetAttribute(                                                      \
        hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel<            \
            MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,          \
            WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), true>,           \
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);                   \
    const int N_SWIZZLE = (N + (stride) - 1) / (stride);                       \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),  \
              N_SWIZZLE);                                                      \
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel<                \
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M, WARP_TILE_N, \
        WARP_TILE_K, A_PAD, B_PAD, (stages), true>                             \
        <<<grid, block, smem_max_size>>>(                                      \
            reinterpret_cast<half *>(a.data_ptr()),                            \
            reinterpret_cast<half *>(b.data_ptr()),                            \
            reinterpret_cast<half *>(c.data_ptr()), M, N, K);                  \
  }

#define LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_RR_KERNEL(stages) \
  {                                                                            \
    const int smem_max_size =                                                  \
        ((stages) * BM * (BK + A_PAD) * WARP_TILE_K * sizeof(half) +           \
         (stages) * BK * (BN + B_PAD) * WARP_TILE_K * sizeof(half));           \
    cudaFuncSetAttribute(                                                      \
        hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel<            \
            MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,          \
            WARP_TILE_N, WARP_TILE_K, A_PAD, B_PAD, (stages), false>,          \
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);                   \
    dim3 block(NUM_THREADS);                                                   \
    dim3 grid(div_ceil(N, BN), div_ceil(M, BM));                               \
    hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr_kernel<                \
        MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M, WARP_TILE_N, \
        WARP_TILE_K, A_PAD, B_PAD, (stages), false>                            \
        <<<grid, block, smem_max_size>>>(                                      \
            reinterpret_cast<half *>(a.data_ptr()),                            \
            reinterpret_cast<half *>(b.data_ptr()),                            \
            reinterpret_cast<half *>(c.data_ptr()), M, N, K);                  \
  }

// reduce registers usage.
// 128x128, mma2x4, warp4x4x2(64,32,32), stages, block swizzle, dsmem, reg
// double buffers
void hgemm_mma_m16n8k16_mma2x4_warp4x4x2_stages_dsmem_rr(
    torch::Tensor a, torch::Tensor b, torch::Tensor c, int stages, bool swizzle,
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
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_RR_KERNEL(
          2, swizzle_stride);
      break;
    case 3: // ~51KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_RR_KERNEL(
          3, swizzle_stride);
      break;
    case 4: // ~68KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_RR_KERNEL(
          4, swizzle_stride);
      break;
    case 5: // ~85KB
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_RR_KERNEL(
          5, swizzle_stride);
      break;
    default:
      LAUNCH_16816_STAGE_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_RR_KERNEL(
          2, swizzle_stride);
      break;
    }
  } else {
    switch (stages) {
    case 2:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_RR_KERNEL(2);
      break;
    case 3:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_RR_KERNEL(3);
      break;
    case 4:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_RR_KERNEL(4);
      break;
    case 5:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_RR_KERNEL(5);
      break;
    default:
      LAUNCH_16816_STAGE_NO_SWIZZLE_MMA2x4_WARP4x4x2_DSMEM_RR_KERNEL(2);
      break;
    }
  }
}

#endif
