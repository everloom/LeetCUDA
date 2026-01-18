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

// NN: A/B/C All row major
// TN: A row major MxK, B col major NxK, C row major MxN
// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void __launch_bounds__(256)
    hgemm_mma_m16n8k16_mma2x4_warp4x4_stages_dsmem_tn_kernel(half *A, half *B,
                                                             half *C, int M,
                                                             int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K;                            // 16

  extern __shared__ half smem[];
  half *s_a = smem;
  half *s_b = smem + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // BMxBK 128*16
  constexpr int s_b_stage_offset = BN * (BK + B_PAD); // BNxBK 128*16

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2;      // 0,1
  const int warp_n = warp_id / 2;      // 0,1,2,3

  int load_smem_a_m = tid / 2;                 // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // col 0,8
  int load_smem_b_n = tid / 2;                 // row 0~127
  int load_smem_b_k = (tid % 2 == 0) ? 0 : 8;  // col 0,8
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of c
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

  // may avoid cvta overhead ? only cvta smem base ptr once for cp.async.
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) {     // 0, 1
    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global col of b
    int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k;

    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
             sizeof(half));
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr +
         (k * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) + load_smem_b_k) *
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
    int load_gmem_b_k = k * BK + load_smem_b_k; // global col of b
    int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k;

    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset +
                            load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
                               sizeof(half));
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset +
                            load_smem_b_n * (BK + B_PAD) + load_smem_b_k) *
                               sizeof(half));
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();

    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];
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
      int lane_smem_b_n = warp_smem_b_n + lane_id % 8; // 0~7, MMA_N=8
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8;     // 0,8
      uint32_t lane_smem_b_ptr =
          (smem_b_base_ptr + (smem_sel * s_b_stage_offset +
                              lane_smem_b_n * (BK + B_PAD) + lane_smem_b_k) *
                                 sizeof(half));
      LDMATRIX_X2(RB[j][0], RB[j][1], lane_smem_b_ptr);
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
  if ((K_STAGE - 2) > 0) {
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
        int lane_smem_b_n = warp_smem_b_n + lane_id % 8; // 0~7, MMA_N=8
        int lane_smem_b_k = ((lane_id / 8) % 2) * 8;     // 0,8
        uint32_t lane_smem_b_ptr =
            (smem_b_base_ptr + (stage_sel * s_b_stage_offset +
                                lane_smem_b_n * (BK + B_PAD) + lane_smem_b_k) *
                                   sizeof(half));
        LDMATRIX_X2(RB[j][0], RB[j][1], lane_smem_b_ptr);
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
}


// NN: A/B/C All row major
// TN: A row major MxK, B col major NxK, C row major MxN
// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
          const int MMA_TILE_M = 2, const int MMA_TILE_N = 4,
          const int WARP_TILE_M = 4, const int WARP_TILE_N = 4,
          const int A_PAD = 0, const int B_PAD = 0, const int K_STAGE = 2,
          const bool BLOCK_SWIZZLE = false>
__global__ void __launch_bounds__(256)
    cosine_kernel(half *A, half *B,
                                                             half *C, int M,
                                                             int N, int K) {
  // BLOCK_SWIZZLE 0/1 control use block swizzle or not.
  const int bx = ((int)BLOCK_SWIZZLE) * blockIdx.z * gridDim.x + blockIdx.x;
  const int by = blockIdx.y;
  const int NUM_K_TILES = div_ceil(K, MMA_K);
  constexpr int BM = MMA_M * MMA_TILE_M * WARP_TILE_M; // 16*2*4=128
  constexpr int BN = MMA_N * MMA_TILE_N * WARP_TILE_N; // 8*4*4=128
  constexpr int BK = MMA_K;                            // 16

  extern __shared__ half smem[];
  half *s_a = smem;
  half *s_b = smem + K_STAGE * BM * (BK + A_PAD);
  constexpr int s_a_stage_offset = BM * (BK + A_PAD); // BMxBK 128*16
  constexpr int s_b_stage_offset = BN * (BK + B_PAD); // BNxBK 128*16

  const int tid = threadIdx.y * blockDim.x + threadIdx.x; // within block
  const int warp_id = tid / WARP_SIZE; // 0~7 warp_id within block
  const int lane_id = tid % WARP_SIZE; // 0~31
  const int warp_m = warp_id % 2;      // 0,1
  const int warp_n = warp_id / 2;      // 0,1,2,3

  int load_smem_a_m = tid / 2;                 // row 0~127
  int load_smem_a_k = (tid % 2 == 0) ? 0 : 8;  // col 0,8
  int load_smem_b_n = tid / 2;                 // row 0~127
  int load_smem_b_k = (tid % 2 == 0) ? 0 : 8;  // col 0,8
  int load_gmem_a_m = by * BM + load_smem_a_m; // global row of c
  int load_gmem_b_n = bx * BN + load_smem_b_n; // global col of c
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

  // may avoid cvta overhead ? only cvta smem base ptr once for cp.async.
  uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
  uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

#pragma unroll
  for (int k = 0; k < (K_STAGE - 1); ++k) {     // 0, 1
    int load_gmem_a_k = k * BK + load_smem_a_k; // global col of a
    int load_gmem_a_addr = load_gmem_a_m * K + load_gmem_a_k;
    int load_gmem_b_k = k * BK + load_smem_b_k; // global col of b
    int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k;

    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr +
         (k * s_a_stage_offset + load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
             sizeof(half));
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr +
         (k * s_b_stage_offset + load_smem_b_n * (BK + B_PAD) + load_smem_b_k) *
             sizeof(half));
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();
  }

  CP_ASYNC_WAIT_GROUP(K_STAGE - 2); // s2->0, s3->1, s4->2
  __syncthreads();


  // 说一下这里ra_norm和rb_norm的shape的问题
  // 因为对于m16n8k16的mma来说，对于A矩阵，每个thread的RA寄存器的数据分布在两行，所以这里ra_norm的shape的2就对应了两行
  // 对于B矩阵，每个thread的RB寄存器的数据分布在一列，所以没有ra_norm那样的第二维
  float RA_NORM[WARP_TILE_M][2];
  float RB_NORM[WARP_TILE_N];
  for(int i = 0; i < WARP_TILE_M; ++i) {
    RA_NORM[i][0] = 0.0f;
    RA_NORM[i][1] = 0.0f;
  }

  for(int i = 0; i < WARP_TILE_N; ++i) {
    RB_NORM[i] = 0.0f;
  }

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
    int load_gmem_b_k = k * BK + load_smem_b_k; // global col of b
    int load_gmem_b_addr = load_gmem_b_n * K + load_gmem_b_k;

    uint32_t load_smem_a_ptr =
        (smem_a_base_ptr + (smem_sel_next * s_a_stage_offset +
                            load_smem_a_m * (BK + A_PAD) + load_smem_a_k) *
                               sizeof(half));
    CP_ASYNC_CG(load_smem_a_ptr, &A[load_gmem_a_addr], 16);

    uint32_t load_smem_b_ptr =
        (smem_b_base_ptr + (smem_sel_next * s_b_stage_offset +
                            load_smem_b_n * (BK + B_PAD) + load_smem_b_k) *
                               sizeof(half));
    CP_ASYNC_CG(load_smem_b_ptr, &B[load_gmem_b_addr], 16);

    CP_ASYNC_COMMIT_GROUP();

    uint32_t RA[WARP_TILE_M][4];
    uint32_t RB[WARP_TILE_N][2];
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
      int lane_smem_b_n = warp_smem_b_n + lane_id % 8; // 0~7, MMA_N=8
      int lane_smem_b_k = ((lane_id / 8) % 2) * 8;     // 0,8
      uint32_t lane_smem_b_ptr =
          (smem_b_base_ptr + (smem_sel * s_b_stage_offset +
                              lane_smem_b_n * (BK + B_PAD) + lane_smem_b_k) *
                                 sizeof(half));
      LDMATRIX_X2(RB[j][0], RB[j][1], lane_smem_b_ptr);
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
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
        // [FIX] Extract 2 halfs from uint32, convert to float, compute sum of squares
        // RA[...] is uint32_t, holding 2 packed halfs.
        // 1. Reinterpret uint32_t as half2
        half2 val_h2 = reinterpret_cast<half2&>(RA[i][0]);
        // 2. Convert half2 to float2 to avoid overflow during square
        float2 val_f2 = __half22float2(val_h2);
        // 3. Accumulate a^2 + b^2
        RA_NORM[i][0] = __fmaf_rn(val_f2.x, val_f2.x, RA_NORM[i][0]);
        RA_NORM[i][0] = __fmaf_rn(val_f2.y, val_f2.y, RA_NORM[i][0]);

        // 2. Process RA[...][2] -> Accumulate to RA_NORM[i][0]
        val_h2 = reinterpret_cast<half2&>(RA[i][2]);
        val_f2 = __half22float2(val_h2);
        RA_NORM[i][0] = __fmaf_rn(val_f2.x, val_f2.x, RA_NORM[i][0]);
        RA_NORM[i][0] = __fmaf_rn(val_f2.y, val_f2.y, RA_NORM[i][0]);

        // 3. Process RA[...][1] -> Accumulate to RA_NORM[i][1]
        val_h2 = reinterpret_cast<half2&>(RA[i][1]);
        val_f2 = __half22float2(val_h2);
        RA_NORM[i][1] = __fmaf_rn(val_f2.x, val_f2.x, RA_NORM[i][1]);
        RA_NORM[i][1] = __fmaf_rn(val_f2.y, val_f2.y, RA_NORM[i][1]);

        // 4. Process RA[...][3] -> Accumulate to RA_NORM[i][1]
        val_h2 = reinterpret_cast<half2&>(RA[i][3]);
        val_f2 = __half22float2(val_h2);
        RA_NORM[i][1] = __fmaf_rn(val_f2.x, val_f2.x, RA_NORM[i][1]);
        RA_NORM[i][1] = __fmaf_rn(val_f2.y, val_f2.y, RA_NORM[i][1]);
    }
#pragma unroll
    for(int i = 0; i < WARP_TILE_N; i++){
        half2 val_h2 = reinterpret_cast<half2&>(RB[i][0]);
        float2 val_f2 = __half22float2(val_h2);
        RB_NORM[i] = __fmaf_rn(val_f2.x, val_f2.x, RB_NORM[i]);
        RB_NORM[i] = __fmaf_rn(val_f2.y, val_f2.y, RB_NORM[i]);

        val_h2 = reinterpret_cast<half2&>(RB[i][1]);
        val_f2 = __half22float2(val_h2);
        RB_NORM[i] = __fmaf_rn(val_f2.x, val_f2.x, RB_NORM[i]);
        RB_NORM[i] = __fmaf_rn(val_f2.y, val_f2.y, RB_NORM[i]);
    }

    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();
  }

  // make sure all memory issues ready.
  if ((K_STAGE - 2) > 0) {
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
        int lane_smem_b_n = warp_smem_b_n + lane_id % 8; // 0~7, MMA_N=8
        int lane_smem_b_k = ((lane_id / 8) % 2) * 8;     // 0,8
        uint32_t lane_smem_b_ptr =
            (smem_b_base_ptr + (stage_sel * s_b_stage_offset +
                                lane_smem_b_n * (BK + B_PAD) + lane_smem_b_k) *
                                   sizeof(half));
        LDMATRIX_X2(RB[j][0], RB[j][1], lane_smem_b_ptr);
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
#pragma unroll
      for (int i = 0; i < WARP_TILE_M; ++i) {
        // [FIX] Extract 2 halfs from uint32, convert to float, compute sum of squares
        // RA[...] is uint32_t, holding 2 packed halfs.
        // 1. Reinterpret uint32_t as half2
        half2 val_h2 = reinterpret_cast<half2&>(RA[i][0]);
        // 2. Convert half2 to float2 to avoid overflow during square
        float2 val_f2 = __half22float2(val_h2);
        // 3. Accumulate a^2 + b^2
        RA_NORM[i][0] = __fmaf_rn(val_f2.x, val_f2.x, RA_NORM[i][0]);
        RA_NORM[i][0] = __fmaf_rn(val_f2.y, val_f2.y, RA_NORM[i][0]);

        // 2. Process RA[...][2] -> Accumulate to RA_NORM[i][0]
        val_h2 = reinterpret_cast<half2&>(RA[i][2]);
        val_f2 = __half22float2(val_h2);
        RA_NORM[i][0] = __fmaf_rn(val_f2.x, val_f2.x, RA_NORM[i][0]);
        RA_NORM[i][0] = __fmaf_rn(val_f2.y, val_f2.y, RA_NORM[i][0]);

        // 3. Process RA[...][1] -> Accumulate to RA_NORM[i][1]
        val_h2 = reinterpret_cast<half2&>(RA[i][1]);
        val_f2 = __half22float2(val_h2);
        RA_NORM[i][1] = __fmaf_rn(val_f2.x, val_f2.x, RA_NORM[i][1]);
        RA_NORM[i][1] = __fmaf_rn(val_f2.y, val_f2.y, RA_NORM[i][1]);

        // 4. Process RA[...][3] -> Accumulate to RA_NORM[i][1]
        val_h2 = reinterpret_cast<half2&>(RA[i][3]);
        val_f2 = __half22float2(val_h2);
        RA_NORM[i][1] = __fmaf_rn(val_f2.x, val_f2.x, RA_NORM[i][1]);
        RA_NORM[i][1] = __fmaf_rn(val_f2.y, val_f2.y, RA_NORM[i][1]);
    }
#pragma unroll
    for(int i = 0; i < WARP_TILE_N; i++){
        half2 val_h2 = reinterpret_cast<half2&>(RB[i][0]);
        float2 val_f2 = __half22float2(val_h2);
        RB_NORM[i] = __fmaf_rn(val_f2.x, val_f2.x, RB_NORM[i]);
        RB_NORM[i] = __fmaf_rn(val_f2.y, val_f2.y, RB_NORM[i]);

        val_h2 = reinterpret_cast<half2&>(RB[i][1]);
        val_f2 = __half22float2(val_h2);
        RB_NORM[i] = __fmaf_rn(val_f2.x, val_f2.x, RB_NORM[i]);
        RB_NORM[i] = __fmaf_rn(val_f2.y, val_f2.y, RB_NORM[i]);
    }

    }
  }
  // printf("here 1\n");
        /*
    A矩阵的范数：
    这里ra_norm的shape是[WARP_TILE_M][2]
    首先参考ptx中关于A的mma的layout，使用一个warp reduce sum，将相邻几个线程的结果做累加
    具体的是，将相邻3个线程的结果累加到最开始的那个线程
    以第0号线程为例，第0号线程的ra_norm[i][0,1]存放了第0行和第8行的部分累加结果，reduce sum需要做的是
        将1号线程、2号线程、3号线程的ra_norm[i][0,1]的结果都累加到0号线程的ra_norm[i][0,1]上
    然后是第4号线程同理，reduce sum将第5、6、7号线程的ra_norm[i][0,1]累加到第4号线程上
    这样操作后，0、4、8、12、16、20、24、28号线程就分别得到了(0,8) (1,9) (2,10) (3,11) (4,12) (5,13) (6,14) (7,15)行的完整累加结果
    然后再对0、4、8、12、16、20、24、28号线程上的累加结果开平方就得到了对应行的L2范数

    B矩阵的范数：
    rb_norm的shape是[WARP_TILE_N]
    参考ptx中对于B的mma的layout，同样也需要一个warp reduce sum
    以第0号线程为例，0线程将1 2 3号线程的rb_norm[j]累加到自己的rb_norm[j]上
    4号线程将5 6 7号线程的rb_norm[j]累加到自己的rb_norm[j]上
    这样0 4 8 12 16 20 24 28号线程就分别得到了第0 1 2 3 4 5 6 7列的完整累加结果
    然后做一个开平方就得到了对应列的L2范数
    */
   // 这里实际运行的时候发现了一个问题
  //  就是这里的shfl函数如果使用__shfl_down_sync会出现卡死的现象，但改为__shfl_sync就不会
  //  使用__shfl_down_sync时srclaneid那里还是甜的lane_id + 1 / 2 / 3这类的
  //  卡死的原因没具体分析，后面有时间了再分析一下
#pragma unroll
    for (int i = 0; i < WARP_TILE_M; ++i) {
        float tmp = RA_NORM[i][0];
        tmp += __shfl_sync(0xffffffff, RA_NORM[i][0], lane_id + 1);
        tmp += __shfl_sync(0xffffffff, RA_NORM[i][0], lane_id + 2);
        tmp += __shfl_sync(0xffffffff, RA_NORM[i][0], lane_id + 3);
        RA_NORM[i][0] = tmp;

        tmp = RA_NORM[i][1];
        tmp += __shfl_sync(0xffffffff, RA_NORM[i][1], lane_id + 1);
        tmp += __shfl_sync(0xffffffff, RA_NORM[i][1], lane_id + 2);
        tmp += __shfl_sync(0xffffffff, RA_NORM[i][1], lane_id + 3);
        RA_NORM[i][1] = tmp;
    }

    // 实现 B 矩阵范数的 Warp Reduce Sum
#pragma unroll
    for (int j = 0; j < WARP_TILE_N; ++j) {
        float tmp = RB_NORM[j];
        tmp += __shfl_sync(0xffffffff, RB_NORM[j], lane_id + 1);
        tmp += __shfl_sync(0xffffffff, RB_NORM[j], lane_id + 2);
        tmp += __shfl_sync(0xffffffff, RB_NORM[j], lane_id + 3);
        RB_NORM[j] = tmp;
    }
    // printf("here 2\n");
  {
    for (int i = 0; i < WARP_TILE_M; ++i) {
      // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
      // thus, we only need 8 memory issues with 128 bits after shfl_sync.
      // may reuse RA[4][4] as RC0 ? only new RC1[4][4].
      uint32_t RC0[WARP_TILE_N][4];
      uint32_t RC1[WARP_TILE_N][4];
#pragma unroll
      for (int j = 0; j < WARP_TILE_N; ++j) {
        // printf("here 2 1\n");
        // How to use LDST128BITS here? __shfl_sync -> lane 0 -> store 8 half.
        // thus, we only need 8 memory issues with 128 bits after shfl_sync.
        // /*
        // // 这里i表示当前正在处理的warp负责区域中的哪一个 16 行的水平条带
        // // A的threadblock tile的大小是128*16，由于MMA_TILE_M = 2，所以实际上A的warp tile大小是64*16
        // // B的threadblock tile大小是12**16，由于MMA_TILE_N = 4，所以实际上B的warp tile大小是32*16
        // // 同时由于WARP_TILE_M为4，所以A的warp tile被横向划分为了4个16*16的矩阵，这里的i就表示当前16*16矩阵是第几个
        // // WARP_TILE_N为4，所以B的warp tile被纵向划分为了4个8*16的矩阵
        // // 这里threadblock tile大小是128*128，被分成了2*4份（二维），每份64*32，相当于一共8个warp tile
        // 一个warp的RC[i][j]存储了一个16*8的结果，这16*8结果在一个warp中的排布见ptx文件
        // 一个warp tile的大小是64*32，这里i固定然后对j做for循环（WARP_TILE_N=4）
        // 所以当j的循环结束时，相当于将一个16*32的结果放到了一个warp的RC中(B warp tile的大小是32*16,被分成了4个8*16，对应了WARP_TILE_N=4)
        // 具体是这样的，一个warp有32个线程，其中只有0，4，8，12，16，20，24，28这8个线程存储了结果，以0号线程为例
        // 0号线程的RC0和RC1总共的大小是4*4*2，可以存32*2个half
        // 所以一个warp中总共8个线程共存放了16*32的结果
        // */
        // /*
        // 上面说一个warp中总共8个线程共存放了16*32的结果，但那是j的for循环执行完毕的情况
        // 这里只以j=0为例进行讲解，当j固定为某个值时，一个warp中所有RC0和RC1中存储了一个16*8的结果
        // 然后由于warp shuffle，16*8的结果被存放在了0，4，8，12，16，20，24，28这8个线程中
        // 具体的，以0号线程为例，RC0[j][0,1,2,3]存放了第0行的8个half结果，RC1[j][0,1,2,3]存放了第8行的8个half结果
        // 对于第4号线程，RC0[j][0,1,2,3]存放了第1行的8个half结果，RC1[j][0,1,2,3]存放了第9行的8个half结果
        // 上面的解释，你直接看大概率看不懂，强烈建议自己画一下这个gemm的layout层次图，然后对着ptx文档的那个矩阵结果在寄存器中的分布图来理解
        // */
        // RC0[j][0] = RC[i][j][0];
        // RC1[j][0] = RC[i][j][1];
        // RC0[j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
        // RC0[j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
        // RC0[j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
        // RC1[j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
        // RC1[j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
        // RC1[j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
        // printf("here 2 2\n");
        RC0[j][0] = RC[i][j][0];
        RC1[j][0] = RC[i][j][1];
        RC0[j][1] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 1);
        RC0[j][2] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 2);
        RC0[j][3] = __shfl_sync((0xffffffff), RC[i][j][0], lane_id + 3);
        RC1[j][1] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 1);
        RC1[j][2] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 2);
        RC1[j][3] = __shfl_sync((0xffffffff), RC[i][j][1], lane_id + 3);
        // printf("here 2 3\n");

        /*
        说一下这里的代码逻辑，主要实现了将矩阵乘的结果RC除以对应的L2范数的乘积
        即计算了这样的公式: RC / (sqrt(RA_NORM) * sqrt(RB_NORM))
        首先这里之所以有lane_id % 4 == 0的判断，是因为一个warp中只有0，4，8，。。。线程持有一个16*8的结果（j=0）
        这里以线程0为例，前面说了，RC0[j][0，1，2，3]持有第0行的8个half，RC1[j][0，1，2，3]持有第8行的8个half（看不懂的去看ptx文档上的那个图）
        然后我们需要将RC0[j][0，1，2，3]和RC1[j][0，1，2，3]中的结果除以范数
        首先对于RC0[j][0]，RC0[j][0]实际上是有两个half，对应了16*8矩阵的[0][0]和[0][1]位置的结果，它需要除以第A矩阵第0行的范数和B矩阵第0/1列的范数开根号的乘积
            对于A矩阵第0行的平方和，其实就是RA_NORM[i][0]；而对于B矩阵的第0列和第1列的平方和，分别分布在第0号线程和第4号线程的RA_NORM[j]中，所以你可以看到这里使用了warp shuffle来从其他线程拿RB_NORM[j]
        然后对于RC0[j][1]，RC0[j][1]实际上是有两个half，对应了16*8矩阵的[0][2]和[0][3]位置的结果，它需要除以第A矩阵第0行的范数和B矩阵第2/3列的范数开根号的乘积
            对于A矩阵第0行的平方和，同样是RA_NORM[i][0]；而对于B矩阵的第2列和第3列的平方和，分别分布在第8号线程和第·1号线程的RA_NORM[j]中，所以你可以看到这里使用了warp shuffle来从其他线程拿RB_NORM[j]
        以此类推
        然后就是关于warp shuffle中的这个4 * (2 * k)和4 * (2 * k + 1)咋来的，这里需要你对照着ptx文档中B矩阵和C矩阵的layout图来理解了
            假设k=0，那么4 * (2 * k)=0，4 * (2 * k + 1)=4，刚好对应了B矩阵的第0列和第1列的平方和所在的线程
            假设k=1，那么4 * (2 * k)=8，4 * (2 * k + 1)=12，刚好对应了B矩阵的第2列和第3列的平方和所在的线程
            以此类推
        然后就计算余弦相似度，然后再将结果存放回RC寄存器中
        对于RC1[j][0,1,2,3]的处理同理，只不过处理RC1[j][0,1,2,3]时，使用的是A矩阵的第8行的平方和，其实就是RA_NORM[i][1]

        还有这里之所以全用float类型，我是看gemini 3pro说，sm80没有针对half的开根号指令，直接对half使用开根号的话，会先将half转为float再开根号，所以这里干脆直接全用float了
        */
      //  printf("here 3\n");
            // rsqrtf(x)方法：计算 1 / sqrt(x), 作用精度为fp32
        float rsqrt_a_norm0 = rsqrtf(RA_NORM[i][0]);
        float rsqrt_a_norm1 = rsqrtf(RA_NORM[i][1]);
#pragma unroll
        for (int k = 0; k < 4; k++) {
            half2 val_h2_0 = reinterpret_cast<half2&>(RC0[j][k]);
            float2 val_f2_0 = __half22float2(val_h2_0);
            float b_norm0_f = __shfl_sync((0xffffffff), RB_NORM[j], 4 * (2 * k));
            float b_norm1_f = __shfl_sync((0xffffffff), RB_NORM[j], 4 * (2 * k + 1));
            float rsqrt_b_norm0 = rsqrtf(b_norm0_f);
            float rsqrt_b_norm1 = rsqrtf(b_norm1_f);
            float res0_f = val_f2_0.x * rsqrt_a_norm0 * rsqrt_b_norm0;
            float res1_f = val_f2_0.y * rsqrt_a_norm0 * rsqrt_b_norm1;
            half2 res_h2 = __float22half2_rn(make_float2(res0_f, res1_f));
            RC0[j][k] = reinterpret_cast<uint32_t&>(res_h2);

            half2 val_h2_1 = reinterpret_cast<half2&>(RC1[j][k]);
            float2 val_f2_1 = __half22float2(val_h2_1);
            float res2_f = val_f2_1.x * rsqrt_a_norm1 * rsqrt_b_norm0;
            float res3_f = val_f2_1.y * rsqrt_a_norm1 * rsqrt_b_norm1;
            half2 res_h2_1 = __float22half2_rn(make_float2(res2_f, res3_f));
            RC1[j][k] = reinterpret_cast<uint32_t&>(res_h2_1);
        }
      }
      // printf("here 4\n");
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
}

// 128x128, mma2x4, warp4x4(64,32,16), stages, block swizzle, dsmem, TN
template <const int K_STAGE = 2, const int BLOCK_SWIZZLE_STRIDE = 2048>
void launch_gemm_kernel_tn(half *a, half *b, half *c, int M, int N, int K) {
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
  const int smem_max_size = ((K_STAGE) * BM * (BK + A_PAD) * sizeof(half) +
                             (K_STAGE) * BN * (BK + B_PAD) * sizeof(half));

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
template <const int K_STAGE = 2, const int BLOCK_SWIZZLE_STRIDE = 2048>
void launch_cosine_kernel_tn(half *a, half *b, half *c, int M, int N, int K) {
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
  const int smem_max_size = ((K_STAGE) * BM * (BK + A_PAD) * sizeof(half) +
                             (K_STAGE) * BN * (BK + B_PAD) * sizeof(half));

  cudaFuncSetAttribute(
      cosine_kernel<MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,
                    WARP_TILE_N, A_PAD, B_PAD, K_STAGE, true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

  const int N_SWIZZLE = (N + BLOCK_SWIZZLE_STRIDE - 1) / BLOCK_SWIZZLE_STRIDE;
  dim3 block(256); // 2 * 4 * 32
  dim3 grid((div_ceil(N, BN) + N_SWIZZLE - 1) / N_SWIZZLE, div_ceil(M, BM),
            N_SWIZZLE);

  cosine_kernel<MMA_M, MMA_N, MMA_K, MMA_TILE_M, MMA_TILE_N, WARP_TILE_M,
                WARP_TILE_N, A_PAD, B_PAD, K_STAGE, true>
      <<<grid, block, smem_max_size>>>(a, b, c, M, N, K);
}

#ifndef BUILD_PYTORCH_EXTENSION
int main(int argc, char *argv[]) {
  const int test_num = 64;
  int M_list[test_num];
  int N_list[test_num];
  int K_list[test_num];

  for (int i = 0; i < test_num; i++) {
    M_list[i] = (i + 1) * 256;
    N_list[i] = (i + 1) * 256;
    K_list[i] = (i + 1) * 256;
  }

  if (argc > 1) M_list[0] = std::stoi(argv[1]);
  if (argc > 2) N_list[0] = std::stoi(argv[2]);
  if (argc > 3) K_list[0] = std::stoi(argv[3]);

  int outer_repeat = 10, inner_repeat = 1, warmup = 1;
  if (argc > 4) warmup = std::stoi(argv[4]);
  if (argc > 5) inner_repeat = std::stoi(argv[5]);

  printf("ALGO = COSINE SIMILARITY TN MMA=2x4 WARP=4x4 STAGES=2 BLOCK SWIZZLE=2048\n");

  // Note: Standard gemm_error_check_tn from utils.h checks C = A * B.
  // Cosine Similarity computes C = (A . B) / (|A| * |B|).
  // So standard GEMM verification will fail. 
  // We skip correctness check here and focus on performance profiling.

  for (int j = 0; j < test_num; j++) {
    int M = M_list[j], N = N_list[j], K = K_list[j];

    double max_sec = 0.0;
    double min_sec = DBL_MAX;
    double total_sec = 0.0;

    for (int k = 0; k < outer_repeat; k++) {
      double this_sec = perf_gemm<half>(launch_cosine_kernel_tn<2, 2048>,
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