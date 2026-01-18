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

// CP_ASYNC definitions
#define CP_ASYNC_COMMIT_GROUP() asm volatile("cp.async.commit_group;\n" ::)
#define CP_ASYNC_WAIT_ALL() asm volatile("cp.async.wait_all;\n" ::)
#define CP_ASYNC_WAIT_GROUP(n) asm volatile("cp.async.wait_group %0;\n" ::"n"(n))
#define CP_ASYNC_CG(dst, src, bytes) \
    asm volatile("cp.async.cg.shared.global.L2::128B [%0], [%1], %2;\n" ::"r"(dst), "l"(src), "n"(bytes))

// LDMATRIX definitions
#define LDMATRIX_X2(R0, R1, addr) \
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n" : "=r"(R0), "=r"(R1) : "r"(addr))
#define LDMATRIX_X4(R0, R1, R2, R3, addr) \
    asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, [%4];\n" : "=r"(R0), "=r"(R1), "=r"(R2), "=r"(R3) : "r"(addr))

// MMA definition
#define HMMA16816(RD0, RD1, RA0, RA1, RA2, RA3, RB0, RB1, RC0, RC1)             \
    asm volatile(                                                               \
        "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0, %1}, {%2, %3, " \
        "%4, %5}, {%6, %7}, {%8, %9};\n"                                        \
        : "=r"(RD0), "=r"(RD1)                                                  \
        : "r"(RA0), "r"(RA1), "r"(RA2), "r"(RA3), "r"(RB0), "r"(RB1), "r"(RC0), \
          "r"(RC1))

HOST_DEVICE_INLINE
int div_ceil(int a, int b) { return (a % b != 0) ? (a / b + 1) : (a / b); }

// =================================================================================
// Kernel Optimized with Loop Unrolling (2x) and Double Issue Load
// =================================================================================
template <const int MMA_M = 16, const int MMA_N = 8, const int MMA_K = 16,
            const int INNER_DIM = 128, const int MMA_TILE_M = 4,
          const int A_PAD = 8, const int B_PAD = 0, const int K_STAGE = 4> // A_PAD=8 for SMEM bank conflict avoidance
__global__ void __launch_bounds__(128)
    cosine_kernel_tn_unrolled(half *A, half *B,
                  half *C, int M,
                  int N, int K)
{
    // XOR Swizzle logic to keep DRAM row locality while avoiding camping
    const int idx_y = blockIdx.y;
    const int swizzle_log = 3; 
    const int swizzle_mask = (1 << swizzle_log) - 1;
    int block_y_swizzled = idx_y;
    if (gridDim.y > swizzle_mask) {
         block_y_swizzled = idx_y ^ ((idx_y >> swizzle_log) & swizzle_mask);
    }
    if (block_y_swizzled >= gridDim.y) block_y_swizzled = idx_y;
    const int by = block_y_swizzled;

    const int NUM_K_TILES = div_ceil(K, MMA_K);
    extern __shared__ half smem[];
    half *s_b = smem;
    half *s_a = s_b + MMA_N * INNER_DIM;
    
    // A_PAD is now 8 halfs (16 bytes), creating a stride of 136 halfs (272 bytes)
    // This avoids SMEM bank conflicts when loading/storing
    const int s_a_stage_offset = MMA_M  * MMA_TILE_M * (INNER_DIM + A_PAD);
    
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int warp_id = tid / WARP_SIZE;
    const int lane_id = tid % WARP_SIZE;

    uint32_t smem_a_base_ptr = __cvta_generic_to_shared(s_a);
    uint32_t smem_b_base_ptr = __cvta_generic_to_shared(s_b);

    int load_smem_a_m = tid / 2; 
    int load_smem_a_k = (tid % 2) * 8;
    int load_smem_b_n = tid / 16;
    int load_smem_b_k = (tid % 16) * 8;

    int load_gmem_a_m = by * MMA_M * MMA_TILE_M + load_smem_a_m;

    if (load_gmem_a_m >= M || load_smem_b_n >= N) return;

    // Load B (Tiny, reused)
    int load_gmem_b_addr = load_smem_b_n * INNER_DIM + load_smem_b_k;
    uint32_t load_smem_b_addr = smem_b_base_ptr + load_gmem_b_addr * sizeof(half);
    CP_ASYNC_CG(load_smem_b_addr, &B[load_gmem_b_addr], 16);

    // Prologue: Load K_STAGE - 1 tiles
    // K_STAGE should be even usually for perfect unroll, but we handle it generally
    #pragma unroll
    for (int k = 0; k < (K_STAGE - 1); ++k) {     
        int load_gmem_a_k = k * MMA_K + load_smem_a_k;
        int load_gmem_a_addr = load_gmem_a_m * INNER_DIM + load_gmem_a_k;
        uint32_t load_smem_a_addr = (smem_a_base_ptr + 
            (k * s_a_stage_offset + load_smem_a_m * MMA_K + load_smem_a_k) * sizeof(half));
        CP_ASYNC_CG(load_smem_a_addr, &A[load_gmem_a_addr], 16);
        CP_ASYNC_COMMIT_GROUP(); 
    }
    
    // Wait until we have at least 2 stages ready to compute (Double buffering logic)
    // Actually standard wait is K_STAGE - 2.
    CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
    __syncthreads();

    uint32_t RA[4];
    uint32_t RB[2];
    uint32_t RC[2] = {0, 0};
    
    half2 RA_NORM_H2[2] = {__float2half2_rn(0.0f), __float2half2_rn(0.0f)};
    half2 RB_NORM_H2 = __float2half2_rn(0.0f);

    int b_stage = 0;
    
    // =========================================================
    // Unrolled Main Loop (Step = 2)
    // =========================================================
    #pragma unroll
    for(int k = (K_STAGE - 1); k < NUM_K_TILES; k += 2){
        
        // --- LOAD PHASE: Issue 2 loads back-to-back ---
        // Load Tile k
        {
            int gmem_load_stage = k % K_STAGE;
            int load_a_gmem_k = k * MMA_K + load_smem_a_k;
            int load_a_gmem_addr = load_gmem_a_m * INNER_DIM + load_a_gmem_k;
            uint32_t load_smem_a_addr = (smem_a_base_ptr + 
                (gmem_load_stage * s_a_stage_offset + load_smem_a_m * MMA_K + load_smem_a_k) * sizeof(half));
            // Check bounds for K (only necessary if K is not multiple of 32)
            if (k < NUM_K_TILES) 
                CP_ASYNC_CG(load_smem_a_addr, &A[load_a_gmem_addr], 16);
        }
        
        // Load Tile k + 1 (Consecutive memory access!)
        {
            int k_next = k + 1;
            int gmem_load_stage = k_next % K_STAGE;
            int load_a_gmem_k = k_next * MMA_K + load_smem_a_k;
            int load_a_gmem_addr = load_gmem_a_m * INNER_DIM + load_a_gmem_k;
            uint32_t load_smem_a_addr = (smem_a_base_ptr + 
                (gmem_load_stage * s_a_stage_offset + load_smem_a_m * MMA_K + load_smem_a_k) * sizeof(half));
            
            // This second load hits the next 16 bytes. 
            // Combined with above, we load 32 contiguous bytes per thread per iteration.
            if (k_next < NUM_K_TILES) 
                CP_ASYNC_CG(load_smem_a_addr, &A[load_a_gmem_addr], 16);
        }
        
        CP_ASYNC_COMMIT_GROUP(); // Commit both loads as one group

        // --- COMPUTE PHASE: Process 2 tiles ---
        // We need to carefully handle the circular buffer indices for calculation
        // The loads we just issued are for future stages. 
        // We compute on the "oldest" data in smem.
        
        // Compute Tile 1
        int k_curr = k - (K_STAGE - 1);
        int compute_stage = k_curr % K_STAGE;
        
        // Load B from SMEM
        int lane_smem_b_n = lane_id % 8;
        int lane_smem_b_k = lane_id / 8 * 8;
        uint32_t lane_smem_b_addr = smem_b_base_ptr + 
            (lane_smem_b_n * INNER_DIM + (b_stage * 16 + lane_smem_b_k)) * sizeof(half);
        LDMATRIX_X2(RB[0], RB[1], lane_smem_b_addr);
        b_stage++;

        // Load A from SMEM
        int lane_smem_a_m = warp_id * MMA_M + lane_id % 16;
        int lane_smem_a_k = lane_id / 16 * 8;
        uint32_t lane_smem_a_addr = smem_a_base_ptr + 
            (compute_stage * s_a_stage_offset + lane_smem_a_m * MMA_K + lane_smem_a_k) * sizeof(half);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], lane_smem_a_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
        
        // A Norm Accum
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
        // B Norm Accum
        {
            half2 val_h2 = reinterpret_cast<half2&>(RB[0]);
            RB_NORM_H2 = __hfma2(val_h2, val_h2, RB_NORM_H2);
            val_h2 = reinterpret_cast<half2&>(RB[1]);
            RB_NORM_H2 = __hfma2(val_h2, val_h2, RB_NORM_H2);
        }

        // Compute Tile 2 (The next one)
        // We only wait here because we consumed one stage above.
        // But since we commit groups together, the wait logic is tricky.
        // Actually, with commit_group() called once per 2 loads, 1 group = 2 loads.
        // So we wait for K_STAGE - 2 groups.
        
        k_curr = k - (K_STAGE - 1) + 1;
        compute_stage = k_curr % K_STAGE;

        // Repeat Logic for second tile
        lane_smem_b_addr = smem_b_base_ptr + 
            (lane_smem_b_n * INNER_DIM + (b_stage * 16 + lane_smem_b_k)) * sizeof(half);
        LDMATRIX_X2(RB[0], RB[1], lane_smem_b_addr);
        b_stage++;

        lane_smem_a_addr = smem_a_base_ptr + 
            (compute_stage * s_a_stage_offset + lane_smem_a_m * MMA_K + lane_smem_a_k) * sizeof(half);
        LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], lane_smem_a_addr);

        HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);
        
        // A Norm Accum
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
        // B Norm Accum
        {
            half2 val_h2 = reinterpret_cast<half2&>(RB[0]);
            RB_NORM_H2 = __hfma2(val_h2, val_h2, RB_NORM_H2);
            val_h2 = reinterpret_cast<half2&>(RB[1]);
            RB_NORM_H2 = __hfma2(val_h2, val_h2, RB_NORM_H2);
        }
        
        // Wait for space for next iteration's double load
        CP_ASYNC_WAIT_GROUP(K_STAGE - 2);
        __syncthreads();
    }
    
    // Drain Pipeline (Epilogue)
    // The main loop handles "Loading Future" and "Computing Past".
    // We stop loading when k < NUM_K_TILES.
    // We still have K_STAGE-1 tiles loaded but not computed.
    // Note: Due to step=2, we might have 1 leftover compute or slightly different drain logic.
    // For simplicity, let's just drain whatever is left.
    // We need to calculate how many compute steps are remaining.
    // Total compute steps = NUM_K_TILES.
    // Steps done inside loop = (NUM_K_TILES - (K_STAGE-1)) / 2 * 2 ... roughly.
    
    // A simpler way to handle drain in unrolled loop is to just re-run a standard loop
    // for the remaining stages.
    
    CP_ASYNC_WAIT_GROUP(0);
    __syncthreads();

    // Re-calculate where we left off in computation
    // The loop variable 'k' represented the LOAD index.
    // The COMPUTE index was lagging by (K_STAGE - 1).
    // Let's iterate the compute index 'c' until end.
    
    // How many tiles did we compute inside?
    // We stopped when load index k >= NUM_K_TILES.
    // Start computing from the first non-computed tile.
    int k_load_start = (K_STAGE - 1);
    // Integer math to find the first k that failed the loop condition:
    // k = k_load_start + n * 2. 
    int tiles_loaded_in_loop = ((NUM_K_TILES - k_load_start + 1) / 2) * 2; // approximation
    // To be safe and precise, we just loop the Compute from [Total_Computed] to [NUM_K_TILES]
    
    // Instead of complex math, let's just count how many we computed:
    // compute_idx = k - (K_STAGE - 1)
    
    // We can just loop from the first non-computed tile to the end.
    // The first tile index is 0.
    // We process 2 tiles per loop iteration.
    // Number of full loop iterations = (NUM_K_TILES - (K_STAGE-1) + 1) / 2
    int loop_iters = (NUM_K_TILES - (K_STAGE - 1) + 1) / 2;
    int computed_so_far = loop_iters * 2;
    
    for(int c = computed_so_far; c < NUM_K_TILES; c++) {
         int compute_stage = c % K_STAGE;
         
         // Reuse standard compute logic
         int lane_smem_b_n = lane_id % 8;
         int lane_smem_b_k = lane_id / 8 * 8;
         uint32_t lane_smem_b_addr = smem_b_base_ptr + 
             (lane_smem_b_n * INNER_DIM + (b_stage * 16 + lane_smem_b_k)) * sizeof(half);
         LDMATRIX_X2(RB[0], RB[1], lane_smem_b_addr);
         b_stage++;

         int lane_smem_a_m = warp_id * MMA_M + lane_id % 16;
         int lane_smem_a_k = lane_id / 16 * 8;
         uint32_t lane_smem_a_addr = smem_a_base_ptr + 
             (compute_stage * s_a_stage_offset + lane_smem_a_m * MMA_K + lane_smem_a_k) * sizeof(half);
         LDMATRIX_X4(RA[0], RA[1], RA[2], RA[3], lane_smem_a_addr);

         HMMA16816(RC[0], RC[1], RA[0], RA[1], RA[2], RA[3], RB[0], RB[1], RC[0], RC[1]);

         // Norms
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

    // Epilogue reduction and store (Same as before)
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

    if(lane_id % 4 == 0){
        int store_smem_c_m = warp_id * MMA_M + lane_id / 4;
        int store_smem_c_n = 0;
        int store_gmem_c_m = by * MMA_M * MMA_TILE_M + store_smem_c_m;
        int store_gmem_c_n = store_smem_c_n;
        int store_gmem_c_addr0 = store_gmem_c_m * MMA_N + store_gmem_c_n;
        int store_gmem_c_addr1 = (store_gmem_c_m + 8) * MMA_N + store_gmem_c_n;
        LDST128BITS(C[store_gmem_c_addr0]) = LDST128BITS(RC0[0]);
        LDST128BITS(C[store_gmem_c_addr1]) = LDST128BITS(RC1[0]);
    }
}

template <const int K_STAGE = 4> // Use Stage 4 to smooth out double issue
void launch_cosine_kernel_tn(half *a, half *b, half *c, int M, int N, int K)
{
    constexpr int MMA_M = 16;
    constexpr int MMA_N = 8;
    constexpr int MMA_K = 16;
    constexpr int INNER_DIM = 128;
    constexpr int MMA_TILE_M = 4;
    constexpr int A_PAD = 0; // IMPORTANT
    constexpr int B_PAD = 0;
    
    // Size calculation needs to account for PAD
    const int smem_max_size = MMA_N * INNER_DIM * sizeof(half) + 
                            K_STAGE * MMA_M * MMA_TILE_M * (INNER_DIM + A_PAD) * sizeof(half);

    cudaFuncSetAttribute(
        cosine_kernel_tn_unrolled<MMA_M, MMA_N, MMA_K, INNER_DIM, MMA_TILE_M, A_PAD, B_PAD, K_STAGE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    constexpr int BM = MMA_M * MMA_TILE_M;
    dim3 block(128); 
    dim3 grid(div_ceil(N, MMA_N), div_ceil(M, BM));

    static bool is_configured = false;
    if (!is_configured) {
        cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 4096); 
        is_configured = true;
    }
    
    cudaStream_t stream = 0; 
    cudaStreamAttrValue stream_attribute;
    stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(b);
    stream_attribute.accessPolicyWindow.num_bytes = 2048; 
    stream_attribute.accessPolicyWindow.hitRatio  = 1.0;
    stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

    // 这个代码加了block swizzle, 都是ai加上的
    cosine_kernel_tn_unrolled<MMA_M, MMA_N, MMA_K, INNER_DIM, MMA_TILE_M, A_PAD, B_PAD, K_STAGE>
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
