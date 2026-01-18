// kernel 1

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cute/tensor.hpp>

using namespace cute;

// =================================================================================================
// Kernel Implementation
// =================================================================================================

template <typename T>
__global__ void fused_cosine_similarity_kernel_inline(
    void* __restrict__ C_ptr, 
    const void* __restrict__ A_ptr, 
    const void* __restrict__ B_ptr,
    int M, int N, int K) 
{
    // Cast void pointers to typed pointers for arithmetic
    T* C_data = reinterpret_cast<T*>(C_ptr);
    const T* A_data = reinterpret_cast<const T*>(A_ptr);
    const T* B_data = reinterpret_cast<const T*>(B_ptr);

    // -------------------------------------------------------------------------------
    // 0. Configuration
    // -------------------------------------------------------------------------------
    using bM = Int<128>;
    using bN = Int<128>;
    using bK = Int<32>;

    // SM80 (Ampere) 16x8x16 FP16 Tensor Core instruction
    using mma_op = SM80_16x8x16_F16F16F16F16_TN;
    using mma_traits = MMA_Traits<mma_op>;
    using mma_atom = MMA_Atom<mma_traits>;
    
    // TiledMMA: 128x128x32
    using TiledMMA = decltype(make_tiled_mma(
        mma_atom{}, 
        make_layout(make_shape(Int<8>{}, Int<16>{}, Int<2>{}))
    ));

    // Copy Operations
    using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
    using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;

    using TiledCopyA = decltype(make_tiled_copy(
        g2s_copy_atom{},
        make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})) 
        , make_layout(make_shape(Int<1>{}, Int<8>{})) 
    ));
    using TiledCopyB = TiledCopyA;

    // -------------------------------------------------------------------------------
    // 1. Global Memory Tensors
    // -------------------------------------------------------------------------------
    Tensor A = make_tensor(make_gmem_ptr(A_data), make_shape(M, K), make_stride(K, Int<1>{}));
    Tensor B = make_tensor(make_gmem_ptr(B_data), make_shape(N, K), make_stride(K, Int<1>{}));
    // Tensor C is used for tiling logic, but we do manual writeback
    Tensor C = make_tensor(make_gmem_ptr(C_data), make_shape(M, N), make_stride(N, Int<1>{}));

    // -------------------------------------------------------------------------------
    // 2. Shared Memory
    // -------------------------------------------------------------------------------
    auto smem_layout_A = composition(Swizzle<3, 3, 3>{},
                                     make_layout(make_shape(bM{}, bK{}), make_stride(bK{}, Int<1>{})));
    auto smem_layout_B = composition(Swizzle<3, 3, 3>{},
                                     make_layout(make_shape(bN{}, bK{}), make_stride(bK{}, Int<1>{})));

    extern __shared__ char smem_raw[];
    T* smem_A_ptr = (T*)smem_raw;
    T* smem_B_ptr = smem_A_ptr + cosize(smem_layout_A);
    float* smem_norm_A = (float*)(smem_B_ptr + cosize(smem_layout_B));
    float* smem_norm_B = smem_norm_A + size(bM{});

    Tensor sA = make_tensor(make_smem_ptr(smem_A_ptr), smem_layout_A);
    Tensor sB = make_tensor(make_smem_ptr(smem_B_ptr), smem_layout_B);

    for (int i = threadIdx.x; i < size(bM{}); i += blockDim.x) smem_norm_A[i] = 0.0f;
    for (int i = threadIdx.x; i < size(bN{}); i += blockDim.x) smem_norm_B[i] = 0.0f;
    __syncthreads();

    // -------------------------------------------------------------------------------
    // 3. Partitioning
    // -------------------------------------------------------------------------------
    int pid_m = blockIdx.y;
    int pid_n = blockIdx.x;

    Tensor gA = local_tile(A, make_shape(bM{}, bK{}), make_coord(pid_m, _)); 
    Tensor gB = local_tile(B, make_shape(bN{}, bK{}), make_coord(pid_n, _)); 
    
    TiledMMA tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
    
    // Accumulator
    Tensor gC = local_tile(C, make_shape(bM{}, bN{}), make_coord(pid_m, pid_n));
    Tensor tCrC = thr_mma.partition_fragment_C(gC); 
    clear(tCrC);

    // Copy Partitions
    TiledCopyA tiled_copy_a;
    auto thr_copy_a = tiled_copy_a.get_thread_slice(threadIdx.x);
    Tensor tAgA = thr_copy_a.partition_S(gA); 
    Tensor tAsA = thr_copy_a.partition_D(sA);

    TiledCopyB tiled_copy_b;
    auto thr_copy_b = tiled_copy_b.get_thread_slice(threadIdx.x);
    Tensor tBgB = thr_copy_b.partition_S(gB); 
    Tensor tBsB = thr_copy_b.partition_D(sB);

    // MMA Partitions
    Tensor tCsA = thr_mma.partition_A(sA); 
    Tensor tCsB = thr_mma.partition_B(sB);

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    // Identity Coordinates
    auto idA = make_identity_tensor(make_shape(bM{}, bK{}));
    auto idB = make_identity_tensor(make_shape(bN{}, bK{}));
    Tensor tCcA = thr_mma.partition_A(idA); 
    Tensor tCcB = thr_mma.partition_B(idB);

    // Norm Accumulators
    Tensor tNormA = make_tensor<float>(make_shape(size(tCrA)));
    Tensor tNormB = make_tensor<float>(make_shape(size(tCrB)));
    clear(tNormA);
    clear(tNormB);

    // -------------------------------------------------------------------------------
    // 4. Main Loop
    // -------------------------------------------------------------------------------
    int k_tiles = size<2>(gA);
    
    // Prologue
    copy(tiled_copy_a, tAgA(_,_,_,0), tAsA);
    copy(tiled_copy_b, tBgB(_,_,_,0), tBsB);
    cp_async_fence();
    cp_async_wait<0>();
    __syncthreads();

    for (int k = 0; k < k_tiles; ++k) {
        copy(tCsA, tCrA);
        copy(tCsB, tCrB);

        if (k < k_tiles - 1) {
            copy(tiled_copy_a, tAgA(_,_,_,k+1), tAsA);
            copy(tiled_copy_b, tBgB(_,_,_,k+1), tBsB);
            cp_async_fence();
        }

        gemm(tiled_mma, tCrC, tCrA, tCrB, tCrC);

        CUTE_UNROLL
        for (int i = 0; i < size(tCrA); ++i) {
            float val = static_cast<float>(tCrA(i));
            tNormA(i) += val * val;
        }
        CUTE_UNROLL
        for (int i = 0; i < size(tCrB); ++i) {
            float val = static_cast<float>(tCrB(i));
            tNormB(i) += val * val;
        }

        if (k < k_tiles - 1) {
            cp_async_wait<0>();
            __syncthreads();
        }
    }

    // -------------------------------------------------------------------------------
    // 5. Reduction & Epilogue
    // -------------------------------------------------------------------------------
    CUTE_UNROLL
    for (int i = 0; i < size(tNormA); ++i) {
        int row = get<0>(tCcA(i)); 
        atomicAdd(&smem_norm_A[row], tNormA(i));
    }
    CUTE_UNROLL
    for (int i = 0; i < size(tNormB); ++i) {
        int col = get<0>(tCcB(i)); 
        atomicAdd(&smem_norm_B[col], tNormB(i));
    }
    __syncthreads();

    auto idC = make_identity_tensor(make_shape(bM{}, bN{}));
    Tensor tCcC = thr_mma.partition_C(idC);

    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        int r = get<0>(tCcC(i));
        int c = get<1>(tCcC(i));
        
        float nA = sqrtf(smem_norm_A[r]);
        float nB = sqrtf(smem_norm_B[c]);
        float div = nA * nB + 1e-6f;
        
        float val = static_cast<float>(tCrC(i));
        tCrC(i) = static_cast<T>(val / div);
    }

    // Manual Writeback
    CUTE_UNROLL
    for (int i = 0; i < size(tCrC); ++i) {
        int r_local = get<0>(tCcC(i));
        int c_local = get<1>(tCcC(i));
        
        int r_global = pid_m * 128 + r_local;
        int c_global = pid_n * 128 + c_local;

        if (r_global < M && c_global < N) {
            // Now using the typed pointer C_data
            C_data[r_global * N + c_global] = tCrC(i);
        }
    }
}

// =================================================================================================
// Launcher
// =================================================================================================

void cosine_similarity_cute_launch(torch::Tensor A, torch::Tensor B, torch::Tensor C) {
    int M = A.size(0);
    int K = A.size(1);
    int N = B.size(0);

    using T = cutlass::half_t;
    
    int smem_size = 24 * 1024; 

    dim3 grid((N + 127) / 128, (M + 127) / 128);
    dim3 block(128); 

    fused_cosine_similarity_kernel_inline<T><<<grid, block, smem_size>>>(
        C.data_ptr(),
        A.data_ptr(),
        B.data_ptr(),
        M, N, K
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cosine_similarity_cute", &cosine_similarity_cute_launch, "CuTe Fused Cosine Similarity");
}




// // kernel2


// // // 精度无损性能劣化

// #include <torch/extension.h>
// #include <cuda.h>
// #include <cuda_runtime.h>
// #include <cute/tensor.hpp>

// // 定义半精度浮点数类型简写
// using half_t = cute::half_t;

// // ========================================================================
// // 1. Helper Kernels: 计算行/列的平方和 (Squared Norms)
// // ========================================================================

// template <typename T>
// __global__ void row_sq_sum_kernel(const T* __restrict__ matrix, float* __restrict__ norms, int rows, int cols) {
//     int tid = threadIdx.x;
//     int row_idx = blockIdx.x;

//     if (row_idx >= rows) return;

//     float sum = 0.0f;
//     for (int k = tid; k < cols; k += blockDim.x) {
//         float val = static_cast<float>(matrix[row_idx * cols + k]);
//         sum += val * val;
//     }

//     // Warp Reduction
//     for (int offset = 16; offset > 0; offset /= 2) {
//         sum += __shfl_down_sync(0xffffffff, sum, offset);
//     }

//     // Block Reduction
//     static __shared__ float shared_sum[32]; 
//     int lane = tid % 32;
//     int warp = tid / 32;
    
//     if (lane == 0) {
//         shared_sum[warp] = sum;
//     }
//     __syncthreads();

//     if (warp == 0) {
//         sum = (lane < (blockDim.x / 32)) ? shared_sum[lane] : 0.0f;
//         for (int offset = 16; offset > 0; offset /= 2) {
//             sum += __shfl_down_sync(0xffffffff, sum, offset);
//         }
//         if (tid == 0) {
//             norms[row_idx] = sum;
//         }
//     }
// }


// // ========================================================================
// // 2. CuTe GEMM Implementation with Fused Cosine Normalization
// // ========================================================================

// template <typename Config>
// __global__ void 
// gemm_cosine_fused(
//     void *Dptr, const void *Aptr, const void *Bptr, 
//     const float* NormA_ptr, const float* NormB_ptr,
//     int m, int n, int k) 
// {
//     using namespace cute;
//     using T = typename Config::T;
//     using SmemLayoutA = typename Config::SmemLayoutA;
//     using SmemLayoutB = typename Config::SmemLayoutB;
//     using SmemLayoutC = typename Config::SmemLayoutC;
//     using TiledMMA = typename Config::MMA;

//     using S2RCopyAtomA = typename Config::S2RCopyAtomA;
//     using S2RCopyAtomB = typename Config::S2RCopyAtomB;
    
//     constexpr int kTileM = Config::kTileM;
//     constexpr int kTileN = Config::kTileN;
//     constexpr int kTileK = Config::kTileK;
//     constexpr int kStage = Config::kStage;

//     extern __shared__ char shm_raw[];
//     T *Ashm = (T*)shm_raw;
//     T *Bshm = (T*)(shm_raw + cute::cosize(SmemLayoutA{}) * sizeof(T));
    
//     // Norm 缓存区放在 Shared Memory 末尾
//     size_t gemm_shm_size = cute::max(
//         cute::cosize(SmemLayoutA{}) + cute::cosize(SmemLayoutB{}),
//         cute::cosize(SmemLayoutC{})
//     ) * sizeof(T);
    
//     float* sNormA = (float*)(shm_raw + gemm_shm_size);
//     float* sNormB = sNormA + kTileM;

//     int idx = threadIdx.x;
//     int ix = blockIdx.x; 
//     int iy = blockIdx.y; 

//     // ------------------------------------------------------------
//     // Phase 0: Load Norms into Shared Memory
//     // ------------------------------------------------------------
//     if (idx < kTileM) {
//         int global_row = iy * kTileM + idx;
//         sNormA[idx] = (global_row < m) ? NormA_ptr[global_row] : 1.0f;
//     }
//     if (idx < kTileN) {
//         int global_col = ix * kTileN + idx;
//         sNormB[idx] = (global_col < n) ? NormB_ptr[global_col] : 1.0f;
//     }

//     // ------------------------------------------------------------
//     // Phase 1: GEMM Pipeline
//     // ------------------------------------------------------------
//     Tensor A = make_tensor(make_gmem_ptr((T *)Aptr), make_shape(m, k), make_stride(k, Int<1>{}));
//     Tensor B = make_tensor(make_gmem_ptr((T *)Bptr), make_shape(n, k), make_stride(k, Int<1>{}));
//     Tensor D = make_tensor(make_gmem_ptr((T *)Dptr), make_shape(m, n), make_stride(n, Int<1>{}));

//     Tensor gA = local_tile(A, make_tile(Int<kTileM>{}, Int<kTileK>{}), make_coord(iy, _));
//     Tensor gB = local_tile(B, make_tile(Int<kTileN>{}, Int<kTileK>{}), make_coord(ix, _));
//     Tensor gD = local_tile(D, make_tile(Int<kTileM>{}, Int<kTileN>{}), make_coord(iy, ix));

//     auto sA = make_tensor(make_smem_ptr(Ashm), SmemLayoutA{});
//     auto sB = make_tensor(make_smem_ptr(Bshm), SmemLayoutB{});

//     TiledMMA tiled_mma;
//     auto thr_mma = tiled_mma.get_slice(idx);
//     auto tCrA = thr_mma.partition_fragment_A(gA(_, _, 0));
//     auto tCrB = thr_mma.partition_fragment_B(gB(_, _, 0));
//     auto tCrD = thr_mma.partition_fragment_C(gD);

//     clear(tCrD);

//     // Copy Definitions
//     using G2SCopyA = typename Config::G2SCopyA;
//     using G2SCopyB = typename Config::G2SCopyB;
    
//     G2SCopyA g2s_tiled_copy_a;
//     auto g2s_thr_copy_a = g2s_tiled_copy_a.get_slice(idx);
//     auto tAgA_copy = g2s_thr_copy_a.partition_S(gA);
//     auto tAsA_copy = g2s_thr_copy_a.partition_D(sA);

//     G2SCopyB g2s_tiled_copy_b;
//     auto g2s_thr_copy_b = g2s_tiled_copy_b.get_slice(idx);
//     auto tBgB_copy = g2s_thr_copy_b.partition_S(gB);
//     auto tBsB_copy = g2s_thr_copy_b.partition_D(sB);

//     auto s2r_tiled_copy_a = make_tiled_copy_A(S2RCopyAtomA{}, tiled_mma);
//     auto s2r_thr_copy_a = s2r_tiled_copy_a.get_slice(idx);
//     auto tAsA = s2r_thr_copy_a.partition_S(sA);
//     auto tCrA_view = s2r_thr_copy_a.retile_D(tCrA);

//     auto s2r_tiled_copy_b = make_tiled_copy_B(S2RCopyAtomB{}, tiled_mma);
//     auto s2r_thr_copy_b = s2r_tiled_copy_b.get_slice(idx);
//     auto tBsB = s2r_thr_copy_b.partition_S(sB);
//     auto tCrB_view = s2r_thr_copy_b.retile_D(tCrB);

//     int itile_to_read = 0;
//     int ismem_read = 0;
//     int ismem_write = 0;

//     // Prologue
//     #pragma unroll
//     for (int istage = 0; istage < kStage - 1; ++istage) {
//         cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, istage), tAsA_copy(_, _, _, istage));
//         cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, istage), tBsB_copy(_, _, _, istage));
//         cp_async_fence();
//         ++itile_to_read;
//         ++ismem_write;
//     }

//     cp_async_wait<kStage - 2>();
//     __syncthreads();

//     int ik = 0;
//     cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik, ismem_read), tCrA_view(_, _, ik));
//     cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik, ismem_read), tCrB_view(_, _, ik));

//     // Main Loop
//     int ntile = k / kTileK;
//     #pragma unroll 1
//     for (int itile = 0; itile < ntile; ++itile) {
//         int nk = size<2>(tCrA);
//         #pragma unroll
//         for (int ik = 0; ik < nk; ++ik) {
//             int ik_next = (ik + 1) % nk;
//             if (ik == nk - 1) {
//                 cp_async_wait<kStage - 2>();
//                 __syncthreads();
//                 ismem_read = (ismem_read + 1) % kStage;
//             }
//             cute::copy(s2r_tiled_copy_a, tAsA(_, _, ik_next, ismem_read), tCrA_view(_, _, ik_next));
//             cute::copy(s2r_tiled_copy_b, tBsB(_, _, ik_next, ismem_read), tCrB_view(_, _, ik_next));

//             if (ik == 0) {
//                 if (itile_to_read < ntile) {
//                     cute::copy(g2s_tiled_copy_a, tAgA_copy(_, _, _, itile_to_read), tAsA_copy(_, _, _, ismem_write));
//                     cute::copy(g2s_tiled_copy_b, tBgB_copy(_, _, _, itile_to_read), tBsB_copy(_, _, _, ismem_write));
//                     ++itile_to_read;
//                     ismem_write = (ismem_write + 1) % kStage;
//                 }
//                 cp_async_fence();
//             }
//             cute::gemm(tiled_mma, tCrD, tCrA(_, _, ik), tCrB(_, _, ik), tCrD);
//         }
//     }

//     // ------------------------------------------------------------
//     // Phase 2: Epilogue (Reg -> Smem)
//     // ------------------------------------------------------------
//     __syncthreads();
    
//     // Use simple layout for C in Smem (No Swizzle to simplify coordinates)
//     auto sC = make_tensor(sA(_, _, ismem_read).data(), SmemLayoutC{});

//     using R2SCopyAtomC = typename Config::R2SCopyAtomC;
//     auto r2s_tiled_copy_c = make_tiled_copy_C(R2SCopyAtomC{}, tiled_mma);
//     auto r2s_thr_copy_c = r2s_tiled_copy_c.get_slice(idx);
//     auto tCrC_r2s = r2s_thr_copy_c.retile_S(tCrD);   
//     auto tCsC_r2s = r2s_thr_copy_c.partition_D(sC);  

//     cute::copy(r2s_tiled_copy_c, tCrC_r2s, tCsC_r2s);
//     __syncthreads();

//     // ------------------------------------------------------------
//     // Phase 3: Epilogue (Smem -> Global with Fused Normalization)
//     // ------------------------------------------------------------
//     // 定义 S2G Copy (使用普通 Auto Copy)
//     // 为了方便，我们使用 TiledCopy，这样可以保证线程分块均匀
//     // 这里复用 TiledMMA 的线程分布，但操作的是简单的 Tile
//     auto s2g_tiled_copy_c = make_tiled_copy_C(Copy_Atom<AutoVectorizingCopy, T>{}, tiled_mma);
//     auto s2g_thr_copy_c = s2g_tiled_copy_c.get_slice(idx);

//     // 源: Shared Memory
//     auto tCsC_s2g = s2g_thr_copy_c.partition_S(sC);
//     // 目标: Global Memory
//     auto tCgC_s2g = s2g_thr_copy_c.partition_D(gD);

//     // 关键：为 Shared Memory 创建一个 Identity Tensor，用于获取坐标
//     // sC 是 (kTileM, kTileN)
//     auto cS = make_identity_tensor(make_shape(Int<kTileM>{}, Int<kTileN>{}));
//     // 对其进行相同的划分，得到当前线程处理的坐标
//     auto tCcS_s2g = s2g_thr_copy_c.partition_S(cS);

//     const float eps = 1e-6f;

//     // 手动展开 Copy Loop 并插入归一化
//     // tCsC_s2g 和 tCgC_s2g 通常是 rank-3 tensor: (CopyOp, M_rest, N_rest)
//     // 但这里的 layout 可能比较复杂，我们直接迭代 size() 即可
//     // CuTe 的 tensor 迭代器保证相同索引对应相同的逻辑位置
    
//     #pragma unroll
//     for (int i = 0; i < size(tCsC_s2g); ++i) {
//         // 1. 读取数值 (Smem)
//         T val = tCsC_s2g(i);
        
//         // 2. 读取坐标 (Smem frame)
//         auto coord = tCcS_s2g(i);
//         int local_m = get<0>(coord);
//         int local_n = get<1>(coord);
        
//         // 3. 计算 Global 坐标
//         int global_m = iy * kTileM + local_m;
//         int global_n = ix * kTileN + local_n;

//         // 4. 读取 Norm
//         // 边界检查：如果 padding 区域，norm 设为 1 (不影响结果，因为 C 通常也是 0 或无效)
//         float n_a = (global_m < m) ? sNormA[local_m] : 1.0f;
//         float n_b = (global_n < n) ? sNormB[local_n] : 1.0f;

//         // 5. 归一化
//         float f_val = static_cast<float>(val);
//         float scale = rsqrtf(n_a + eps) * rsqrtf(n_b + eps);
        
//         // 6. 写入 Global
//         if (global_m < m && global_n < n) { // 基本的边界检查
//             tCgC_s2g(i) = static_cast<T>(f_val * scale);
//         }
//     }
// }

// // ========================================================================
// // 3. Configuration Struct
// // ========================================================================

// namespace config {
// using namespace cute;

// template <typename T_, int kTileM_ = 128, int kTileN_ = 128, int kTileK_ = 32, int kStage_ = 3>
// struct GemmConfig {
//     using T = T_;
//     static constexpr int kTileM = kTileM_;
//     static constexpr int kTileN = kTileN_;
//     static constexpr int kTileK = kTileK_;
//     static constexpr int kStage = kStage_;

//     using SmemLayoutAtom = decltype(composition(
//         Swizzle<3, 3, 3>{},
//         make_layout(make_shape(Int<8>{}, Int<kTileK>{}),
//                     make_stride(Int<kTileK>{}, Int<1>{}))));
    
//     using SmemLayoutA = decltype(tile_to_shape(SmemLayoutAtom{},
//                     make_shape(Int<kTileM>{}, Int<kTileK>{}, Int<kStage>{})));
//     using SmemLayoutB = decltype(tile_to_shape(SmemLayoutAtom{},
//                     make_shape(Int<kTileN>{}, Int<kTileK>{}, Int<kStage>{})));

//     using mma_op = SM80_16x8x16_F16F16F16F16_TN; 
//     using mma_traits = MMA_Traits<mma_op>;
//     using mma_atom = MMA_Atom<mma_traits>;
    
//     static constexpr int kMmaEURepeatM = 2;
//     static constexpr int kMmaEURepeatN = 2;
//     static constexpr int kMmaEURepeatK = 1;
    
//     using MMA = decltype(make_tiled_mma(mma_atom{}, 
//                         make_layout(make_shape(Int<kMmaEURepeatM>{}, Int<kMmaEURepeatN>{}, Int<kMmaEURepeatK>{})), 
//                         Tile<Int<16 * kMmaEURepeatM>, Int<16 * 2 * kMmaEURepeatN>, Int<16 * kMmaEURepeatK>>{}));

//     using g2s_copy_op = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
//     using g2s_copy_traits = Copy_Traits<g2s_copy_op>;
//     using g2s_copy_atom = Copy_Atom<g2s_copy_traits, T>;
    
//     using G2SCopyA = decltype(make_tiled_copy(g2s_copy_atom{},
//                             make_layout(make_shape(Int<32>{}, Int<4>{}), make_stride(Int<4>{}, Int<1>{})),
//                             make_layout(make_shape(Int<1>{}, Int<8>{}))));
//     using G2SCopyB = G2SCopyA;

//     using s2r_copy_op = SM75_U32x4_LDSM_N;
//     using s2r_copy_traits = Copy_Traits<s2r_copy_op>;
//     using s2r_copy_atom = Copy_Atom<s2r_copy_traits, T>;
//     using S2RCopyAtomA = s2r_copy_atom;
//     using S2RCopyAtomB = s2r_copy_atom;

//     // [Change] SmemLayoutC: Simple RowMajor layout, No Swizzle
//     // Layout: (M, N) -> (N, 1)
//     using SmemLayoutC = Layout<Shape<Int<kTileM>, Int<kTileN>>, Stride<Int<kTileN>, Int<1>>>;

//     using R2SCopyAtomC = Copy_Atom<UniversalCopy<int>, T>;
// };
// }

// // ========================================================================
// // 4. C++ Wrapper
// // ========================================================================

// void cosine_similarity_cute(at::Tensor a, at::Tensor b, at::Tensor c) {
//     TORCH_CHECK(a.is_cuda(), "A must be a CUDA tensor");
//     TORCH_CHECK(b.is_cuda(), "B must be a CUDA tensor");
//     TORCH_CHECK(c.is_cuda(), "C must be a CUDA tensor");
//     TORCH_CHECK(a.dtype() == torch::kHalf, "A must be float16");
//     TORCH_CHECK(b.dtype() == torch::kHalf, "B must be float16");
    
//     int M = a.size(0);
//     int K = a.size(1);
//     int N = b.size(0);
//     int K_b = b.size(1);
    
//     TORCH_CHECK(K == K_b, "K dimension mismatch");
    
//     const half_t* A_ptr = reinterpret_cast<const half_t*>(a.data_ptr<at::Half>());
//     const half_t* B_ptr = reinterpret_cast<const half_t*>(b.data_ptr<at::Half>());
//     half_t* C_ptr = reinterpret_cast<half_t*>(c.data_ptr<at::Half>());

//     auto options = torch::TensorOptions().dtype(torch::kFloat32).device(a.device());
//     auto norm_a = torch::empty({M}, options);
//     auto norm_b = torch::empty({N}, options);
    
//     float* d_norm_a = norm_a.data_ptr<float>();
//     float* d_norm_b = norm_b.data_ptr<float>();

//     int block_size = 256;
//     row_sq_sum_kernel<<<M, block_size>>>(A_ptr, d_norm_a, M, K);
//     row_sq_sum_kernel<<<N, block_size>>>(B_ptr, d_norm_b, N, K);

//     using Config = config::GemmConfig<half_t>;
    
//     dim3 block = dim3(128, 1, 1); 
//     dim3 grid((N + Config::kTileN - 1) / Config::kTileN,
//               (M + Config::kTileM - 1) / Config::kTileM);
    
//     int shm_gemm = cute::max(
//         int(cute::cosize(typename Config::SmemLayoutA{}) + cute::cosize(typename Config::SmemLayoutB{})),
//         int(cute::cosize(typename Config::SmemLayoutC{}))
//     ) * sizeof(half_t);
    
//     int shm_norms = (Config::kTileM + Config::kTileN) * sizeof(float);
//     int shm_total = shm_gemm + shm_norms;

//     cudaFuncSetAttribute(gemm_cosine_fused<Config>, cudaFuncAttributeMaxDynamicSharedMemorySize, shm_total);
    
//     gemm_cosine_fused<Config><<<grid, block, shm_total>>>(
//         C_ptr, A_ptr, B_ptr, 
//         d_norm_a, d_norm_b, 
//         M, N, K
//     );
// }

// PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//     m.def("cosine_similarity_cute", &cosine_similarity_cute, "CuTe Cosine Similarity");
// }
