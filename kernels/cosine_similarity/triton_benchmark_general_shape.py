import os
import torch
import triton
import triton.language as tl
import torch.nn.functional as F
from torch.utils.cpp_extension import load
import os
# [ADD THIS] 强制指定 CUDA 架构，防止 PyTorch 尝试为不支持 mma.h 的旧架构编译
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0" 
# =============================================================================
# 0. Load Custom CUDA Kernels
# =============================================================================

# 1. Try to load CuTe kernel (Commented out as requested)
_cute_lib = None
try:
    CUTLASS_REPO_PATH = os.environ.get(
        "CUTLASS_REPO_PATH", os.path.expanduser("../../third-party/cutlass")
    )
    
    _cute_lib = load(
        name="cosine_cute_lib",
        sources=["cosine_cute.cu"],
        extra_cuda_cflags=[
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
        ],
        extra_cflags=["-std=c++17"],
        extra_include_paths=[os.path.join(CUTLASS_REPO_PATH, "include")],
        verbose=False,
    )
    CUTE_AVAILABLE = True
except Exception as e:
    print(f"Warning: CuTe kernel not available: {e}")
    CUTE_AVAILABLE = False
    
##############################################
# 2. Load CUDA V2 Kernel
_cuda_v2_lib = None
CUDA_V2_AVAILABLE = False
try:
    # 假设 utils.h 在当前目录下，或者你需要指定正确的 include 路径
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 为了让 python 能调用 C++ 函数，我们需要一个 wrapper。
    # 由于 cosine_similarity_v3.cu 主要是 main 函数和 kernel，
    # 我们通常需要编写一个 pybind11 的绑定文件 (.cpp) 或者直接在 .cu 文件里加 PYBIND11_MODULE。
    # 但这里为了简单，我们假设你已经有一个适配 PyTorch 的 wrapper 或者我们直接编译 .cu
    # 如果 cosine_similarity_v3.cu 只有 main 函数而没有 PyTorch binding，直接 load 会失败。
    # 
    # **关键假设**：为了让这段代码工作，我们需要一个 C++ wrapper 来暴露 launch_cosine_kernel_tn 给 Python。
    # 下面我将动态生成一个 wrapper 文件来辅助编译。
    
    wrapper_code = """
    #include <torch/extension.h>
    #include <cuda_fp16.h>
    
    // Forward declaration of the launcher function in cosine_similarity_v3.cu
    // 注意：你需要确保 cosine_similarity_v3.cu 里的 launch_cosine_kernel_tn 不是 static 的，且能被链接
    // 或者我们直接 include 这个 .cu 文件（虽然不推荐，但对于单文件脚本最简单）
    
    // 由于 .cu 文件里有 main 函数，直接 include 会冲突。
    // 建议：将 cosine_similarity_v3.cu 中的 main 函数用 #ifndef BUILD_PYTORCH_EXTENSION 包裹
    // 或者我们这里简单粗暴地声明一下，假设 .cu 文件被修改为可以作为库使用。
    
    // 为了演示，我们假设 cosine_similarity_v3.cu 已经被修改为库形式，或者我们在这里重新声明
    // 实际上，最稳妥的方式是把 launch_cosine_kernel_tn 复制过来或者 include 头文件。
    // 这里我们采用一种 trick：创建一个 wrapper.cu，include 原文件，但在 include 前定义宏来屏蔽 main
    
    void launch_cosine_kernel_tn_wrapper(torch::Tensor a, torch::Tensor b, torch::Tensor c, int M, int N, int K);
    
    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("cosine_similarity_v3", &launch_cosine_kernel_tn_wrapper, "Cosine Similarity V2");
    }
    """
    
    # 我们需要一个 wrapper.cu 来包含原文件并提供 PyTorch 接口
    # 这里我们动态写入一个 wrapper.cu 文件
    with open("cosine_wrapper.cu", "w") as f:
        f.write("""
// [FIX] 先包含 CUDA 头文件，避免 PyTorch 头文件干扰
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>

// [FIX] 包含 PyTorch 头文件
#include <torch/extension.h>

// 定义宏以避免 main 函数冲突 (需要在 cosine_similarity_v3.cu 中配合 #ifndef BUILD_PYTORCH_EXTENSION)
#define BUILD_PYTORCH_EXTENSION

// [FIX] 包含原始 CUDA 文件
// 注意：这要求 cosine_similarity_v3.cu 在同一目录下
#include "cosine_similarity_v3.cu"

// Wrapper 函数
void launch_cosine_kernel_tn_wrapper(torch::Tensor a, torch::Tensor b, torch::Tensor c, int M, int N, int K) {
    // 检查输入是否连续
    // auto a_cont = a.contiguous();
    // auto b_cont = b.contiguous();
    
    // 获取数据指针
    half* a_ptr = reinterpret_cast<half*>(a.data_ptr<at::Half>());
    half* b_ptr = reinterpret_cast<half*>(b.data_ptr<at::Half>());
    half* c_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());
    
    // 调用原始的 launcher
    // 模板参数需要与 main 函数中一致: <2, 2048>
    launch_cosine_kernel_tn<2, 2048>(a_ptr, b_ptr, c_ptr, M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cosine_similarity_v3", &launch_cosine_kernel_tn_wrapper, "Cosine Similarity V2");
}
        """)

    _cuda_v2_lib = load(
        name="cosine_cuda_v2_lib",
        sources=["cosine_wrapper.cu"], 
        extra_cuda_cflags=[
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-arch=sm_80", # 显式指定架构
            # "-D__CUDA_ARCH__=800" # 显式定义架构宏，有时能帮助 IDE 或预处理器
        ],
        extra_cflags=["-std=c++17"],
        extra_include_paths=[curr_dir], 
        verbose=True, # 打开 verbose 查看详细编译命令
    )
    CUDA_V2_AVAILABLE = True
    print("CUDA V2 Kernel loaded successfully.")

except Exception as e:
    print(f"Warning: CUDA V2 kernel not available: {e}")
    CUDA_V2_AVAILABLE = False

##############################################
# 3. Load CUDA V4 (CuTe Pipeline) Kernel
_cuda_v4_lib = None
CUDA_V4_AVAILABLE = False
try:
    curr_dir = os.path.dirname(os.path.abspath(__file__))
    
    with open("cosine_wrapper_v4.cu", "w") as f:
        f.write("""
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <torch/extension.h>

#define BUILD_PYTORCH_EXTENSION
#include "cosine_similarity_v4.cu"

void launch_cosine_similarity_v4_torch(torch::Tensor a, torch::Tensor b, torch::Tensor c, int M, int N, int K) {
    using T = cute::half_t;
    launch_cosine_similarity_v4<T, 2, true>(
        reinterpret_cast<T*>(a.data_ptr<at::Half>()),
        reinterpret_cast<T*>(b.data_ptr<at::Half>()),
        reinterpret_cast<T*>(c.data_ptr<at::Half>()),
        M, N, K,
        2048
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cosine_similarity_v4", &launch_cosine_similarity_v4_torch, "Cosine Similarity V4 (CuTe)");
}
        """)

    # 获取当前脚本所在目录的绝对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 推断项目根目录 (假设当前脚本在 kernels/cosine_similarity/)
    project_root = os.path.abspath(os.path.join(script_dir, "../../"))
    
    CUTLASS_REPO_PATH = os.environ.get(
        "CUTLASS_REPO_PATH", os.path.join(project_root, "third-party/cutlass")
    )

    _cuda_v4_lib = load(
        name="cosine_cuda_v4_lib",
        sources=["cosine_wrapper_v4.cu"], 
        extra_cuda_cflags=[
            "-O3",
            "-U__CUDA_NO_HALF_OPERATORS__",
            "-U__CUDA_NO_HALF_CONVERSIONS__",
            "-U__CUDA_NO_HALF2_OPERATORS__",
            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
            "--use_fast_math",
            "-arch=sm_80",
        ],
        extra_cflags=["-std=c++17"],
        extra_include_paths=[
            curr_dir,
            os.path.join(CUTLASS_REPO_PATH, "include")
        ], 
        verbose=True,
    )
    CUDA_V4_AVAILABLE = True
    print("CUDA V4 Kernel (CuTe) loaded successfully.")

except Exception as e:
    print(f"Warning: CUDA V4 kernel not available: {e}")
    CUDA_V4_AVAILABLE = False

#####################################################################
# =============================================================================
# 1. Triton Kernel v1: Simple (最初版本)
# =============================================================================

@triton.jit
def cosine_similarity_kernel_simple(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 获取当前程序的 pid，确定计算结果矩阵的哪个 Block
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 创建 ranges
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)

    # 初始化指针
    # A 形状 [M, K], B 形状 [N, K] (注意这里B是未转置的内存布局，但在计算时我们当作B^T处理)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_bn[None, :] * stride_bn + offs_k[:, None] * stride_bk) # 注意这里转置了B的读取方式

    # 初始化累加器
    # accumulator 存储 A * B^T 的点积
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # norm_a_sq 存储 A 的行的平方和，形状 [BLOCK_M]
    norm_a_sq = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    
    # norm_b_sq 存储 B 的行的平方和，形状 [BLOCK_N]
    norm_b_sq = tl.zeros((BLOCK_N, ), dtype=tl.float32)

    # 沿着 K 维度循环
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 1. 加载数据
        # 处理边界掩码，防止越界
        mask_k = offs_k < K - k * BLOCK_K
        
        # 加载 A 的分块 [BLOCK_M, BLOCK_K]
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        # 加载 B 的分块 (实际上是 B^T 的一部分) [BLOCK_K, BLOCK_N]
        b = tl.load(b_ptrs, mask=mask_k[:, None], other=0.0)

        # 2. 计算点积 (利用 Tensor Core)
        acc += tl.dot(a, b)

        # 3. 顺带计算 L2 范数的平方和 (利用 FPU)
        # 计算 A 的平方和。a 是 [BLOCK_M, BLOCK_K]，我们需要沿轴 1 求和
        # 注意：Triton 的 reduction 需要 fp32 以保证精度
        _a_sq = a.to(tl.float32) * a.to(tl.float32)
        norm_a_sq += tl.sum(_a_sq, axis=1)

        # 计算 B 的平方和。b 是 [BLOCK_K, BLOCK_N]，我们需要沿轴 0 求和
        _b_sq = b.to(tl.float32) * b.to(tl.float32)
        norm_b_sq += tl.sum(_b_sq, axis=0)

        # 指针步进
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # 4. Epilogue: 计算最终的 Cosine Similarity
    # 计算 1 / ||A|| 和 1 / ||B||
    # 加上 eps 防止除以 0
    eps = 1e-6
    rnorm_a = tl.rsqrt(norm_a_sq + eps)
    rnorm_b = tl.rsqrt(norm_b_sq + eps)

    # 利用广播机制将归一化系数应用到结果矩阵
    # acc [BLOCK_M, BLOCK_N]
    # rnorm_a [BLOCK_M] -> [BLOCK_M, 1]
    # rnorm_b [BLOCK_N] -> [1, BLOCK_N]
    c = acc * rnorm_a[:, None] * rnorm_b[None, :]

    # 写回结果
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    
    # 边界掩码
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)

def run_triton_simple(a, b):
    M, K = a.shape
    N, K_b = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']), triton.cdiv(N, META['BLOCK_N']))
    
    cosine_similarity_kernel_simple[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
    )
    return c

# =============================================================================
# 2. Triton Kernel v2: Optimized (深度优化版)
# =============================================================================

@triton.jit
def cosine_similarity_kernel_opt(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr
):
    # -----------------------------------------------------------
    # 1. Swizzling (L2 Cache 优化)
    # 通过重新排列 Block 的执行顺序，增加数据的复用率
    # -----------------------------------------------------------
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # -----------------------------------------------------------
    # 2. 指针初始化
    # -----------------------------------------------------------
    # A 的指针：指向 row block [pid_m * BLOCK_M, :]
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)

    # B 的指针：指向 row block [pid_n * BLOCK_N, :]
    # 关键优化：我们按 B 在内存中的物理布局(行)加载，而不是按逻辑上的列加载
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    b_ptrs = b_ptr + (offs_bn[:, None] * stride_bn + offs_k[None, :] * stride_bk)

    # -----------------------------------------------------------
    # 3. 累加器初始化
    # -----------------------------------------------------------
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    norm_a_sq = tl.zeros((BLOCK_M, ), dtype=tl.float32)
    norm_b_sq = tl.zeros((BLOCK_N, ), dtype=tl.float32)

    # -----------------------------------------------------------
    # 4. 主循环 (K 维度)
    # -----------------------------------------------------------
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # 边界检查 mask
        mask_k = offs_k < K - k * BLOCK_K
        
        # 加载 A [BLOCK_M, BLOCK_K]
        a = tl.load(a_ptrs, mask=mask_k[None, :], other=0.0)
        
        # 加载 B [BLOCK_N, BLOCK_K] 
        # 优化：这里是连续内存访问 (Coalesced Load)，速度极快
        b = tl.load(b_ptrs, mask=mask_k[None, :], other=0.0)

        # 计算点积: A * B^T
        # Triton 的 dot 能够高效处理转置，在 Shared Memory 中完成 Swizzle
        acc += tl.dot(a, b.trans())

        # 计算 A 的平方和 (沿 K 轴: axis=1)
        _a = a.to(tl.float32)
        norm_a_sq += tl.sum(_a * _a, axis=1)

        # 计算 B 的平方和 (沿 K 轴: axis=1)
        # 注意：这里 B 是 [BLOCK_N, BLOCK_K]，所以也是沿 axis=1 求和
        # 之前的代码这里比较别扭，现在顺畅了
        _b = b.to(tl.float32)
        norm_b_sq += tl.sum(_b * _b, axis=1)

        # 指针步进
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # -----------------------------------------------------------
    # 5. Epilogue (归一化)
    # -----------------------------------------------------------
    eps = 1e-6
    rnorm_a = tl.rsqrt(norm_a_sq + eps) # [BLOCK_M]
    rnorm_b = tl.rsqrt(norm_b_sq + eps) # [BLOCK_N]

    # 广播并计算最终结果
    c = acc * rnorm_a[:, None] * rnorm_b[None, :]

    # -----------------------------------------------------------
    # 6. 写回结果
    # -----------------------------------------------------------
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    # 转换回 fp16 写回
    tl.store(c_ptrs, c.to(tl.float16), mask=c_mask)

def run_triton_opt(a, b):
    M, K = a.shape
    N, K_b = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    cosine_similarity_kernel_opt[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
        GROUP_SIZE_M=8,
        num_stages=3,
        num_warps=8
    )
    return c

# =============================================================================
# 3. PyTorch Implementations
# =============================================================================

def run_torch_native(a, b):
    # Broadcasting way
    return F.cosine_similarity(a.unsqueeze(1), b.unsqueeze(0), dim=-1)

def run_torch_mm(a, b):
    # Norm + MM way
    a_n = F.normalize(a, p=2, dim=1)
    b_n = F.normalize(b, p=2, dim=1)
    return torch.mm(a_n, b_n.t())

# =============================================================================
# 4. CuTe Implementation (Disabled)
# =============================================================================

def run_cute(a, b):
    if not CUTE_AVAILABLE:
        raise RuntimeError("CuTe kernel is not available. Please check compilation errors.")
    M, K = a.shape
    N, K_b = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    _cute_lib.cosine_similarity_cute(a, b, c)
    return c
    # pass

# =============================================================================
# 5. CUDA V2/V4 Implementation
# =============================================================================

def run_cuda_v2(a, b):
    if not CUDA_V2_AVAILABLE:
        raise RuntimeError("CUDA V2 kernel is not available.")
    M, K = a.shape
    N, K_b = b.shape
    if not a.is_contiguous(): a = a.contiguous()
    if not b.is_contiguous(): b = b.contiguous()
    
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    _cuda_v2_lib.cosine_similarity_v3(a, b, c, M, N, K)
    return c

def run_cuda_v4(a, b):
    if not CUDA_V4_AVAILABLE:
        raise RuntimeError("CUDA V4 kernel is not available.")
    M, K = a.shape
    N, K_b = b.shape
    if not a.is_contiguous(): a = a.contiguous()
    if not b.is_contiguous(): b = b.contiguous()
    
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    _cuda_v4_lib.cosine_similarity_v4(a, b, c, M, N, K)
    return c

# =============================================================================
# 6. Benchmark Configuration
# =============================================================================

BATCH_SIZES = [16, 64, 128, 256, 512, 1024, 2048, 4096]
SEQ_LENS = [64, 128, 256, 512, 1024]

configs = []
for seqlen in SEQ_LENS:
    configs.append(
        triton.testing.Benchmark(
            x_names=['bs'], 
            x_vals=BATCH_SIZES, 
            line_arg='provider',
            # line_vals=['triton-opt', 'triton-simple', 'torch-mm', 'torch-native', 'cute'],
            # line_names=['Triton Optimized', 'Triton Simple', 'Torch Norm+MM', 'Torch Native', 'CuTe'],
            # styles=[('blue', '-'), ('red', '-'), ('green', '--'), ('orange', ':'), ('purple', '-.')],
            line_vals=['triton-opt', 'triton-simple', 'torch-mm'] + (['cuda-v2'] if CUDA_V2_AVAILABLE else []) + (['cuda-v4'] if CUDA_V4_AVAILABLE else []) + (['cute'] if CUTE_AVAILABLE else []),
            line_names=['Triton Optimized', 'Triton Simple', 'Torch Norm+MM'] + (['CUDA V2'] if CUDA_V2_AVAILABLE else []) + (['CUDA V4 (CuTe)'] if CUDA_V4_AVAILABLE else []) + (['CuTe'] if CUTE_AVAILABLE else []),
            styles=[('blue', '-'), ('red', '-'), ('green', '--')] + ([('orange', '-.')] if CUDA_V2_AVAILABLE else []) + ([('cyan', '-.')] if CUDA_V4_AVAILABLE else []) + ([('purple', '-.')] if CUTE_AVAILABLE else []),
            ylabel='Time (ms)',
            plot_name=f'cosine-similarity-seqlen-{seqlen}',
            args={'seqlen': seqlen},
        )
    )

@triton.testing.perf_report(configs)
def benchmark(bs, seqlen, provider):
    a = torch.randn(bs, seqlen, device='cuda', dtype=torch.float16)
    b = torch.randn(bs, seqlen, device='cuda', dtype=torch.float16)
    
    quantiles = [0.5, 0.2, 0.8]
    
    if provider == 'triton-opt':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_triton_opt(a, b), quantiles=quantiles)
    elif provider == 'triton-simple':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_triton_simple(a, b), quantiles=quantiles)
    elif provider == 'torch-mm':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_torch_mm(a, b), quantiles=quantiles)
    elif provider == 'torch-native':
        try:
            # 简单的 OOM 保护
            if bs > 1024 and seqlen > 512: 
                raise RuntimeError("Skipping to prevent OOM")
            ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_torch_native(a, b), quantiles=quantiles)
        except RuntimeError:
            ms, min_ms, max_ms = float('inf'), float('inf'), float('inf')
    elif provider == 'cuda-v2':
        if not CUDA_V2_AVAILABLE:
            ms, min_ms, max_ms = float('inf'), float('inf'), float('inf')
        else:
            try:
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_cuda_v2(a, b), quantiles=quantiles)
            except Exception as e:
                print(f"Warning: CUDA V2 kernel failed: {e}")
                ms, min_ms, max_ms = float('inf'), float('inf'), float('inf')
    elif provider == 'cuda-v4':
        if not CUDA_V4_AVAILABLE:
            ms, min_ms, max_ms = float('inf'), float('inf'), float('inf')
        else:
            try:
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_cuda_v4(a, b), quantiles=quantiles)
            except Exception as e:
                print(f"Warning: CUDA V4 kernel failed: {e}")
                ms, min_ms, max_ms = float('inf'), float('inf'), float('inf')
    elif provider == 'cute':
        if not CUTE_AVAILABLE:
            ms, min_ms, max_ms = float('inf'), float('inf'), float('inf')
        else:
            try:
                ms, min_ms, max_ms = triton.testing.do_bench(lambda: run_cute(a, b), quantiles=quantiles)
            except Exception as e:
                print(f"Warning: CuTe kernel failed: {e}")
                ms, min_ms, max_ms = float('inf'), float('inf'), float('inf')
    else:
        raise ValueError(f"Unknown provider: {provider}")
            
    return ms, min_ms, max_ms

# =============================================================================
# 7. Sanity Check
# =============================================================================
def sanity_check():
    print("Running Sanity Check...")
    bs, seqlen = 128, 128
    a = torch.randn(bs, seqlen, device='cuda', dtype=torch.float16)
    b = torch.randn(bs, seqlen, device='cuda', dtype=torch.float16)

    t_opt = run_triton_opt(a, b)
    t_sim = run_triton_simple(a, b)
    p_mm = run_torch_mm(a, b)
    
    diff_opt = (t_opt - p_mm).abs().max()
    diff_sim = (t_sim - p_mm).abs().max()
    
    print(f"Max Diff (Optimized vs Torch): {diff_opt:.6f}")
    print(f"Max Diff (Simple vs Torch):    {diff_sim:.6f}")
    
    assert diff_opt < 1e-2, "Triton Optimized check failed"
    assert diff_sim < 1e-2, "Triton Simple check failed"
    
    # Check CUDA V2 kernel if available
    if CUDA_V2_AVAILABLE:
        try:
            t_cuda = run_cuda_v2(a, b)
            # 注意：CUDA Kernel 可能没有做边界检查，如果 bs/seqlen 不是 128 的倍数可能会有问题
            # 但这里 128 是对齐的。
            diff_cuda = (t_cuda - p_mm).abs().max()
            print(f"Max Diff (CUDA V2 vs Torch):     {diff_cuda:.6f}")
            # 稍微放宽一点误差，因为 fp16 累加精度问题
            assert diff_cuda < 5e-2, "CUDA V2 check failed"
        except Exception as e:
            print(f"Warning: CUDA V2 sanity check failed: {e}")

    # Check CUDA V4 kernel if available
    if CUDA_V4_AVAILABLE:
        try:
            t_cuda4 = run_cuda_v4(a, b)
            diff_cuda4 = (t_cuda4 - p_mm).abs().max()
            print(f"Max Diff (CUDA V4 vs Torch):     {diff_cuda4:.6f}")
            assert diff_cuda4 < 5e-2, "CUDA V4 check failed"
        except Exception as e:
            print(f"Warning: CUDA V4 sanity check failed: {e}")

    # Check CuTe kernel if available
    if CUTE_AVAILABLE:
        try:
            t_cute = run_cute(a, b)
            diff_cute = (t_cute - p_mm).abs().max()
            print(f"Max Diff (CuTe vs Torch):        {diff_cute:.6f}")
            assert diff_cute < 1e-2, "CuTe check failed"
        except Exception as e:
            print(f"Warning: CuTe sanity check failed: {e}")

    print("Sanity Check Passed!\n")

if __name__ == '__main__':
    # 为了避免 main 函数冲突，我们需要在 cosine_similarity_v3.cu 中添加 #ifndef BUILD_PYTORCH_EXTENSION
    # 请确保你已经修改了 .cu 文件，或者接受 wrapper 中的 trick。
    sanity_check()
    print("Starting Benchmark...")
    benchmark.run(show_plots=True, print_data=True, save_path='.')