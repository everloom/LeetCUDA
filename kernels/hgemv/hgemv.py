import os
import time
from functools import partial
from typing import Optional

import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

CUTLASS_REPO_PATH = os.environ.get(
    "CUTLASS_REPO_PATH", os.path.expanduser("../../third-party/cutlass")
)

# Load the CUDA kernel as a python module
lib = load(
    name="hgemv_lib",
    sources=["hgemv.cu", "hgemv_cute.cu"],
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
)

# Load cuBLAS kernel for comparison
# 加载 cuBLAS 实现用于性能对比
cublas_lib = load(
    name="hgemv_cublas_lib",
    sources=["hgemv_cublas.cu"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    ],
    extra_cflags=["-std=c++17"],
    extra_ldflags=["-lcublas"],
)


def check_correctness(result: torch.Tensor, reference: torch.Tensor, tag: str, rtol: float = 1e-2, atol: float = 1e-3):
    """
    验证结果的正确性
    
    Args:
        result: 待验证的结果
        reference: 参考结果（通常是 PyTorch 的结果）
        tag: 标签名称
        rtol: 相对误差容忍度
        atol: 绝对误差容忍度
    
    Returns:
        bool: 是否通过验证
    """
    try:
        # 计算最大绝对误差和相对误差
        abs_diff = torch.abs(result - reference)
        max_abs_diff = torch.max(abs_diff).item()
        
        rel_diff = abs_diff / (torch.abs(reference) + 1e-8)
        max_rel_diff = torch.max(rel_diff).item()
        
        # 使用 torch.allclose 进行验证
        is_correct = torch.allclose(result, reference, rtol=rtol, atol=atol)
        
        # 计算通过率
        close_mask = torch.isclose(result, reference, rtol=rtol, atol=atol)
        pass_rate = close_mask.float().mean().item() * 100
        
        status = "✓ PASS" if is_correct else "✗ FAIL"
        print(f"  [{status}] {tag:>25}: max_abs_diff={max_abs_diff:.6f}, max_rel_diff={max_rel_diff:.6f}, pass_rate={pass_rate:.2f}%")
        
        return is_correct
    except Exception as e:
        print(f"  [✗ ERROR] {tag:>25}: {e}")
        return False


def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    warmup: int = 10,
    iters: int = 200,
    show_all: bool = False,
    reference: Optional[torch.Tensor] = None,
):
    if out is not None:
        out.fill_(0)
    if out is not None:
        for i in range(warmup):
            perf_func(a, b, out)
    else:
        for i in range(warmup):
            _ = perf_func(a, b)

    torch.cuda.synchronize()
    start = time.time()
    # iters
    if out is not None:
        for i in range(iters):
            perf_func(a, b, out)
    else:
        for i in range(iters):
            out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    total_time = (end - start) * 1000  # ms
    mean_time = total_time / iters
    out_info = f"out_{tag}"
    out_val = out.flatten().detach().cpu().numpy().tolist()[:3]
    out_val = [round(v, 8) for v in out_val]
    print(f"{out_info:>30}: {out_val}, time:{mean_time:.8f}ms", end="")
    
    # 验证正确性
    if reference is not None:
        print()  # 先换行
        is_correct = check_correctness(out, reference, tag)
    else:
        print()  # 如果没有参考结果，直接换行
    
    if show_all:
        print(out)
    return out.clone(), mean_time


print("-" * 80)
M, N, K = 1024, 1, 128
print(f"Matrix Layout Analysis for M={M}, N={N}, K={K}:")
print(f"  A: shape ({M}, {K}) (行主序 Row-major)")
print(f"  B: shape ({K}, {N}) (行主序 Row-major, 向量)")
print(f"  C: shape ({M}, {N}) (行主序 Row-major, 向量)")
print(f"  使用 NN 模式 (所有矩阵都是行主序)")
print("-" * 80)
a = torch.randn((M, K)).cuda().half().contiguous()
b = torch.randn((K, N)).cuda().half().contiguous()
c = torch.randn((M, N)).cuda().half().contiguous()

# 首先运行 PyTorch 作为参考结果
print("Running PyTorch reference...")
# 是否进行结果正确性检查开关
# reference, ref_time = run_benchmark(partial(torch.matmul, out=c), a, b, "f16_th (reference)")
reference = None
print("\nVerifying other implementations against PyTorch reference:")
print("-" * 80)

run_benchmark(cublas_lib.hgemv_cublas_nn, a, b, "cublas_nn", c, reference=reference)
run_benchmark(lib.hgemv_k32_f16, a, b, "k32f16", c, reference=reference)
run_benchmark(lib.hgemv_k128_f16x4, a, b, "k128f16x4", c, reference=reference)
run_benchmark(lib.hgemv_f16_cute, a, b, "hgemv_f16_cute", c, reference=reference)
run_benchmark(lib.hgemv_f16x8_cute, a, b, "hgemv_f16x8_cute", c, reference=reference)
run_benchmark(lib.hgemv_tensor_core_cute, a, b, "hgemv_tensor_core_cute", c, reference=reference)
print("-" * 80)

M, N, K = 1024, 1, 16
print(f"\nMatrix Layout Analysis for M={M}, N={N}, K={K}:")
print(f"  A: shape ({M}, {K}) (行主序 Row-major)")
print(f"  B: shape ({K}, {N}) (行主序 Row-major, 向量)")
print(f"  C: shape ({M}, {N}) (行主序 Row-major, 向量)")
print(f"  使用 NN 模式 (所有矩阵都是行主序)")
print("-" * 80)
a = torch.randn((M, K)).cuda().half().contiguous()
b = torch.randn((K, N)).cuda().half().contiguous()
c = torch.randn((M, N)).cuda().half().contiguous()

# 首先运行 PyTorch 作为参考结果
print("Running PyTorch reference...")
# 是否进行结果正确性检查开关
# reference, ref_time = run_benchmark(partial(torch.matmul, out=c), a, b, "f16_th (reference)")
reference = None
print("\nVerifying other implementations against PyTorch reference:")
print("-" * 80)

run_benchmark(cublas_lib.hgemv_cublas_nn, a, b, "cublas_nn", c, reference=reference)
run_benchmark(lib.hgemv_k16_f16, a, b, "k16f16", c, reference=reference)
run_benchmark(lib.hgemv_f16_cute, a, b, "hgemv_f16_cute", c, reference=reference)
run_benchmark(lib.hgemv_f16x8_cute, a, b, "hgemv_f16x8_cute", c, reference=reference)
run_benchmark(lib.hgemv_tensor_core_cute, a, b, "hgemv_tensor_core_cute", c, reference=reference)
print("-" * 80)
