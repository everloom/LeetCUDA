import os
import torch
import torch.nn.functional as F
from torch.cuda import nvtx
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0' # 强制指定

def run_torch_mm_profile(a, b):
    # Norm + MM way
    # 标记第一个 Normalize
    with nvtx.range("a_normalize"):
        a_n = F.normalize(a, p=2, dim=1)
    
    # 标记第二个 Normalize
    with nvtx.range("b_normalize"):
        b_n = F.normalize(b, p=2, dim=1)
    
    # 标记矩阵乘法
    # .t() 只是元数据操作，耗时忽略不计，主要耗时在 mm
    with nvtx.range("mm"):
        res = torch.mm(a_n, b_n.t())
    
    return res

def _cos_sim_logic(a, b):
    a_n = F.normalize(a, p=2, dim=1)
    b_n = F.normalize(b, p=2, dim=1)
    return torch.mm(a_n, b_n.t())

# 编译
_compiled_cos_sim = torch.compile(_cos_sim_logic, backend="inductor")

def run_torch_mm_profile_compile(a, b):
    # 使用编译后的函数
    # 第一次运行时会看到编译日志（如果开启了 TORCH_LOGS）
    with nvtx.range("inductor_exec"):
        res = _compiled_cos_sim(a, b)
    return res

def profile_torch_mm():
    # bs_a = 8
    # seqlen_a = 128
    
    # bs_b = 250000
    # seqlen_b = 128
    
    bs_b = 8
    seqlen_b = 128
    
    bs_a = 250000
    seqlen_a = 128
    
    a = torch.ones(bs_a, seqlen_a, device='cuda', dtype=torch.float16)
    b = torch.ones(bs_b, seqlen_b, device='cuda', dtype=torch.float16)


    # # print("Warming up...")
    for _ in range(10):
        run_torch_mm_profile(a, b)
        
    run_torch_mm_profile(a, b)


if __name__ == '__main__':
    # 注释掉原来的 benchmark 逻辑，只运行 profiling 函数
    # sanity_check()
    # print("Starting Benchmark...")
    # benchmark.run(show_plots=True, print_data=True, save_path='.')
    
    profile_torch_mm()