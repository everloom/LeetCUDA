
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
        