
#include <torch/extension.h>
#include <vector>

// Define BUILD_PYTORCH_EXTENSION to disable main function in the included file
#ifndef BUILD_PYTORCH_EXTENSION
#define BUILD_PYTORCH_EXTENSION
#endif

// Include the kernel source directly to access templates
// 假设 cosine_similarity_v4.cu 在同一目录下
#include "cosine_similarity_v4.cu"

void run_cosine_v4_wrapper(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
    int M = a.size(0);
    int K = a.size(1);
    int N = b.size(0);

    // Ensure inputs are contiguous and on CUDA
    auto a_contig = a.contiguous();
    auto b_contig = b.contiguous();
    
    half* a_ptr = reinterpret_cast<half*>(a_contig.data_ptr<at::Half>());
    half* b_ptr = reinterpret_cast<half*>(b_contig.data_ptr<at::Half>());
    half* c_ptr = reinterpret_cast<half*>(c.data_ptr<at::Half>());

    // Launch with K_STAGE=3 as optimized (consistent with your main function)
    launch_cosine_kernel_tn<3>(a_ptr, b_ptr, c_ptr, M, N, K);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("run_cosine_v4", &run_cosine_v4_wrapper, "Run Cosine Similarity V4");
}
