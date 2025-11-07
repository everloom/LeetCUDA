#include "cublas_v2.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <torch/extension.h>
#include <torch/types.h>

static cublasHandle_t g_handle = nullptr;

void init_cublas_handle() {
  if (g_handle == nullptr) {
    cublasStatus_t status = cublasCreate(&g_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to create cuBLAS handle: %d", status);
      exit(EXIT_FAILURE);
    }
    status = cublasSetMathMode(g_handle, CUBLAS_TENSOR_OP_MATH);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to set cuBLAS Math Mode: %d", status);
      exit(EXIT_FAILURE);
    }
  }
}

void destroy_cublas_handle() {
  if (g_handle != nullptr) {
    cublasStatus_t status = cublasDestroy(g_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
      printf("Failed to destroy cuBLAS handle: %d", status);
    }
    g_handle = nullptr;
  }
}

// NN: A/B/C All row major
// A: M x K, B: K x N, C: M x N
// 对于 hgemv，N=1，所以 B 和 C 都是向量
void hgemv_cublas_nn(torch::Tensor a, torch::Tensor b, torch::Tensor c) {
  // 检查数据类型
  TORCH_CHECK(a.dtype() == torch::kHalf, "a must be half");
  TORCH_CHECK(b.dtype() == torch::kHalf, "b must be half");
  TORCH_CHECK(c.dtype() == torch::kHalf, "c must be half");
  
  // 获取维度
  const int M = a.size(0);
  const int K = a.size(1);
  const int N = b.size(1);
  
  // 检查形状
  TORCH_CHECK(b.size(0) == K, "b shape mismatch");
  TORCH_CHECK(c.size(0) == M && c.size(1) == N, "c shape mismatch");
  
  static half alpha = 1.0;
  static half beta = 0.0;
  
  if (g_handle == nullptr) {
    init_cublas_handle();
  }
  
  // cuBLAS 使用列主序约定，但我们传入的是行主序矩阵
  // 通过调整参数顺序和转置操作来适配
  // 计算: C = A × B (所有矩阵行主序)
  // cuBLAS 视角: C^T = B^T × A^T (列主序)
  // 因此传入顺序为 B, A，操作均为 CUBLAS_OP_N
  cublasGemmEx(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
               reinterpret_cast<half *>(b.data_ptr()), CUDA_R_16F, N,
               reinterpret_cast<half *>(a.data_ptr()), CUDA_R_16F, K, &beta,
               reinterpret_cast<half *>(c.data_ptr()), CUDA_R_16F, N,
               CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("hgemv_cublas_nn", &hgemv_cublas_nn, "HGEMV cuBLAS NN (all row major)");
  m.def("init_cublas_handle", &init_cublas_handle, "Initialize cuBLAS handle");
  m.def("destroy_cublas_handle", &destroy_cublas_handle, "Destroy cuBLAS handle");
}

