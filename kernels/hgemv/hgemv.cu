#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <float.h>
#include <stdio.h>
#include <stdlib.h>
#include <torch/extension.h>
#include <torch/types.h>
#include <vector>

#define WARP_SIZE 32
#define INT4(value) (reinterpret_cast<int4 *>(&(value))[0])
#define FLOAT4(value) (reinterpret_cast<float4 *>(&(value))[0])
#define HALF2(value) (reinterpret_cast<half2 *>(&(value))[0])
#define BFLOAT2(value) (reinterpret_cast<__nv_bfloat162 *>(&(value))[0])

// FP16
// Warp Reduce Sum
template <const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ half warp_reduce_sum_f16(half val) {
#pragma unroll
  for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
    // 这里的理解可以参考softmax.cu中的warp_reduce_max_f32的注释
    val += __shfl_xor_sync(0xffffffff, val, mask);
  }
  return val;
}

// HGEMV: Warp HGEMV K32
// 假设K为32的倍数，每个warp负责一行
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
// 这里以M=1024，K=128，N=1为例
__global__ void hgemv_k32_f16_kernel(half *a, half *x, half *y, int M, int K) {
  int tx = threadIdx.x;         // 0~31
  int ty = threadIdx.y;         // 0~4
  int bx = blockIdx.x;          // 0~M/4
  int lane = tx % WARP_SIZE;    // 0~31
  int m = bx * blockDim.y + ty; // (0~M/4) * 4 + (0~3)
  if (m < M) {
    half sum = 0.0f;
    // 这里是计算一行需要多少个warp来处理
    int NUM_WARPS = (K + WARP_SIZE - 1) / WARP_SIZE;
#pragma unroll
/*
解释一下这里为什么要用for循环
在这里的例子中，K=128，而在这个kernel中，一个warp负责处理一行的计算
一行的计算的意思是，1024*128（a）和128*1（b）的乘法，结果是1024*1（c）的，一行的计算就是，a的一行和b做点积运算∑ai*bi
这里K=128，而warp只有32个线程，所以明显每个线程需要负责多个元素的计算，所以这里使用了for循环
循环NUM_WARPS次，就是每个线程需要负责a中一行与b中一行的点积运算中的NUM_WARPS个元素乘法(num_warps个ai*bi)，对应int k = w * WARP_SIZE + lane和sum += a[m * K + k] * x[k]这两行
而sum += a[m * K + k] * x[k]中的m*k的意思是行主序的offset，m是行号，而K就是行方向上的偏移，小k就是列方向上的偏移
for循环算完之后，∑a*b的结果就汇聚到了每行第一个warp的32个线程中，然后再调用一个warp reduce sum就能在laneid 0上得到hgemv的结果
*/
    for (int w = 0; w < NUM_WARPS; ++w) {
      // 若NUM_WARPS>=2，先将当前行的数据累加到第一个warp中
      int k = w * WARP_SIZE + lane;
      sum += a[m * K + k] * x[k];
    }
    sum = warp_reduce_sum_f16<WARP_SIZE>(sum);
    if (lane == 0)
      y[m] = sum;
  }
}

// HGEMV: Warp HGEMV K128 + half2x2
// 假设K为128的倍数 float4
// grid(M/4), block(32,4) blockDim.x=32=K, blockDim.y=4
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
/*
这个kernel没细看，看起来和hgemv_k32_f16_kernel的区别就是
1、这里要求K是128的倍数了，不过仍然还是一个warp处理一行
2、使用了循环展开和向量化访存
*/
__global__ void hgemv_k128_f16x4_kernel(half *a, half *x, half *y, int M,
                                        int K) {
  // 每个线程负责4个元素，一个warp覆盖128个元素
  int tx = threadIdx.x;         // 0~31
  int ty = threadIdx.y;         // 0~3
  int bx = blockIdx.x;          // 0~M/4
  int lane = tx % WARP_SIZE;    // 0~31
  int m = blockDim.y * bx + ty; // (0~M/4) * 4 + (0~3)

  if (m < M) {
    half sum = 0.0f;
    // process 4*WARP_SIZE elements per warp.
    int NUM_WARPS = (((K + WARP_SIZE - 1) / WARP_SIZE) + 4 - 1) / 4;
#pragma unroll
    for (int w = 0; w < NUM_WARPS; ++w) {
      int k = (w * WARP_SIZE + lane) * 4;
      half2 reg_x_0 = HALF2(x[k + 0]);
      half2 reg_x_1 = HALF2(x[k + 2]);
      half2 reg_a_0 = HALF2(a[m * K + k + 0]);
      half2 reg_a_1 = HALF2(a[m * K + k + 2]);
      sum += (reg_x_0.x * reg_a_0.x + reg_x_0.y * reg_a_0.y +
              reg_x_1.x * reg_a_1.x + reg_x_1.y * reg_a_1.y);
    }
    sum = warp_reduce_sum_f16<WARP_SIZE>(sum);
    if (lane == 0)
      y[m] = sum;
  }
}

// HGEMV: Warp HGEMV K16
// 假设K为16 < 32,每个warp负责2行，每行有16个元素
// NUM_THREADS=128, NUM_WARPS=NUM_THREADS/WARP_SIZE;
// NUM_ROWS=NUM_WARPS * ROW_PER_WARP, grid(M/NUM_ROWS), block(32,NUM_WARPS)
// a: MxK, x: Kx1, y: Mx1, compute: y = a * x
template <const int ROW_PER_WARP = 2>
__global__ void hgemv_k16_f16_kernel(half *A, half *x, half *y, int M, int K) {
  constexpr int K_WARP_SIZE = (WARP_SIZE + ROW_PER_WARP - 1) / ROW_PER_WARP;
  int tx = threadIdx.x;      // 0~31
  int ty = threadIdx.y;      // 0~NUM_WARPS
  int bx = blockIdx.x;       // 0~M/NUM_ROWS (NUM_ROWS=NUM_WARPS * ROW_PER_WARP)
  int lane = tx % WARP_SIZE; // 0~31
  int k = lane % K_WARP_SIZE; // 0~15
  // gloabl row of a: MxK and y:Mx1, blockDim.y=NUM_WARPS
  int m = (blockDim.y * bx + ty) * ROW_PER_WARP + lane / K_WARP_SIZE;
  if (m < M) {
    half sum = A[m * K + k] * x[k];
    sum = warp_reduce_sum_f16<K_WARP_SIZE>(sum);
    // 注意是k == 0，而不是lane == 0
    if (k == 0)
      y[m] = sum;
  }
}

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

#define ASSERT_K_IS_MULTIBLE_OF(V)                                             \
  if (K % (V) != 0) {                                                          \
    throw std::runtime_error("K must be multiple of " #V);                     \
  }

#define ASSERT_K_IS_EQUAL_OF(V)                                                \
  if (K != (V)) {                                                              \
    throw std::runtime_error("K must be " #V);                                 \
  }

void hgemv_k32_f16(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
  CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
  ASSERT_K_IS_MULTIBLE_OF(32)

  dim3 block(32, 4);
  dim3 grid((M + 4 - 1) / 4);

  hgemv_k32_f16_kernel<<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                                        reinterpret_cast<half *>(x.data_ptr()),
                                        reinterpret_cast<half *>(y.data_ptr()),
                                        M, K);
}

void hgemv_k128_f16x4(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
  CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
  ASSERT_K_IS_MULTIBLE_OF(128)

  dim3 block(32, 4);
  dim3 grid((M + 4 - 1) / 4);

  hgemv_k128_f16x4_kernel<<<grid, block>>>(
      reinterpret_cast<half *>(a.data_ptr()),
      reinterpret_cast<half *>(x.data_ptr()),
      reinterpret_cast<half *>(y.data_ptr()), M, K);
}

void hgemv_k16_f16(torch::Tensor a, torch::Tensor x, torch::Tensor y) {
  CHECK_TORCH_TENSOR_DTYPE(a, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(x, torch::kHalf)
  CHECK_TORCH_TENSOR_DTYPE(y, torch::kHalf)
  const int M = a.size(0);
  const int K = a.size(1);
  CHECK_TORCH_TENSOR_SHAPE(a, M, K)
  CHECK_TORCH_TENSOR_SHAPE(x, K, 1)
  CHECK_TORCH_TENSOR_SHAPE(y, M, 1)
  ASSERT_K_IS_EQUAL_OF(16)

  constexpr int NUM_THREADS = 128;
  constexpr int ROW_PER_WARP = 2;
  constexpr int NUM_WARPS = NUM_THREADS / WARP_SIZE; // 4
  constexpr int NUM_ROWS = NUM_WARPS * ROW_PER_WARP; // 4 * 2 = 8

  dim3 block(32, NUM_WARPS);
  dim3 grid((M + NUM_ROWS - 1) / NUM_ROWS);

  hgemv_k16_f16_kernel<ROW_PER_WARP>
      <<<grid, block>>>(reinterpret_cast<half *>(a.data_ptr()),
                        reinterpret_cast<half *>(x.data_ptr()),
                        reinterpret_cast<half *>(y.data_ptr()), M, K);
}

extern void hgemv_f16_cute(torch::Tensor, torch::Tensor, torch::Tensor);
extern void hgemv_f16x8_cute(torch::Tensor, torch::Tensor, torch::Tensor);
extern void hgemv_tensor_core_cute(torch::Tensor, torch::Tensor, torch::Tensor);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  TORCH_BINDING_COMMON_EXTENSION(hgemv_k32_f16)
  TORCH_BINDING_COMMON_EXTENSION(hgemv_k128_f16x4)
  TORCH_BINDING_COMMON_EXTENSION(hgemv_k16_f16)
  TORCH_BINDING_COMMON_EXTENSION(hgemv_f16_cute)
  TORCH_BINDING_COMMON_EXTENSION(hgemv_f16x8_cute)
  TORCH_BINDING_COMMON_EXTENSION(hgemv_tensor_core_cute)
}
