cmake_minimum_required(VERSION 3.17)

# 定义项目名称和语言
project(CosineSimilarity LANGUAGES CXX CUDA)

# 查找 CUDA 工具链
find_package(CUDAToolkit REQUIRED)

# 设置默认构建类型为 Release (对应 -O3)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

# 设置 C++ 和 CUDA 标准 (对应 -std=c++17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# 设置目标 GPU 架构 (对应 -arch=sm_80)
# 如果你想让它自动检测当前 GPU，可以注释掉这一行，或者设置为 "native"
set(CMAKE_CUDA_ARCHITECTURES 80)

# 添加可执行文件
add_executable(cosine_similarity_v4.1 cosine_similarity_v4.1.cu)
add_executable(gemm_test gemm_test.cu)

# 设置包含目录 (对应 -I.)
# 这样编译器才能找到同目录下的 utils.h
target_include_directories(cosine_similarity_v4.1 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(gemm_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# 链接库 (对应 -lcublas)
target_link_libraries(cosine_similarity_v4.1 PRIVATE CUDA::cudart CUDA::cublas)
target_link_libraries(gemm_test PRIVATE CUDA::cudart CUDA::cublas)

# 编译选项
# -lineinfo: 生成行号信息，用于 ncu profiling
# Release 模式下 CMake 会自动添加 -O3
target_compile_options(cosine_similarity_v4.1 PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -O3>
)
target_compile_options(gemm_test PRIVATE 
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo -O3>
)

# # Debug 模式下的额外选项
# if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#     target_compile_options(cosine_similarity_v4.1 PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-G>)
# endif()