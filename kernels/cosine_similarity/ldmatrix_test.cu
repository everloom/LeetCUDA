#include "cuda_runtime.h"
#include <iostream>
#include <cuda_fp16.h>
using namespace std;
//Github @jielahou
__global__ void test_ldmatrix_trans(half* read, half* write){
    __align__(16) __shared__ half smem [8*8*4];
    uint32_t reg[4];
    //global --> shared
    //32 threads 每个thread需负责读入8个half 可以用128bit向量指令一口气读入
    const int start_id = threadIdx.x * 8;
    (reinterpret_cast<float4*>(&smem[start_id]))[0] = (reinterpret_cast<float4*>(&read[start_id]))[0];

    //use ldmatrix.trans
    asm("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];\n\t"
        : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
        : "l"(&smem[start_id]));
    
    //reg --> global
    (reinterpret_cast<float*>(&write[start_id + 0]))[0] = (reinterpret_cast<float*>(&reg[0]))[0];
    (reinterpret_cast<float*>(&write[start_id + 2]))[0] = (reinterpret_cast<float*>(&reg[1]))[0];
    (reinterpret_cast<float*>(&write[start_id + 4]))[0] = (reinterpret_cast<float*>(&reg[2]))[0];
    (reinterpret_cast<float*>(&write[start_id + 6]))[0] = (reinterpret_cast<float*>(&reg[3]))[0];

    
}


template<typename T,int N> void init_mem(T (&ptr)[N]){
    for(int i=0;i<N;i++){
        ptr[i] = i;
    }
}



int main(){
    //FP16, (8*8) * 4个
    half host_send[8*8*4];
    half host_receive[8*8*4];
    half* device_read;
    half* device_write;
    init_mem(host_send);

    constexpr int data_size = 8*8*4 * sizeof(half);
    cudaMalloc(&device_read, data_size);
    cudaMalloc(&device_write, data_size);
    
    cudaMemcpy(device_read, host_send, data_size, cudaMemcpyHostToDevice);


    test_ldmatrix_trans<<<1, 32>>>(device_read, device_write);

    cudaDeviceSynchronize();
    cudaMemcpy(host_receive, device_write, data_size, cudaMemcpyDeviceToHost);

    for(int i=0;i<32;i++){
        cout << "thread " << i << " holds: ";
        for(int j=0;j<8;j+=2){
            cout << "(" << j/2 << ")" << " " << __half2float(host_receive[i*8 + j]) << ", "  << __half2float(host_receive[i*8 + (j + 1)]) << ",";
        }
        cout << endl;
    }
    return 0;
}