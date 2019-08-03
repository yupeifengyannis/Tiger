#include <cuda_runtime.h>
#include <cuda.h>
#include <iostream>
#include "tiger/utils/math_function.hpp"

__global__ void show_data(float* data, int N){
    int i = threadIdx.x;
    if(i < N){
	printf("%f\n", data[i]);
    }
}

void test_tiger_gpu_gemm(){
    // create host memory space and initialize
    int M = 5;
    int K = 4;
    int N = 3;
    float* A = new float[M * K];
    float* B = new float[K * N];
    float* C = new float[M * N];
    for(int i = 0; i < M * K; i++){
	A[i] = i + 1;
    }
    for(int i = 0; i < K * N; i++){
	B[i] = i + 1;
    }
    

    // create device memory space and initialize
    float* d_A;
    float* d_B;
    float* d_C;
    cudaMalloc((float**)&d_A, M * K * sizeof(float));
    cudaMalloc((float**)&d_B, K * N * sizeof(float));
    cudaMalloc((float**)&d_C, M * N * sizeof(float));
    cudaMemcpy(d_A, A, M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, K * N * sizeof(float), cudaMemcpyHostToDevice);
    
    show_data<<<1, 32>>>(d_A, M * K);
    show_data<<<1, 32>>>(d_B, K * N);

    float alpha = 1;
    float beta = 1;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    cublasSgemm_v2(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N,
	    M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
    //tiger::tiger_gpu_gemm<float>(CblasNoTrans, CblasNoTrans, M, N, K,
    //    float(1), d_A, d_B, float(1), d_C);
    // transfer data from device to host
    cudaMemcpy(C, d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    
    for(int i = 0; i < M; i++){
	for(int j = 0; j < N; j++){
	    std::cout << C[i * N + j] << " ";	    
	}
	std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main(){
    test_tiger_gpu_gemm();
}
