#include <cuda_runtime.h>
#include <cuda.h>
#include <cblas.h>
#include <iostream>
#include "tiger/utils/math_function.hpp"

template <typename Dtype>
__global__ void show_data(Dtype* data, int N){
    int i = threadIdx.x;
    if(i < N){
	printf("%f\n", data[i]);
    }
}
template <typename Dtype>
void test_tiger_gpu_gemm_notrans(){
    // create host memory space and initialize
    int M = 5;
    int K = 4;
    int N = 3;
    Dtype* A = new Dtype[M * K];
    Dtype* B = new Dtype[K * N];
    Dtype* C = new Dtype[M * N];
    for(int i = 0; i < M * K; i++){
	A[i] = i + 1;
    }
    for(int i = 0; i < K * N; i++){
	B[i] = i + 1;
    }
    for(int i = 0; i < M * N; i++){
	C[i] = 0;
    }
    // create device memory space and initialize
    Dtype* d_A;
    Dtype* d_B;
    Dtype* d_C;
    cudaMalloc((Dtype**)&d_A, M * K * sizeof(Dtype));
    cudaMalloc((Dtype**)&d_B, K * N * sizeof(Dtype));
    cudaMalloc((Dtype**)&d_C, M * N * sizeof(Dtype));
    cudaMemcpy(d_A, A, M * K * sizeof(Dtype), cudaMemcpyDefault);
    cudaMemcpy(d_B, B, K * N * sizeof(Dtype), cudaMemcpyDefault);
    cudaMemcpy(d_C, C, M * N * sizeof(Dtype), cudaMemcpyDefault);
    tiger::tiger_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M, N, K,
        Dtype(1), d_A, d_B, Dtype(1), d_C);
    // transfer data from device to host
    cudaMemcpy(C, d_C, M * N * sizeof(Dtype), cudaMemcpyDefault);
    
    for(int i = 0; i < M; i++){
	for(int j = 0; j < N; j++){
	    std::cout << C[i * N + j] << " ";	    
	}
	std::cout << std::endl;
    }
    std::cout << std::endl;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(A);
    free(B);
    free(C);
}
template <typename Dtype>
void test_tiger_gpu_gemm_trans(){
    int M = 3;
    int N = 2;
    Dtype* A = new Dtype[M * N];
    Dtype* B = new Dtype[M * N];
    Dtype* C = new Dtype[M * M];
    for(int i = 0; i < M * M; i++){
	C[i] = 0;
    }
    for(int i = 0; i < M * N; i++){
	A[i] = i;
	B[i] = i;
    }
    // crate device memory space and initialize
    Dtype* d_A;
    Dtype* d_B;
    Dtype* d_C;
    cudaMalloc((Dtype**)&d_A, M * N * sizeof(Dtype));
    cudaMalloc((Dtype**)&d_B, M * N * sizeof(Dtype));
    cudaMalloc((Dtype**)&d_C, M * M * sizeof(Dtype));
    cudaMemcpy(d_A, A, M * N * sizeof(Dtype), cudaMemcpyDefault);
    cudaMemcpy(d_B, B, M * N * sizeof(Dtype), cudaMemcpyDefault);
    // 记得要给d_C矩阵进行初始化
    cudaMemcpy(d_C, C, M * M * sizeof(Dtype), cudaMemcpyDefault);
    tiger::tiger_gpu_gemm<Dtype>(CblasNoTrans, CblasTrans, 3, 3, 2,
	    Dtype(1), d_A, d_B, Dtype(1), d_C);
    //transfer data from device to host
    cudaMemcpy(C, d_C, M * M * sizeof(Dtype), cudaMemcpyDefault);
    for(int i = 0; i < M; i++){
	for(int j = 0; j < M; j++){
	    std::cout << C[i * M + j] << " ";
	}
	std::cout << std::endl;
    }
    std::cout << std::endl;

    // free host memroy sapce and device memroy space
    free(A);
    free(B);
    free(C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

template <typename Dtype>
void test_tiger_gpu_gemv_notrans(){
    int M = 3;
    int N = 2;
    Dtype* A = new Dtype[M * N];
    Dtype* x = new Dtype[N];
    Dtype* y = new Dtype[M];
    for(int i = 0; i < M * N; i++){
	A[i] = i;
    }
    for(int i = 0; i < N; i++){
	x[i] = i;
    }
    for(int i = 0; i < M; i++){
	y[i] = 0;
    }
    //create device memory space and initialize
    Dtype* d_A;
    Dtype* d_x;
    Dtype* d_y;
    cudaMalloc((Dtype**)&d_A, M * N * sizeof(Dtype));
    cudaMalloc((Dtype**)&d_x, N * sizeof(Dtype));
    cudaMalloc((Dtype**)&d_y, M * sizeof(Dtype));
    cudaMemcpy(d_A, A, M * N * sizeof(Dtype), cudaMemcpyDefault);
    cudaMemcpy(d_x, x, N * sizeof(Dtype), cudaMemcpyDefault);
    cudaMemcpy(d_y, y, M * sizeof(Dtype), cudaMemcpyDefault);
    tiger::tiger_gpu_gemv<Dtype>(CblasNoTrans, M, N, Dtype(1), d_A, d_x, Dtype(1), d_y);
    //transfer data from device to host
    cudaMemcpy(y, d_y, M * sizeof(Dtype), cudaMemcpyDefault);
    for(int i = 0; i < M; i++){
	std::cout << y[i] << " ";
    }
    std::cout << std::endl;
    free(A);
    free(x);
    free(y);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

}


template <typename Dtype>
void test_tiger_gpu_gemv_trans(){
    int M = 3;
    int N = 2;
    Dtype* A = new Dtype[M * N];
    Dtype* x = new Dtype[M];
    Dtype* y = new Dtype[N];
    for(int i = 0; i < M * N; i++){
	A[i] = i;
    }
    for(int i = 0; i < M; i++){
	x[i] = i;
    }
    for(int i = 0; i < N; i++){
	y[i] = 0;
    }
    //create device memory space and initialize
    Dtype* d_A;
    Dtype* d_x;
    Dtype* d_y;
    cudaMalloc((Dtype**)&d_A, M * N * sizeof(Dtype));
    cudaMalloc((Dtype**)&d_x, M * sizeof(Dtype));
    cudaMalloc((Dtype**)&d_y, N * sizeof(Dtype));
    cudaMemcpy(d_A, A, M * N * sizeof(Dtype), cudaMemcpyDefault);
    cudaMemcpy(d_x, x, M * sizeof(Dtype), cudaMemcpyDefault);
    cudaMemcpy(d_y, y, N * sizeof(Dtype), cudaMemcpyDefault);
    tiger::tiger_gpu_gemv<Dtype>(CblasTrans, M, N, Dtype(1), d_A, d_x, Dtype(1), d_y);
    //transfer data from device to host
    cudaMemcpy(y, d_y, N * sizeof(Dtype), cudaMemcpyDefault);
    for(int i = 0; i < N; i++){
	std::cout << y[i] << " ";
    }
    std::cout << std::endl;
    free(A);
    free(x);
    free(y);
    cudaFree(d_A);
    cudaFree(d_x);
    cudaFree(d_y);

}

template <typename Dtype>
void test_tiger_gpu_axpy(){
    int N = 10;
    Dtype alpha = 2;
    Dtype* X = new Dtype[N];
    Dtype* Y = new Dtype[N];
    for(int i = 0; i < N; i++){
	X[i] = i;
	Y[i] = 0;
    }
    Dtype* d_X;
    Dtype* d_Y;
    cudaMalloc((Dtype**)&d_X, N * sizeof(Dtype));
    cudaMalloc((Dtype**)&d_Y, N * sizeof(Dtype));
    cudaMemcpy(d_X, X, N * sizeof(Dtype), cudaMemcpyDefault);
    cudaMemcpy(d_Y, Y, N * sizeof(Dtype), cudaMemcpyDefault);
    tiger::tiger_gpu_axpy(N, alpha, d_X, d_Y);
    cudaMemcpy(Y, d_Y, N * sizeof(Dtype), cudaMemcpyDefault);
    for(int i = 0; i < N; i++){
	std::cout << Y[i] << std::endl;
    }
    
    free(X);
    free(Y);
    cudaFree(d_X);
    cudaFree(d_Y);

}



int main(){
    std::cout << "float no transpose no transpose" << std::endl;
    test_tiger_gpu_gemm_notrans<float>();
    std::cout << "double no transpose no transpose " << std::endl;
    test_tiger_gpu_gemm_notrans<double>();
    std::cout << "float no transpose transpose " << std::endl;
    test_tiger_gpu_gemm_trans<float>();
    std::cout << "double no transpose transpose " << std::endl;
    test_tiger_gpu_gemm_trans<double>();
    
    std::cout << "test sgemm function " << std::endl;
    std::cout << "float no transpose" << std::endl;
    test_tiger_gpu_gemv_notrans<float>();
    
    std::cout << "double no transpose " << std::endl;
    test_tiger_gpu_gemv_notrans<double>();

    std::cout << "float transpose " << std::endl;
    test_tiger_gpu_gemv_trans<float>();
    std::cout << "double transpose " << std::endl;
    test_tiger_gpu_gemv_trans<double>();
    
    std::cout << "test axpy function " << std::endl;
    test_tiger_gpu_axpy<float>();
    test_tiger_gpu_axpy<double>();
}






