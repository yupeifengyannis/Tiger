
#include "tiger/utils/math_function.hpp"
#include "tiger/common.hpp"
#include "tiger/utils/device_alternate.hpp"

namespace tiger{
void tiger_gpu_memcpy(const size_t N, const void* src, void* des){
    if(src != des){
	CUDA_CHECK(cudaMemcpy(des, src, N, cudaMemcpyDefault));
    }
}

template <>
void tiger_gpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
	const float alpha, const float* A, const float* B, const float beta,
	float* C){
    
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t cuTransA = 
	(TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB = 
	(TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    // cublasSgemm是float的的矩阵乘法
    cublasHandle_t cublas_handle; 
    CUBLAS_CHECK(cublasSgemm(cublas_handle, cuTransA, cuTransB,
		M, N, K, &alpha, A, lda, B, ldb, &beta, C, N)); 
}






}

