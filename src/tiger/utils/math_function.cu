
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
    cublasCreate(&cublas_handle);
    CUBLAS_CHECK(cublasSgemm(cublas_handle, cuTransB, cuTransA,
		N, M, K, &alpha, B, ldb, A, lda, &beta, C, N)); 
}

template <>
void tiger_gpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
	const double alpha, const double* A, const double* B, const double beta,
	double* C){
    
    int lda = (TransA == CblasNoTrans) ? K : M;
    int ldb = (TransB == CblasNoTrans) ? N : K;
    cublasOperation_t cuTransA = 
	(TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    cublasOperation_t cuTransB = 
	(TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
    // cublasDgemm是double的的矩阵乘法
    cublasHandle_t cublas_handle; 
    cublasCreate(&cublas_handle);
    CUBLAS_CHECK(cublasDgemm(cublas_handle, cuTransB, cuTransA,
		N, M, K, &alpha, B, ldb, A, lda, &beta, C, N)); 
}

template <>
void tiger_gpu_gemv<float>(const CBLAS_TRANSPOSE TransA,
	const int M, const int N, const float alpha, const float* A, const float* x,
	const float beta, float* y){
    cublasOperation_t cuTransA = 
	(TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    CUBLAS_CHECK(cublasSgemv(cublas_handle, cuTransA, N, M, &alpha,
		A, N, x, 1, &beta, y, 1));

}
template <>
void tiger_gpu_gemv<double>(const CBLAS_TRANSPOSE TransA,
	const int M, const int N, const double alpha, const double* A, const double* x,
	const double beta, double* y){
    cublasOperation_t cuTransA = 
	(TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    CUBLAS_CHECK(cublasDgemv(cublas_handle, cuTransA, N, M, &alpha,
		A, N, x, 1, &beta, y, 1));
}

template <>
void tiger_gpu_axpy<float>(const int N, const float alpha, const float* X, float* Y){
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    CUBLAS_CHECK(cublasSaxpy(cublas_handle, N, &alpha, X, 1, Y, 1));
}

template <>
void tiger_gpu_axpy<double>(const int N, const double alpha, const double* X, double* Y){
    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    CUBLAS_CHECK(cublasDaxpy(cublas_handle, N, &alpha, X, 1, Y, 1));
}
}

