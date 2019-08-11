#ifndef TIGER_UTILS_MATH_FUNCTIONS
#define TIGER_UTILS_MATH_FUNCTIONS
#include <string.h>
#include <cblas.h>
#include "tiger/common.hpp"
namespace tiger{

inline void tiger_memset(const size_t N, const int alpha, void* X){
    memset(X, alpha, N);
}

template <typename Dtype>
inline void tiger_set(const int N, const Dtype data, void* X){
    Dtype* Y = static_cast<Dtype*>(X);
    for(int i = 0; i < N; i++){
	Y[i] = data;
    }
}

// 伯努利分布其实就是0-1分布
template <typename Dtype>
void tiger_rng_bernoulli(const int n, const Dtype p, int* r);

#ifndef CPU_ONLY
void tiger_gpu_memcpy(const size_t N, const void* src, void* des);
inline void tiger_gpu_memset(const size_t N, const int alpha, void* X){
    CUDA_CHECK(cudaMemset(X, alpha, N));
}

// 使用gpu进行矩阵矩阵乘法运算
template <typename Dtype>
void tiger_gpu_gemm(const CBLAS_TRANSPOSE TransA,
	const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
	const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
	Dtype* C);

// 使用gpu进行矩阵向量乘法运算
template <typename Dtype>
void tiger_gpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* C);

// 线性变换
template <typename Dtype>
void tiger_gpu_axpy(const int N, const Dtype alpha, const Dtype* X, Dtype* Y);

template <typename Dtype>
void tiger_gpu_axpby(const int N, const Dtype alpha, const Dtype* X, const Dtype beta, Dtype* Y);

void tiger_gpu_rng_uniform(const int n, unsigned int* r);

template <typename Dtype>
void tiger_gpu_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r);

template <typename Dtype>
void tiger_gpu_scal(const int N, const Dtype alpha, Dtype* X);

template <typename Dtype>
void tiger_gpu_add_scalar(const int N, const Dtype alpha, Dtype* Y);


#endif

}

#endif
