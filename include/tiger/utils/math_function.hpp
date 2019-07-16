#ifndef TIGER_UTILS_MATH_FUNCTIONS
#define TIGER_UTILS_MATH_FUNCTIONS
#include "tiger/common.hpp"
#include <string.h>
namespace tiger{

inline void tiger_memset(const size_t N, const int alpha, void* X){
    memset(X, alpha, N);
}

#ifndef CPU_ONLY
void tiger_gpu_memcpy(const size_t N, const void* src, void* des);
inline void tiger_gpu_memset(const size_t N, const int alpha, void* X){
    CUDA_CHECK(cudaMemset(X, alpha, N));
}

#endif

}

#endif
