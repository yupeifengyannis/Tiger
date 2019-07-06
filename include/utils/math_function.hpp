#ifndef UTILS_MATH_FUNCTIONS
#define UTILS_MATH_FUNCTIONS

#include <string.h>
#include "common.hpp"
namespace tiger{

inline void tiger_memset(const size_t N, const int alpha, void* X){
    memset(X, alpha, N);
}

#ifndef CPU_ONLY
void tiger_gpu_memcpy(const size_t N, const void* src, void* des);

#endif

}

#endif
