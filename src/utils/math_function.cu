#include "utils/math_function.hpp"

namespace tiger{

void tiger_gpu_memcpy(const size_t N, const void* src, void* des){
    if(src != des){
	CUDA_CHECK(cudaMemcpy(des, src, N, cudaMemcpyDefault));
    }
}

}
