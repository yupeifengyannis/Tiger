#ifndef TIGER_UITLS_DEVICE_ALTERNATE_HPP
#define TIGER_UITLS_DEVICE_ALTERNATE_HPP

#include <glog/logging.h>

#ifdef CPU_ONLY

#else
//#include <cuda.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda.h>
#include <cublas_v2.h>
#include <driver_types.h>
#define CUDA_CHECK(condition)\
    do{\
	cudaError_t error = condition;\
	CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error);\
    }while(0)


const int CUDA_NUM_THREADS = 512;

inline int GET_BLOCKS(const int N){
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}



#endif




#endif
