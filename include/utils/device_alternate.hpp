#ifndef UITLS_DEVICE_ALTERNATE_HPP
#define UITLS_DEVICE_ALTERNATE_HPP

#include <glog/logging.h>

#ifdef CPU_ONLY

#else
//#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(condition)\
    do{\
	cudaError_t error = condition;\
	CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error);\
    }while(0)

#endif



#endif
