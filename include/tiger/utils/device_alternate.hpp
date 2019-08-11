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


inline const char* cublasGetErrorString(cublasStatus_t error) {
    switch (error) {
	case CUBLAS_STATUS_SUCCESS:
	    return "CUBLAS_STATUS_SUCCESS";
	case CUBLAS_STATUS_NOT_INITIALIZED:
	    return "CUBLAS_STATUS_NOT_INITIALIZED";
	case CUBLAS_STATUS_ALLOC_FAILED:
	    return "CUBLAS_STATUS_ALLOC_FAILED";
	case CUBLAS_STATUS_INVALID_VALUE:
	    return "CUBLAS_STATUS_INVALID_VALUE";
	case CUBLAS_STATUS_ARCH_MISMATCH:
	    return "CUBLAS_STATUS_ARCH_MISMATCH";
	case CUBLAS_STATUS_MAPPING_ERROR:
	    return "CUBLAS_STATUS_MAPPING_ERROR";
	case CUBLAS_STATUS_EXECUTION_FAILED:
	    return "CUBLAS_STATUS_EXECUTION_FAILED";
	case CUBLAS_STATUS_INTERNAL_ERROR:
	    return "CUBLAS_STATUS_INTERNAL_ERROR";
#if CUDA_VERSION >= 6000
	case CUBLAS_STATUS_NOT_SUPPORTED:
	    return "CUBLAS_STATUS_NOT_SUPPORTED";
#endif
#if CUDA_VERSION >= 6050
	case CUBLAS_STATUS_LICENSE_ERROR:
	    return "CUBLAS_STATUS_LICENSE_ERROR";
#endif
    }
    return "Unknown cublas status";
}

inline const char* curandGetErrorString(curandStatus_t error) {
    switch (error) {
	case CURAND_STATUS_SUCCESS:
	    return "CURAND_STATUS_SUCCESS";
	case CURAND_STATUS_VERSION_MISMATCH:
	    return "CURAND_STATUS_VERSION_MISMATCH";
	case CURAND_STATUS_NOT_INITIALIZED:
	    return "CURAND_STATUS_NOT_INITIALIZED";
	case CURAND_STATUS_ALLOCATION_FAILED:
	    return "CURAND_STATUS_ALLOCATION_FAILED";
	case CURAND_STATUS_TYPE_ERROR:
	    return "CURAND_STATUS_TYPE_ERROR";
	case CURAND_STATUS_OUT_OF_RANGE:
	    return "CURAND_STATUS_OUT_OF_RANGE";
	case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
	    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
	case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
	    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
	case CURAND_STATUS_LAUNCH_FAILURE:
	    return "CURAND_STATUS_LAUNCH_FAILURE";
	case CURAND_STATUS_PREEXISTING_FAILURE:
	    return "CURAND_STATUS_PREEXISTING_FAILURE";
	case CURAND_STATUS_INITIALIZATION_FAILED:
	    return "CURAND_STATUS_INITIALIZATION_FAILED";
	case CURAND_STATUS_ARCH_MISMATCH:
	    return "CURAND_STATUS_ARCH_MISMATCH";
	case CURAND_STATUS_INTERNAL_ERROR:
	    return "CURAND_STATUS_INTERNAL_ERROR";
    }
    return "Unknown curand status";
}
#define CUDA_CHECK(condition)\
    do{\
	cudaError_t error = condition;\
	CHECK_EQ(error, cudaSuccess) << " " << cudaGetErrorString(error);\
    }while(0)
#define CUBLAS_CHECK(condition)\
    do{\
	cublasStatus_t status = condition;\
	CHECK_EQ(status, CUBLAS_STATUS_SUCCESS) << " "\
	<< cublasGetErrorString(status);\
    }while(0)
#define CURAND_CHECK(condition)\
    do{\
	curandStatus_t status = condition; \
	CHECK_EQ(status, CURAND_STATUS_SUCCESS) << " "\
	<< curandGetErrorString(status); \
    }while(0)



const int CUDA_NUM_THREADS = 512;

inline int GET_BLOCKS(const int N){
    return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}





#endif




#endif
