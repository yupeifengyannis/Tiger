#include "utils/math_function.hpp"
#include "syncedmem.hpp"

namespace tiger{

SyncedMemory::SyncedMemory() : 
    cpu_ptr_(NULL),
    gpu_ptr_(NULL),
    size_(0),
    head_(SyncedHead::UNINTIALIZED),
    own_cpu_data_(false),
    cpu_malloc_use_cuda_(false),
    own_gpu_data_(false){
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDevice(&device_));
#endif
    }
SyncedMemory::SyncedMemory(size_t size) : 
    cpu_ptr_(NULL),
    gpu_ptr_(NULL),
    size_(size),
    head_(SyncedHead::UNINTIALIZED),
    own_cpu_data_(false),
    cpu_malloc_use_cuda_(false),
    own_gpu_data_(false){
#ifndef CPU_ONLY
    CUDA_CHECK(cudaGetDevice(&device_));
#endif
    }

SyncedMemory::~SyncedMemory(){
    check_device();
    if(cpu_ptr_ && own_cpu_data_){
	tiger_free_host(cpu_ptr_, cpu_malloc_use_cuda_);
    }

#ifndef CPU_ONLY
    if(gpu_ptr_ && own_gpu_data_){
	CUDA_CHECK(cudaFree(gpu_ptr_));
    }
#endif
}

void SyncedMemory::to_cpu(){
    check_device();
    switch(head_){
	case UNINTIALIZED:
	    tiger_malloc_host(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
	    tiger_memset(size_, 0, cpu_ptr_);
	    head_ = HEAD_AT_CPU;
	    own_cpu_data_ = true;
	    break;
	case HEAD_AT_GPU:
#ifndef CPU_ONLY
	    if(cpu_ptr_ == NULL){
		tiger_malloc_host(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
		own_cpu_data_ = true;
	    }
	    tiger_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
	    head_ = SYNCED;
	    break;
	case HEAD_AT_CPU:
	case SYNCED:
	    break;

#endif
    }
}

void SyncedMemory::check_device(){
#ifndef CPU_ONLY
    int device;
    cudaGetDevice(&device);
    CHECK(device == device_);
    if(gpu_ptr_ && own_gpu_data_){
	cudaPointerAttributes attributes;
	CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
	CHECK(attributes.device == device_);
    }
#endif
}




}

