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
#endif
	    break;
	case HEAD_AT_CPU:
	case SYNCED:
	    break;
    }
}

void SyncedMemory::to_gpu(){
    check_device();
#ifndef CPU_ONLY
    switch(head_){
    case UNINTIALIZED:
	CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
	tiger_gpu_memset(size_, 0, gpu_ptr_);
	head_ = HEAD_AT_GPU;
	own_gpu_data_ = true;
	break;
    case HEAD_AT_CPU:
	if(gpu_ptr_ == NULL){
	    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
	    own_gpu_data_ = true;
	}
	tiger_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
	head_ = SYNCED;
	break;
    case HEAD_AT_GPU:
    case SYNCED:
	break;
    }
#endif
}

const void* SyncedMemory::cpu_data(){
    check_device();
    to_cpu();
    return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data){
    check_device();
    CHECK(data);
    if(own_cpu_data_){
	tiger_free_host(cpu_ptr_, cpu_malloc_use_cuda_);
    }
    cpu_ptr_ = data;
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data(){
    check_device();
#ifndef CPU_ONLY
    to_gpu();
    return (const void*)gpu_ptr_;
#endif
    return NULL;
}

void SyncedMemory::set_gpu_data(void* data){
    check_device();
#ifndef CPU_ONLY
    CHECK(data);
    if(own_gpu_data_){
	CUDA_CHECK(cudaFree(gpu_ptr_));
    }
    gpu_ptr_ = data;
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = false;
#endif
}

void* SyncedMemory::mutable_cpu_data(){
    check_device();
    to_cpu();
    head_ = HEAD_AT_CPU;
    return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data(){
    check_device();
#ifndef CPU_ONLY
    to_gpu();
    head_ = HEAD_AT_GPU;
    return gpu_ptr_;
#endif
    return NULL;
}

#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream){
    check_device();
    CHECK(head_ = HEAD_AT_GPU);
    if(gpu_ptr_ == NULL){
	CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
	own_gpu_data_ = true;
    }
    const cudaMemcpyKind put = cudaMemcpyHostToDevice;
    CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
    head_ = SYNCED;
}
#endif

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

