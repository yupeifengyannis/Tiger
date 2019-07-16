#ifndef TIGER_SYCEDMEM_HPP
#define TIGER_SYCEDMEM_HPP
#include <cstdlib>
#include "tiger/common.hpp"


namespace tiger{

inline void tiger_malloc_host(void** ptr, size_t size, bool* use_cuda){
#ifndef CPU_ONLY
    if(Tiger::mode() == Tiger::GPU){
	CUDA_CHECK(cudaMallocHost(ptr, size));
	*use_cuda = true;
	return;
    }
#endif
    *ptr = malloc(size);
    *use_cuda = false;
    CHECK(*ptr) << "malloc error";
}

inline void tiger_free_host(void* ptr, bool use_cuda){
#ifndef CPU_ONLY
    if(use_cuda){
	CUDA_CHECK(cudaFreeHost(ptr));
	return;
    }
#endif
    free(ptr);
}

class SyncedMemory{
public:
    SyncedMemory();
    explicit SyncedMemory(size_t size);
    SyncedMemory(const SyncedMemory&) = delete;
    SyncedMemory& operator=(const SyncedMemory&) = delete;
    ~SyncedMemory();
    const void* cpu_data();
    void set_cpu_data(void* data);
    const void* gpu_data();
    /// \brief 是否需要保证data的size和之前的cpu_ptr_的size要一样
    void set_gpu_data(void* data);
    void* mutable_cpu_data();
    void* mutable_gpu_data();
    enum SyncedHead{
	UNINTIALIZED,
	HEAD_AT_CPU,
	HEAD_AT_GPU,
	SYNCED
    };
    SyncedHead head() const{
	return head_;
    }
    size_t size() const{
	return size_;
    }

#ifndef CPU_ONLY
    void async_gpu_push(const cudaStream_t& stream);
#endif
private:
    void check_device();
    void to_cpu();
    void to_gpu();
    void* cpu_ptr_;
    void* gpu_ptr_;
    size_t size_;
    SyncedHead head_;
    bool own_cpu_data_;
    bool cpu_malloc_use_cuda_;
    bool own_gpu_data_;
    int device_;

};
}
#endif
