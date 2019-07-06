#ifndef SYCEDMEM_HPP
#define SYCEDMEM_HPP
#include <cstdlib>
#include "common.hpp"


namespace tiger{

inline void tiget_malloc_host(void** ptr, size_t size, bool* use_cuda){
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

}







#endif
