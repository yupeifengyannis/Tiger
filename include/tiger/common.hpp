#ifndef TIGER_COMMON_HPP
#define TIGER_COMMON_HPP

#include <string>
#include <memory>
#include <random>
#include "tiger/utils/device_alternate.hpp"

using namespace std;
// TODO(Tiger单例类还有好多东西要去实现)

namespace tiger{

typedef std::mt19937 rng_t;

class Generator{
public:
    Generator();
    explicit Generator(unsigned int seed);
    rng_t* rng();
private:
    std::shared_ptr<rng_t> rng_;
};


class Tiger{
public:
    ~Tiger();
    static Tiger& get();
    enum Brew{
	CPU,
	GPU
    };
    class RNG{
    public:
	RNG();
	explicit RNG(unsigned int seed);
	explicit RNG(const RNG&);
	RNG& operator=(const RNG&);
	// TODO(为什么是void*的返回值)
	void* generator();
    private:
	std::shared_ptr<Generator> generator_;
    };


    inline static RNG& rng_stream(){
	if(!get().random_generator_){
	    get().random_generator_.reset(new RNG());
	}
	return *(get().random_generator_);
    } 
#ifndef CPU_ONLY
    inline static cublasHandle_t cublas_handle(){
	return get().cublas_handle_;
    }
    inline static curandGenerator_t curand_generator(){
	return get().curand_generator_;
    }
#endif

    inline static Brew mode(){
	return get().mode_;
    }
protected:
    Brew mode_;
    std::shared_ptr<RNG> random_generator_;
#ifndef CPU_ONLY
    cublasHandle_t cublas_handle_;
    curandGenerator_t curand_generator_;
#endif
private:
    Tiger();

};
}























#endif
