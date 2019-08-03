#include <sys/types.h>
#include <unistd.h>
#include <ctime>

#include <memory>
#include "tiger/common.hpp"

namespace tiger{

std::shared_ptr<Tiger> instance_;

int64_t cluster_seed_gen(void){
    int64_t pid = getpid();
    int64_t s = time(NULL);
    int64_t seed = std::abs(((s * 181) * (pid - 83) * 359) % 104729);
    return seed;

}

Generator::Generator() : 
    rng_(new rng_t(cluster_seed_gen())){
    }
Generator::Generator(unsigned int seed) : 
    rng_(new rng_t(seed)){
    }
rng_t* Generator::rng(){
    return rng_.get();
}

Tiger::RNG::RNG() : 
    generator_(new Generator()){
    }
Tiger::RNG::RNG(unsigned int seed) : 
    generator_(new Generator(seed)){

    }
Tiger::RNG& Tiger::RNG::operator=(const RNG& other){
    generator_ = other.generator_;
    return *this;
}


void* Tiger::RNG::generator(){
    return static_cast<void*>(generator_->rng());
}

Tiger::Tiger() : 
    mode_(Tiger::CPU){

    }

Tiger::~Tiger(){

}

Tiger& Tiger::get(){
    if(!instance_.get()){
	instance_.reset(new Tiger());
    }
    return *instance_;
}
}
