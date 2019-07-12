#include <glog/logging.h>
#include "common.hpp"


using namespace tiger;

int main(){
    Generator rng_gen;
    rng_t* rng_ptr = rng_gen.rng();
    LOG(INFO) << rng_ptr->max();
}
