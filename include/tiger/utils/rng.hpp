#ifndef TIGER_UTILS_RNG_HPP
#define TIGER_UTILS_RNG_HPP

#include "tiger/common.hpp"

namespace tiger{

inline rng_t* tiger_rng(){
    return static_cast<rng_t*>(Tiger::rng_stream().generator());
}

}




#endif
