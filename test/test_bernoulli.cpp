#include <vector>
#include "tiger/blob.hpp"
#include "tiger/utils/math_function.hpp"

using namespace tiger;

template <typename Dtype>
void test_rng_bernolli(){
    int data[8];
    tiger::tiger_rng_bernoulli(8, Dtype(1), data);
    for(int i = 0; i < 8; i++){
	std::cout << data[i] << " ";
    }
    std::cout << std::endl;
}

void test_gpu_rng_bernolli(){
    unsigned int  data[200];
    tiger::tiger_gpu_rng_uniform(200, data);
    for(int i = 0; i < 200; i++){
	std::cout << data[i] << " ";
    }
    std::cout << std::endl;

}

int main(){
    test_gpu_rng_bernolli();
}

