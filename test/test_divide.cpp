#include <glog/logging.h>
#include <cmath>

void ceil_divide(const int numerator, const int denominator){
    LOG(INFO) << "ceil divide ";
    LOG(INFO) << static_cast<int>(static_cast<float>(ceil(numerator * 1.0 / denominator)));
}

void floor_divide(const int numerator, const int denominator){
    LOG(INFO) << "floor divide ";
    LOG(INFO) << static_cast<int>(floor(static_cast<float>(numerator * 1.0 / denominator)));
}

void test_float_equal(float lhs, float rhs){
    if(lhs == rhs){
	LOG(INFO) << "lhs == rhs";
    }
    else{
	LOG(INFO) << "lhs != rhs";
    }
}

int main(){
    ceil_divide(10, 3);
    floor_divide(10, 3);
    test_float_equal(float(1), double(1));
}
