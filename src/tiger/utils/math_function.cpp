#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>
#include "tiger/common.hpp"
#include "tiger/utils/math_function.hpp"
#include "tiger/utils/rng.hpp"

namespace tiger{

template <typename Dtype>
void tiger_rng_bernoulli(const int n, const Dtype p, int* r){
    CHECK_GE(n, 0);
    CHECK(r);
    CHECK_GE(p, 0);
    CHECK_LE(p, 1);
    boost::bernoulli_distribution<Dtype> random_distribution(p);
    boost::variate_generator<tiger::rng_t*, boost::bernoulli_distribution<Dtype> >
	variate_generator(tiger_rng(), random_distribution);
    for(int i = 0; i < n; i++){
	r[i] = variate_generator();
    }
} 

template 
void tiger_rng_bernoulli<float>(const int n, const float p, int* r);
template
void tiger_rng_bernoulli<double>(const int n, const double p, int* r);


}
