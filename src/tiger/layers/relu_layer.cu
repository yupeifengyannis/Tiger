#include <algorithm>
#include "tiger/layers/relu_layer.hpp"

namespace tiger{

template <typename Dtype>
void ReLULayer<Dtype>::forward_gpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    // TODO()
}

template <typename Dtype>
void ReLULayer<Dtype>::backward_gpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>* >& bottom){
    //TODO()
}

template class ReLULayer<float>;
template class ReLULayer<double>;

}
