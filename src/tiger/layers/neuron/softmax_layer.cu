#include "tiger/layers/neuron/softmax_layer.hpp"
#include "tiger/utils/device_alternate.hpp"

namespace tiger{

template <typename Dtype>
void SoftmaxLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    //TODO
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>* >& bottom){
    //TODO
}

template class SoftmaxLayer<float>;
template class SoftmaxLayer<double>;



}
