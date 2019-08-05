#include "tiger/layers/pooling_layer.hpp"

namespace tiger{

template <typename Dtype>
void PoolingLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    // TODO()
}

template <typename Dtype>
void PoolingLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>* >& bottom){
    // TODO()
}

template class PoolingLayer<float>;
template class PoolingLayer<double>;


}
