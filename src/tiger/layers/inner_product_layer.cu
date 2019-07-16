#include "tiger/layers/inner_product_layer.hpp"

namespace tiger{

template <typename Dtype>
void InnerProductLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){

}

template <typename Dtype>
void InnerProductLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>* >& top,
	const std::vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom){

}

template class InnerProductLayer<float>;
template class InnerProductLayer<double>;



}

