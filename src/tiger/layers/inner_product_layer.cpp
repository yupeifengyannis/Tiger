#include "tiger/layers/inner_product_layer.hpp"

namespace tiger{

template <typename Dtype>
void InnerProductLayer<Dtype>::layer_setup(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){

}

template <typename Dtype>
void InnerProductLayer<Dtype>::reshape(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){

}

template <typename Dtype>
void InnerProductLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){

}

template <typename Dtype>
void InnerProductLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom){

}



}
