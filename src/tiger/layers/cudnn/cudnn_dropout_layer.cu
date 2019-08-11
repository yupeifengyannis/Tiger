#include "tiger/layers/cudnn/cudnn_dropout_layer.hpp"

namespace tiger{


template <typename Dtype>
void CuDNNDropoutLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){

}

template <typename Dtype>
void CuDNNDropoutLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom){

}


}
