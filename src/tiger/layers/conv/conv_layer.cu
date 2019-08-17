#include "tiger/layers/conv/conv_layer.hpp"

namespace tiger{

template <typename Dtype>
void ConvolutionLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    //TODO
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>* >& bottom){
    //TODO
}

template class ConvolutionLayer<float>;
template class ConvolutionLayer<double>;


}
