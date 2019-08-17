#include "tiger/layers/conv/conv_layer.hpp"

namespace tiger{

template <typename Dtype>
void ConvolutionLayer<Dtype>::compute_output_shape(){
    //TODO
}

template <typename Dtype>
void ConvolutionLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>* >& bottom,
    const vector<Blob<Dtype>* >& top){
    // TODO
}
template <typename Dtype>
void ConvolutionLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>* >& bottom){
    // TODO
}

template class ConvolutionLayer<float>;
template class ConvolutionLayer<double>;

}
