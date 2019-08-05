#include "tiger/layers/cudnn_relu_layer.hpp"

namespace tiger{
template <typename Dtype>
void CuDNNReLULayer<Dtype>::layer_setup(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    ReLULayer<Dtype>::layer_setup(bottom, top);
    CUDNN_CHECK(cudnnCreate(&handle_));
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
    cudnn::createTensor4dDesc<Dtype>(&top_desc_);
    cudnn::createActivationDescriptor<Dtype>(&activ_desc_,
	    CUDNN_ACTIVATION_RELU);
    handles_setup_ = true;
}

template <typename Dtype>
void CuDNNReLULayer<Dtype>::reshape(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    CHECK_EQ(bottom[0]->num_axes(), 4) << 
	"bottom data must has 4 axes";
    ReLULayer<Dtype>::reshape(bottom, top);
    const int N = bottom[0]->num();
    const int C = bottom[0]->channels();
    const int H = bottom[0]->height();
    const int W = bottom[0]->width();
    cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, N, C, H, W);
    cudnn::setTensor4dDesc<Dtype>(&top_desc_, N, C, H, W);
}


template <typename Dtype>
CuDNNReLULayer<Dtype>::~CuDNNReLULayer(){
    if(!handles_setup_){
	return;
    }
    cudnnDestroyTensorDescriptor(this->bottom_desc_);
    cudnnDestroyTensorDescriptor(this->top_desc_);
    cudnnDestroy(this->handle_);
}



template class CuDNNReLULayer<float>;
template class CuDNNReLULayer<double>;

}

