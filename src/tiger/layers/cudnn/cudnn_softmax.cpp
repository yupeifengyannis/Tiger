#include "tiger/layers/cudnn/cudnn_softmax.hpp"

namespace tiger{

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::layer_setup(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    SoftmaxLayer<Dtype>::layer_setup(bottom, top);
    CUDNN_CHECK(cudnnCreate(&handle_));
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
    cudnn::createTensor4dDesc<Dtype>(&top_desc_);
    handles_setup_ = true;
}

template <typename Dtype>
void CuDNNSoftmaxLayer<Dtype>::reshape(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    SoftmaxLayer<Dtype>::reshape(bottom, top);
    int N = this->outer_num_;
    int K = bottom[0]->shape(this->softmax_axis_);
    int H = this->inner_num_;
    int W = 1;
    cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, N, K, H, W);
    cudnn::setTensor4dDesc<Dtype>(&top_desc_, N, K, H, W);
}

template <typename Dtype>
CuDNNSoftmaxLayer<Dtype>::~CuDNNSoftmaxLayer(){
    if(handles_setup_){
	cudnnDestroyTensorDescriptor(bottom_desc_);
	cudnnDestroyTensorDescriptor(top_desc_);
	cudnnDestroy(handle_);
    }
}

template class CuDNNSoftmaxLayer<float>;
template class CuDNNSoftmaxLayer<double>;

}
