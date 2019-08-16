#include "tiger/layers/cudnn/cudnn_pooling_layer.hpp"
#include "tiger/utils/cudnn.hpp"

namespace tiger{
template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::layer_setup(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    // 第一步先调用父类的layer_setup函数
    PoolingLayer<Dtype>::layer_setup(bottom, top);
    CUDNN_CHECK(cudnnCreate(&handle_));
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
    cudnn::createTensor4dDesc<Dtype>(&top_desc_);
    cudnn::createPoolingDesc<Dtype>(&pooling_desc_,
	    this->layer_param_.pooling_param().method(),
	    &mode_,
	    this->kernel_h_,
	    this->kernel_w_,
	    this->pad_h_,
	    this->pad_w_,
	    this->stride_h_,
	    this->stride_w_);
    handles_setup_ = true;
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::reshape(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    PoolingLayer<Dtype>::reshape(bottom, top);
    cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, bottom[0]->num(),
	    this->channels_,
	    this->height_,
	    this->width_);
    cudnn::setTensor4dDesc<Dtype>(&top_desc_, bottom[0]->num(),
	    this->channels_,
	    this->pooled_height_,
	    this->pooled_width_);
}

template <typename Dtype>
CuDNNPoolingLayer<Dtype>::~CuDNNPoolingLayer(){
    if(!handles_setup_){
	return;
    }
    cudnnDestroyTensorDescriptor(bottom_desc_);
    cudnnDestroyTensorDescriptor(top_desc_);
    cudnnDestroyPoolingDescriptor(pooling_desc_);
    cudnnDestroy(handle_);
}

template class CuDNNPoolingLayer<float>;
template class CuDNNPoolingLayer<double>;
}
