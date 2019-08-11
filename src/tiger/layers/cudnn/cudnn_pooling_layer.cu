#include "tiger/layers/cudnn/cudnn_pooling_layer.hpp"

namespace tiger{

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    CUDNN_CHECK(cudnnPoolingForward(handle_,
		pooling_desc_,
		cudnn::dataType<Dtype>::one,
		bottom_desc_,
		bottom_data,
		cudnn::dataType<Dtype>::zero,
		top_desc_,
		top_data));
}

template <typename Dtype>
void CuDNNPoolingLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>* >&bottom){
    if(!propagate_down[0]){
	return;
    }
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* bottom_data = top[0]->gpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    CUDNN_CHECK(cudnnPoolingBackward(handle_,
		pooling_desc_,
		cudnn::dataType<Dtype>::one,
		top_desc_,
		top_data,
		top_desc_,
		top_diff,
		bottom_desc_,
		bottom_data,
		cudnn::dataType<Dtype>::zero,
		bottom_desc_,
		bottom_diff));
}

template class CuDNNPoolingLayer<float>;
template class CuDNNPoolingLayer<double>;


}
