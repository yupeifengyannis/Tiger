#include "tiger/layers/cudnn/cudnn_dropout_layer.hpp"

namespace tiger{


template <typename Dtype>
void CuDNNDropoutLayer<Dtype>::layer_setup(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    DropoutLayer<Dtype>::layer_setup(bottom, top);
    CUDNN_CHECK(cudnnCreate(&handle_));
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
    cudnn::createTensor4dDesc<Dtype>(&bottom_desc_);
    CUDNN_CHECK(cudnnCreateDropoutDescriptor(&drop_desc_));  
    handles_setup_ = true;
}

template <typename Dtype>
void CuDNNDropoutLayer<Dtype>::reshape(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    CHECK_EQ(bottom[0]->num_axes(), 4) << 
	"bottom data must has 4 axes";
    DropoutLayer<Dtype>::reshape(bottom, top);
    const int N = bottom[0]->num();
    const int C = bottom[0]->channels();
    const int H = bottom[0]->height();
    const int W = bottom[0]->width();
    cudnn::setTensor4dDesc<Dtype>(&bottom_desc_, N, C, H, W);
    cudnn::setTensor4dDesc<Dtype>(&top_desc_, N, C, H, W);
    // TODO
    // cudnnSetDropoutDescriptor()
}




}
