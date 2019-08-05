#include "tiger/layers/pooling_layer.hpp"

namespace tiger{

template <typename Dtype>
void PoolingLayer<Dtype>::layer_setup(const vector<Blob<Dtype>* >& bottom, 
	const vector<Blob<Dtype>* >& top){
    PoolingParameter pooling_param = this->layer_param_.pooling_param();
    CHECK(pooling_param.has_kernel_h() && pooling_param.has_kernel_w()) << 
	"pooling parameter doesn't has kernel_h or kernel_w";
    kernel_h_ = pooling_param.kernel_h();
    kernel_w_ = pooling_param.kernel_w();
    
    CHECK(pooling_param.has_stride_h() && pooling_param.has_stride_w()) << 
	"pooling paramter doesn't has stride_h or stride_w";
    stride_h_ = pooling_param.stride_h();
    stride_w_ = pooling_param.stride_w();
    
    CHECK(pooling_param.has_pad_h() && pooling_param.has_pad_w()) << 
	"pooling paramter doesn't has pad_w or pad_w";
    pad_h_ = pooling_param.pad_h();
    pad_w_ = pooling_param.pad_w();
    
    CHECK(pooling_param.has_round_mode()) << 
	"pooling paramter doesn't has round mode";
    round_mode_ = pooling_param.round_mode();
}

template <typename Dtype>
void PoolingLayer<Dtype>::reshape(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){

}


}
