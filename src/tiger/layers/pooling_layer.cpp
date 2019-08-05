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
    CHECK_EQ(bottom[0]->num_axes(), 4) << "input must have 4 aexs";
    channels_ = bottom[0]->channels();
    height_ = bottom[0]->height();
    width_ = bottom[0]->width();
    switch(round_mode_){
	case PoolingParameter_RoundMode_CEIL:
	    pooled_height_ = static_cast<int>(ceil(static_cast<float>(
			    height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
	    pooled_width_ = static_cast<int>(ceil(static_cast<float>(
			    width_ + 2 * pad_w_ - kernel_w_)  / stride_w_)) + 1;
	    break;
	case PoolingParameter_RoundMode_FLOOR:
	    pooled_height_ = static_cast<int>(floor(static_cast<float>(
			    height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
	    pooled_width_ = static_cast<int>(floor(static_cast<float>(
			    width_ + 2 * pad_w_ - kernel_w_)  / stride_w_)) + 1;
	    break;
	default:
	    LOG(FATAL) << "unkown rounding mode";
	    break;
    }	
    
    if(pad_h_ || pad_w_){
	if((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_){
	    pooled_height_--;
	}
	if((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_){
	    pooled_width_--;
	}
    }
    
    top[0]->reshape(std::vector<int>{bottom[0]->num(), channels_, pooled_height_, pooled_width_});
    if(top.size() > 1){
	top[1]->reshape_like(*top[0]);
    }
    if(this->layer_param_.pooling_param().method() == 
	    PoolingParameter_PoolMethod_MAX){
	this->max_idx_.reshape(std::vector<int>{bottom[0]->num(), 
	channels_, pooled_height_, pooled_width_});
    }
    if(this->layer_param_.pooling_param().method() == 
	    PoolingParameter_PoolMethod_STOCHASTIC){
	this->rand_idx_.reshape_like(*top[0]);
    }
}

template <typename Dtype>
void PoolingLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    // TODO()
}

template <typename Dtype>
void PoolingLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>* >& bottom){
    // TODO()
}

template class PoolingLayer<float>;
template class PoolingLayer<double>;






}
