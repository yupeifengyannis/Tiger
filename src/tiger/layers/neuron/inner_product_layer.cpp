#include <vector>
#include <memory>
#include "tiger/layers/neuron/inner_product_layer.hpp"
#include "tiger/tiger.pb.h"
#include "tiger/filler.hpp"

namespace tiger{

template <typename Dtype>
void InnerProductLayer<Dtype>::layer_setup(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    // 确定这一层权重的相关参数
    const int num_output = this->layer_param_.inner_param().num_output();
    bias_term_ = this->layer_param_.inner_param().bias_term();
    transpose_ = this->layer_param_.inner_param().transpose();
    N_ = num_output;
    int axis = this->layer_param_.inner_param().axis();
    K_ = bottom[0]->count(axis);
    // 对权重参数进行初始化
    if(this->blobs_.size() > 0){
	LOG(INFO) << "weights and bias is already initlized, skipping!";
    }
    else{
	if(bias_term_){
	    this->blobs_.resize(2);
	}
	else{
	    this->blobs_.resize(1);
	}
	// create weight blob
	std::vector<int> weight_shape(2);
	weight_shape[0] = K_;
	weight_shape[1] = N_;
	this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
	FillerParameter weight_param = this->layer_param_.inner_param().weight_filler();
	std::shared_ptr<Filler<Dtype> > weight_filler(get_filler<Dtype>(weight_param));
	weight_filler->fill_data(this->blobs_[0].get());
	
	if(bias_term_){
	    std::vector<int> bias_shape{N_};
	    this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
	    FillerParameter bias_param = this->layer_param_.inner_param().bias_filler();
	    std::shared_ptr<Filler<Dtype> > bias_filler(get_filler<Dtype>(bias_param));
	    bias_filler->fill_data(this->blobs_[1].get());
	}

    }
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void InnerProductLayer<Dtype>::reshape(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    // 根据输入的bottom参数和输出的权重参数来确定top的维度
    int axis = this->layer_param_.inner_param().axis();
    M_ = bottom[0]->count(0, axis);
    std::vector<int> top_shape{M_, N_};
    top[0]->reshape(top_shape);
    // 确定自己这一层的成员变量的数据
    if(bias_term_){
	vector<int> bias_shape{M_};
	bias_multiplier_.reshape(bias_shape);
	tiger_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
    }
}

template <typename Dtype>
void InnerProductLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){

}

template <typename Dtype>
void InnerProductLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom){
}

template class InnerProductLayer<float>;
template class InnerProductLayer<double>;

REGISTER_LAYER_CLASS(InnerProduct);

}
