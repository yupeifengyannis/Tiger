#include "layers/base_data_layer.hpp"
#include "common.hpp"

namespace tiger{

template <typename Dtype>
BaseDataLayer<Dtype>::BaseDataLayer(const LayerParameter& param) :
    Layer<Dtype>(param),
    transform_param_(param.transform_param()){
    }

template <typename Dtype>
void BaseDataLayer<Dtype>::layer_setup(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    if(top.size() == 1){
	output_label_ = false;
    }
    else{
	output_label_ = true;
    }
    data_transformer_.reset(new DataTransformer<Dtype>(transform_param_, this->phase_));
    data_layer_setup(bottom, top);
}

template <typename Dtype>
BasePrefetchingDataLayer<Dtype>::BasePrefetchingDataLayer(const LayerParameter& param) : 
    BaseDataLayer<Dtype>(param),
    prefetch_(param.data_param().prefetch()),
    prefetch_free_(),
    prefetch_full_(),
    prefetch_current_(){
	for(int i = 0; i < prefetch_.size(); i++){
	    prefetch_[i].reset(new Batch<Dtype>());
	    prefetch_free_.push(prefetch_[i].get());
	}
    }

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::layer_setup(const vector<Blob<Dtype>* >&bottom,
	const vector<Blob<Dtype>* >& top){
    BaseDataLayer<Dtype>::layer_setup(bottom, top);
    for(int i = 0; i < prefetch_.size(); i++){
	prefetch_[i]->data_.mutable_cpu_data();
	if(this->output_label_){
	    prefetch_[i]->label_.mutable_cpu_data();
	}
    }
#ifndef CPU_ONLY
    if(Tiger::mode() == Tiger::GPU){
	for(int i = 0; i < prefetch_.size(); i++){
	    prefetch_[i]->data_.mutable_gpu_data();
	    if(this->output_label_){
		prefetch_[i]->label_.mutable_gpu_data();
	    }
	}
    }
#endif

}


template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    if(prefetch_current_){
	prefetch_free_.push(prefetch_current_);
    }
    prefetch_current_ = prefetch_full_.pop("waiting for data");
    top[0]->reshape_like(prefetch_current_->data_);
    top[0]->set_cpu_data(prefetch_current_->data_.mutable_cpu_data());
    if(this->output_label_){
	top[1]->reshape_like(prefetch_current_->label_);
	top[1]->set_cpu_data(prefetch_current_->label_.mutable_cpu_data());
    }
}

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    if(prefetch_current_){
	prefetch_free_.push(prefetch_current_);
    }
    prefetch_current_ = prefetch_full_.pop("waiting for data");
    top[0]->reshape_like(prefetch_current_->data_);
    top[0]->set_gpu_data(prefetch_current_->data_.mutable_gpu_data());
    if(this->output_label_){
	top[1]->reshape_like(prefetch_current_->label_);
	top[1]->set_gpu_data(prefetch_current_->label_.mutable_gpu_data());
    }
}


template class BasePrefetchingDataLayer<float>;
template class BasePrefetchingDataLayer<double>;


}

