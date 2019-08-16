#include "tiger/utils/math_function.hpp"
#include "tiger/layers/neuron/dropout_layer.hpp"
#include "tiger/layer_factory.hpp"

namespace tiger{

template <typename Dtype>
void DropoutLayer<Dtype>::layer_setup(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    // 调用父类的layer_setup函数
    NeuronLayer<Dtype>::layer_setup(bottom, top);
    threshold_ = this->layer_param_.drop_param().dropout_ratio();
    DCHECK(threshold_ > 0.);
    DCHECK(threshold_ < 1.);
    scale_ = 1. / (1. - threshold_);
    uint_thres_ = static_cast<unsigned int>(UINT_MAX * threshold_);
}

template <typename Dtype>
void DropoutLayer<Dtype>::reshape(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    // 调用父类的reshape函数
    NeuronLayer<Dtype>::reshape(bottom, top);
    std::vector<int> shape_vec = bottom[0]->shape();
    rand_vec_.reshape(shape_vec);
}

template <typename Dtype>
void DropoutLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    unsigned int* mask = rand_vec_.mutable_cpu_data();
    const int count = bottom[0]->count();
    if(this->phase_ == TRAIN){
	tiger_rng_bernoulli<Dtype>(count, 1. - threshold_, mask);
	for(int i = 0; i < count; i++){
	    top_data[i] = bottom_data[i] * mask[i] * scale_;
	}
    }
    else{
	for(int i = 0; i < count; i++){
	    top_data[i] = bottom_data[i];
	}
    }

}

template <typename Dtype>
void DropoutLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>* >& top, 
	const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom){
    if(propagate_down[0]){
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const unsigned int* mask = rand_vec_.cpu_data();
	const int count = bottom[0]->count();
	if(this->phase_ == TRAIN){
	    for(int i = 0; i < count; i++){
		bottom_diff[i] = top_diff[i] * mask[i] * scale_;
	    }
	}
	else{
	    for(int i = 0; i < count; i++){
		bottom_diff[i] = top_diff[i];
	    } 
	}
    }
}

template class DropoutLayer<float>;
template class DropoutLayer<double>;

REGISTER_LAYER_CLASS(Dropout);

}
