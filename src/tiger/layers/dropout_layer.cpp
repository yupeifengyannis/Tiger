#include "tiger/layers/dropout_layer.hpp"

namespace tiger{

template <typename Dtype>
void DropoutLayer<Dtype>::layer_setup(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    // 调用父类的layer_setup函数
    NeuronLayer<Dtype>::layer_setup(bottom, top);
    threshold_ = this->layer_param_.dropout_param().dropout_ratio();
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
    rand_vec_.reshape(*bottom[0]);
}


}
