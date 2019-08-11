#include <cmath>
#include <glog/logging.h>
#include "tiger/layers/neuron/sigmoid_layer.hpp"

namespace tiger{

template <typename Dtype>
static Dtype sigmoid(Dtype x){
    return 0.5 * tanh(0.5 * x) + 0.5; 
}

template <typename Dtype>
void SigmoidLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    LOG(INFO) << "invoking forward_cpu";
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = bottom[0]->count();
    for(int i = 0; i < count; i++){
	top_data[i] = sigmoid(bottom_data[i]);
    }
}

template <typename Dtype>
void SigmoidLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom){
    LOG(INFO) << "invoking backward_cpu";
    if(propagate_down[0]){
	const Dtype* top_data = top[0]->cpu_data();
	const Dtype* top_diff = top[0]->cpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
	const int count = bottom[0]->count();
	for(int i = 0; i < count; i++){
	    const Dtype sigmoid_x = top_data[i];
	    bottom_diff[i] = top_diff[i] * sigmoid_x * (1.0 - sigmoid_x);
	}
    } 
}


template class SigmoidLayer<float>;
template class SigmoidLayer<double>;



}
