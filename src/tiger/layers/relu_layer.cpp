#include <algorithm>
#include "tiger/layers/relu_layer.hpp"

namespace tiger{

template <typename Dtype>
void ReLULayer<Dtype>::forward_cpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    const int count = bottom[0]->count();
    Dtype negative_slop = this->layer_param_.relu_param().negative_slop();
    for(int i = 0; i < count; i++){
	top_data[i] = std::max(bottom_data[i], Dtype(0)) + 
	    negative_slop * std::min(bottom_data[i], Dtype(0));
    }
}

template <typename Dtype>
void ReLULayer<Dtype>::backward_cpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>* >& bottom){
    if(!propagate_down[0]){
	return;
    }
    const Dtype* bottom_data = bottom[0]->cpu_data();
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    const int count = bottom[0]->count();
    Dtype negative_slop = this->layer_param_.relu_param().negative_slop();
    for(int i = 0; i < count; i++){
	bottom_diff[i] = top_diff[i] * ((bottom_data[i] > 0) + 
		negative_slop * (bottom_data[i] <= 0));
    }
}

template class ReLULayer<float>;
template class ReLULayer<double>;


}
