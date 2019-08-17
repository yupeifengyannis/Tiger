#include <algorithm>
#include "tiger/layers/neuron/softmax_layer.hpp"
#include "tiger/utils/math_function.hpp"

namespace tiger{

template <typename Dtype>
void SoftmaxLayer<Dtype>::reshape(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    // 一般这个轴都是设置为1
    softmax_axis_ = this->layer_param_.softmax_param().axis();
    top[0]->reshape_like(*bottom[0]);
    vector<int> mult_dims(1, bottom[0]->count(softmax_axis_));
    sum_multiplier_.reshape(mult_dims);
    Dtype* multiplier_data = sum_multiplier_.mutable_cpu_data();
    tiger_set(sum_multiplier_.count(), Dtype(1), multiplier_data);
    outer_num_ = bottom[0]->count(0, softmax_axis_);
    inner_num_ = bottom[0]->count(softmax_axis_ + 1);
    vector<int> scale_dims = bottom[0]->shape();
    scale_dims[softmax_axis_] = 1;
    scale_.reshape(scale_dims);
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* scale_data = scale_.mutable_cpu_data();
    int channels = bottom[0]->shape(softmax_axis_);
    int dim = bottom[0]->count() / outer_num_;
    tiger_copy(bottom[0]->count(), bottom_data, top_data);
    for(int i = 0; i < outer_num_; ++i){
	tiger_copy(inner_num_, bottom_data + i * dim, scale_data);
	for(int j = 0; j < channels; ++j){
	    for(int k = 0; k < inner_num_; ++k){
		scale_data[k] = std::max(scale_data[k], 
			bottom_data[i * dim + j * inner_num_ + k]);
	    }
	}
    }
    // TODO
}

template <typename Dtype>
void SoftmaxLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>* >& bottom){
    //TODO
}

template class SoftmaxLayer<float>;
template class SoftmaxLayer<double>;



}
