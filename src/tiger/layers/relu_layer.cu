#include <algorithm>
#include "tiger/layers/relu_layer.hpp"

namespace tiger{

template <typename Dtype>
__global__ void relu_forward(const int n, Dtype negative_slop, const Dtype* in, Dtype* out){
    int i = blockIdx.x * blockDim.x  + threadIdx.x;
    if(i < n){
	out[i] = in[i] > 0 ? in[i] : negative_slop * in[i];
    }
}
template <typename Dtype>
void ReLULayer<Dtype>::forward_gpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();
    const Dtype negative_slop = this->layer_param_.relu_param().negative_slop();
    relu_forward<Dtype><<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
	    count, negative_slop, bottom_data, top_data);
}

template <typename Dtype>
__global__ void relu_backward(const int n, Dtype negative_slop, const Dtype* in_diff,
    const Dtype* in_data, Dtype* out_diff){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
	out_diff[i] = in_diff[i] * ((in_data[i] > 0) + 
		(in_data[i] <= 0) * negative_slop);
    }
}


template <typename Dtype>
void ReLULayer<Dtype>::backward_gpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>* >& bottom){
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    const int count = bottom[0]->count();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    Dtype negative_slop = this->layer_param_.relu_param().negative_slop();
    relu_backward<Dtype><<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
	    count, negative_slop, top_diff, bottom_data, bottom_diff);
}

template class ReLULayer<float>;
template class ReLULayer<double>;

}
