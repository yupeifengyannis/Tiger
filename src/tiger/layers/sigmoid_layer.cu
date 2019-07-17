#include <glog/logging.h>
#include "tiger/layers/sigmoid_layer.hpp"
#include "tiger/utils/device_alternate.hpp"

namespace tiger{

template <typename Dtype>
__global__ void sigmoid_forward(const int n, const Dtype* in, Dtype* out){
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; 
	    i += blockDim.x * gridDim.x){
	out[i] = 0.5 * tanh(0.5 * in[i]) + 0.5;
    }
}


template <typename Dtype>
void SigmoidLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    LOG(INFO) << "invoking forward gpu";
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();
    sigmoid_forward<Dtype><<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
	    count, bottom_data, top_data);


}

template <typename Dtype>
__global__ void sigmoid_backward(const int n, const Dtype* top_diff, 
	const Dtype* top_data, Dtype* bottom_diff){
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
	    i += blockDim.x * gridDim.x){
	const Dtype sigmoid_x = top_data[i];
	bottom_diff[i] = top_diff[i] * sigmoid_x * (1 - sigmoid_x);
    }
    
}


template <typename Dtype>
void SigmoidLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom){
    LOG(INFO) << "invoking backward gpu";
    if(propagate_down[0]){
	const Dtype* top_data = top[0]->gpu_data();
	const Dtype* top_diff = top[0]->gpu_diff();
	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
	const int count = bottom[0]->count();
	sigmoid_backward<Dtype><<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
	    count, top_diff, top_data, bottom_diff);
    }
}

template class SigmoidLayer<float>;
template class SigmoidLayer<double>;




}

