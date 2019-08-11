#include "tiger/layers/neuron/tanh_layer.hpp"
#include "tiger/utils/device_alternate.hpp"

namespace tiger{

template <typename Dtype>
__global__ void tanh_forward(const int n, const Dtype* in, Dtype* out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
	out[i] = tanh(in[i]);
    }
}

template <typename Dtype>
void TanhLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();
    tanh_forward<Dtype><<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
	    count, bottom_data, top_data);
}

template <typename Dtype>
__global__ void tanh_backward(const int n, const Dtype* top_diff, const Dtype* top_data,
	Dtype* bottom_diff){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
	bottom_diff[i] = top_diff[i] * (1 - top_data[i] * top_data[i]);
    }
}

template <typename Dtype>
void TanhLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>* >& bottom){
    if(!propagate_down[0]){
	return;
    }
    const Dtype* top_data = top[0]->gpu_data();
    const Dtype* top_diff = top[0]->gpu_diff();
    const int count = bottom[0]->count();
    Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
    tanh_backward<Dtype><<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(
	    count, top_diff, top_data, bottom_diff);
}

template class TanhLayer<float>;
template class TanhLayer<double>;


}
