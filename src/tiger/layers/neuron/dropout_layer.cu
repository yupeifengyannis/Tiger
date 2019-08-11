#include "tiger/layers/neuron/dropout_layer.hpp"
#include "tiger/utils/math_function.hpp"

namespace tiger{




template <typename Dtype>
__global__ void dropout_forward_kernel(const int n, const Dtype* in, const unsigned int* mask,
    const unsigned int threshold, const float scale, Dtype* out){
    int i  = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
	out[i] = in[i] * (mask[i] > threshold)  * scale;
    }
}


template <typename Dtype>
void DropoutLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const int count = bottom[0]->count();
    if(this->phase_ == TRAIN){
	unsigned int* mask = static_cast<unsigned int*>(this->rand_vec_.mutable_gpu_data());
	tiger_gpu_rng_uniform(count, mask);
	dropout_forward_kernel<Dtype><<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count, bottom,
		mask, uint_thres_, scale_, top_data);
    }
    else{
	for(int i = 0; i < count; i++){
	    top_data[i] = bottom_data[i];
	}
    }
}

template <typename Dtype>
__global__ void dropout_backward_kernel(const int n, const Dtype* in, const unsigned int* mask,
    const unsigned int threshold, const float sacle, Dtype* out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n){
	out[i] = in[i] * (mask[i] > threshold) * sacle;
    }
}


template <typename Dtype>
void DropoutLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom){
    if(!propagate_down[0]){
	return;
    }
    const Dtype top_diff = top[0]->gpu_diff();
    Dtype* bottom_diff = bottom[0]->gpu_diff();
    const unsigned int* mask = rand_vec_.gpu_data();
    const int count = bottom[0]->count();
    if(this->phase_ == TRAIN){
	dropout_backward_kernel<Dtype><<<GET_BLOCKS(count), CUDA_NUM_THREADS>>>(count,
		top_diff, mask, uint_thres_, scale_, bottom_diff);	
    }
    else{
	for(int i = 0; i < count; i++){
	    bottom_diff[i] = top_diff[i];
	}
    }
}



}
