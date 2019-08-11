#include "tiger/layers/neuron/inner_product_layer.hpp"
#include "tiger/utils/math_function.hpp"

namespace tiger{
template <typename Dtype>
void InnerProductLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    // 底层数据的维度为 (M_, K_)
    // 权重的维度为 (K_, N_)
    // 输出结果的维度为 (M_, N_)
    const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    const Dtype* weight = this->blobs_[0]->gpu_data();
    tiger_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, K_,
	    Dtype(1), bottom_data, weight, Dtype(0), top_data);
    if(bias_term_){
	tiger_gpu_gemm(CblasNoTrans, CblasNoTrans, M_, N_, 1, Dtype(1),
		bias_multiplier_.gpu_data(), this->blobs_[1]->gpu_data(),
		Dtype(1), top_data);
    }

}

template <typename Dtype>
void InnerProductLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>* >& top,
	const std::vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom){
    // 记住一点：propagate_down是传导到上一层的
    // 而param_propagate_down是传导到给当前层的权重参数的
    // 计算dW
    // 为什么计算dW的时候这个beta是1 TODO
    if(this->param_propagate_down_[0]){
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* bottom_data = bottom[0]->gpu_data();
	tiger_gpu_gemm(CblasTrans, CblasNoTrans, K_, N_, M_,
		Dtype(1), bottom_data, top_diff, Dtype(1),
		this->blobs_[0]->mutable_gpu_diff());
    }
    // 计算db
    // 计算db的时候这个beta是1
    if(bias_term_ && this->param_propagate_down_[1]){
	const Dtype* top_diff = top[0]->gpu_diff();
	tiger_gpu_gemv(CblasTrans, M_, N_, Dtype(1), top_diff,
		this->bias_multiplier_.gpu_data(), Dtype(1),
		this->blobs_[1]->mutable_gpu_diff());
    }
    
    // 计算dA
    // 计算dA的时候这个beta为0
    if(propagate_down[0]){
	const Dtype* top_diff = top[0]->gpu_diff();
	const Dtype* weight = this->blobs_[0]->gpu_data();
	tiger_gpu_gemm(CblasNoTrans, CblasTrans, M_, K_, N_,
		Dtype(1), top_diff, weight, Dtype(0),
		bottom[0]->mutable_gpu_diff());
    } 

}

template class InnerProductLayer<float>;
template class InnerProductLayer<double>;
}

