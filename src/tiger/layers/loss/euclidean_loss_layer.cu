#include "tiger/layers/loss/euclidean_loss_layer.hpp"
#include "tiger/utils/math_function.hpp"
namespace tiger{

template <typename Dtype>
void EuclideanLossLayer<Dtype>::forward_gpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    int count = bottom[0]->count();
    const Dtype* prediction = bottom[0]->gpu_data();
    const Dtype* label = bottom[1]->gpu_data();
    Dtype* diff_data = diff_.mutable_gpu_data();
    for(int i = 0; i < count; i++){
	diff_data[i] = prediction[i] - label[i];
    }
    Dtype dot;
    tiger_gpu_dot(count, diff_.gpu_data(), diff_.gpu_data(), &dot);
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_gpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::backward_gpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>* >& bottom){
    for(int i = 0; i < 2; i++){
	if(propagate_down[i]){
	    const Dtype sign = (i == 0) ? 1 : -1;
	    const Dtype alpha = sign * top[0]->gpu_diff()[0] / bottom[i]->num();
	    tiger_gpu_axpby(
		    bottom[i]->count(),
		    alpha,
		    diff_.gpu_data(),
		    Dtype(0),
		    bottom[i]->mutable_gpu_diff());
	}
    }
}

template class EuclideanLossLayer<float>;
template class EuclideanLossLayer<double>;


}
