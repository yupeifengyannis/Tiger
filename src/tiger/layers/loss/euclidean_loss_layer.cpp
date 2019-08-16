#include <cmath>
#include "tiger/layers/loss/euclidean_loss_layer.hpp"
#include "tiger/utils/math_function.hpp"

namespace tiger{

template <typename Dtype>
void EuclideanLossLayer<Dtype>::reshape(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    LossLayer<Dtype>::reshape(bottom, top);
    CHECK_EQ(bottom[0]->count(), bottom[1]->count()) << 
	"prediction and label must have the same dimension";
    diff_.reshape_like(*bottom[0]);
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::forward_cpu(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    int count = bottom[0]->count();
    const Dtype* prediction = bottom[0]->cpu_data();
    const Dtype* label = bottom[1]->cpu_data();
    Dtype* diff_data = diff_.mutable_cpu_data();
    for(int i = 0; i < count; i++){
	diff_data[i] = prediction[i] - label[i];
    }    
    Dtype dot = Dtype(0);
    for(int i = 0; i < count; i++){
	dot += pow(diff_data[i], 2);
    }
    Dtype loss = dot / bottom[0]->num() / Dtype(2);
    top[0]->mutable_cpu_data()[0] = loss;
}

template <typename Dtype>
void EuclideanLossLayer<Dtype>::backward_cpu(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down,
	const vector<Blob<Dtype>* >& bottom){
    for(int i = 0; i < 2; i++){
	if(propagate_down[i]){
	    const Dtype sign = (i == 0) ? 1 : -1;
	    const Dtype alpha = sign * top[0]->cpu_diff()[0] / bottom[i]->num();
	    tiger_cpu_axpby<Dtype>(bottom[i]->count(), alpha, diff_.cpu_data(), Dtype(0),
		    bottom[i]->mutable_cpu_diff());	    
	}
    }
}

template class EuclideanLossLayer<float>;
template class EuclideanLossLayer<double>;


}
