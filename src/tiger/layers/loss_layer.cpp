#include "tiger/layers/loss_layer.hpp"

namespace tiger{

template <typename Dtype>
void LossLayer<Dtype>::layer_setup(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    // 对于损失权重，其权重参数都是为１
    if(this->layer_param_.loss_weight_size() == 0){
	this->layer_param_.add_loss_weight(Dtype(1));
    }
}

template <typename Dtype>
void LossLayer<Dtype>::reshape(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    CHECK_EQ(bottom[0]->shape(0), bottom[1]->shape(0)) << 
	"the data and label should have the same first dimension";
    vector<int> loss_shape(0);
    top[0]->reshape(loss_shape);
}

template class LossLayer<float>;
template class LossLayer<double>;
}
