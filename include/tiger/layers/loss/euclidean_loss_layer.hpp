#ifndef TIGER_LAYERS_LOSS_EUCLIDEAN_LOSS_LAYER_HPP
#define TIGER_LAYERS_LOSS_EUCLIDEAN_LOSS_LAYER_HPP

#include "tiger/layers/loss/loss_layer.hpp"

namespace tiger{

template <typename Dtype>
class EuclideanLossLayer : public LossLayer<Dtype>{
public:
    explicit EuclideanLossLayer(const LayerParameter& param) : 
	LossLayer<Dtype>(param),
	diff_(){}
    virtual void reshape(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);

    virtual inline const char* type() const{
	return "EuclideanLoss";
    }
protected:
    virtual void forward_cpu(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);

    virtual void forward_gpu(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);

    virtual void backward_cpu(const vector<Blob<Dtype>* >& top,
	    const vector<bool>& propagate_down,
	    const vector<Blob<Dtype>* >& bottom);

    virtual void backward_gpu(const vector<Blob<Dtype>* >& top,
	    const vector<bool>& propagate_down,
	    const vector<Blob<Dtype>* >& bottom);

    Blob<Dtype> diff_;
};

}

#endif
