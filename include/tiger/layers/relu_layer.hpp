#ifndef TIGER_LAYERS_RELU_LAYER_HPP
#define TIGER_LAYERS_RELU_LAYER_HPP

#include "tiger/layer.hpp"
#include "tiger/layers/neuron_layer.hpp"
#include "tiger/tiger.pb.h"

namespace tiger{


template <typename Dtype>
class ReLULayer : public NeuronLayer<Dtype>{
public:
    explicit ReLULayer(const LayerParameter& param) : 
	NeuronLayer<Dtype>(param){}
    virtual inline const char* type() const{
	return "ReLU";
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
};



}


#endif
