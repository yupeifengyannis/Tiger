#ifndef TIGER_LAYERS_TANH_LAYER_HPP
#define TIGER_LAYERS_TANH_LAYER_HPP

#include "tiger/layers/neuron_layer.hpp"

namespace tiger{

template <typename Dtype>
class TanhLayer : public NeuronLayer<Dtype>{
public:
    explicit TanhLayer(const LayerParameter& param) : 
	NeuronLayer<Dtype>(param){}
    virtual inline const char* type() const {
	return "Tanh";
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
