#ifndef TIGER_LAYERS_NEURON_SIGMOID_LAYER_HPP
#define TIGER_LAYERS_NEURON_SIGMOID_LAYER_HPP

#include "tiger/layers/neuron/neuron_layer.hpp"


namespace tiger{

template <typename Dtype>
class SigmoidLayer : public NeuronLayer<Dtype>{
public:
    explicit SigmoidLayer(const LayerParameter& param) : 
	NeuronLayer<Dtype>(param){}
    
    virtual inline const char* type() const override{
	return "Sigmoid";
    }
protected:
    virtual void forward_cpu(const vector<Blob<Dtype>* >& bottom, 
	    const vector<Blob<Dtype>* >& top) override;

    virtual void backward_cpu(const vector<Blob<Dtype>* >& top,
	    const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom) override;
    
    virtual void forward_gpu(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top) override;

    virtual void backward_gpu(const vector<Blob<Dtype>* >& top,
	    const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom) override;

};

}



#endif
