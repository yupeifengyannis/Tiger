#ifndef TIGER_LAYERS_CONV_CONV_LAYER_HPP
#define TIGER_LAYERS_CONV_CONV_LAYER_HPP

#include "tiger/layers/conv/base_conv_layer.hpp"

namespace tiger{
template <typename Dtype>
class ConvolutionLayer : public BaseConvolutionLayer<Dtype>{
public:
    explicit ConvolutionLayer(const LayerParameter& param) : 
	BaseConvolutionLayer<Dtype>(param){}
    virtual inline const char* type() const{
	return "Convolution";
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
    virtual inline bool reverse_dimensions(){
	return false;
    }
    virtual void compute_output_shape();
};
}

#endif
