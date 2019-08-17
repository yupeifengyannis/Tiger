#ifndef TIGER_LAYERS_CONV_DECONV_LAYER_HPP
#define TIGER_LAYERS_CONV_DECONV_LAYER_HPP

#include "tiger/layers/conv/base_conv_layer.hpp"
#include "tiger/tiger.pb.h"

namespace tiger{

template <typename Dtype>
class DeconvolutionLayer : public BaseConvolutionLayer<Dtype>{
public:
    explicit DeconvolutionLayer(const LayerParameter& param) : 
	BaseConvolutionLayer<Dtype>(param){}
    virtual inline const char* type() const{
	return "Deconvolution";
    }
};


}




#endif
