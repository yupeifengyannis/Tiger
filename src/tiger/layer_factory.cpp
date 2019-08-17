#include <memory>
#include <glog/logging.h>
#include "tiger/layer.hpp"
#include "tiger/layer_factory.hpp"
#include "tiger/tiger.pb.h"
#include "tiger/layers/neuron/sigmoid_layer.hpp"
#include "tiger/layers/cudnn/cudnn_sigmoid_layer.hpp"
#include "tiger/layers/conv/pooling_layer.hpp"
#include "tiger/layers/cudnn/cudnn_pooling_layer.hpp"
#include "tiger/layers/neuron/relu_layer.hpp"
#include "tiger/layers/cudnn/cudnn_relu_layer.hpp"
#include "tiger/layers/neuron/tanh_layer.hpp"
#include "tiger/layers/cudnn/cudnn_tanh_layer.hpp"
#include "tiger/layers/neuron/softmax_layer.hpp"
#include "tiger/layers/cudnn/cudnn_softmax.hpp"
#include "tiger/layers/conv/pooling_layer.hpp"
#include "tiger/layers/cudnn/cudnn_pooling_layer.hpp"

namespace tiger{
template <typename Dtype>
std::shared_ptr<Layer<Dtype> > get_sigmoid_layer(const LayerParameter& param){
    if(TIGER == param.backend()){
	return std::shared_ptr<Layer<Dtype> > (new SigmoidLayer<Dtype>(param)); 
    }
    else if(CUDNN == param.backend()){
	return std::shared_ptr<Layer<Dtype> > (new CuDNNSigmoidLayer<Dtype>(param));
    }
    else{
	LOG(INFO) << "Layer " << param.name() << "has unknow backend";
    }
}
REGISTER_LAYER_CREATOR(Sigmoid, get_sigmoid_layer);


template <typename Dtype>
std::shared_ptr<Layer<Dtype> > get_pooling_layer(const LayerParameter& param){
    if(TIGER == param.backend()){
	return std::shared_ptr<Layer<Dtype> > (new PoolingLayer<Dtype>(param));
    }
    else if(CUDNN == param.backend()){
	return std::shared_ptr<Layer<Dtype> > (new CuDNNPoolingLayer<Dtype>(param));
    }
    else{
	LOG(INFO) << "Layer " << param.name() << "has unknow backend";
    }
}
REGISTER_LAYER_CREATOR(Pooling, get_pooling_layer);

template <typename Dtype>
std::shared_ptr<Layer<Dtype> > get_relu_layer(const LayerParameter& param){
    if(TIGER == param.backend()){
	return std::shared_ptr<Layer<Dtype> > (new ReLULayer<Dtype>(param));
    } 
    else if(CUDNN == param.backend()){
	return std::shared_ptr<Layer<Dtype> > (new CuDNNReLULayer<Dtype>(param));
    }
    else{
	LOG(INFO) << "Layer " << param.name() << "has unknow backend";
    }
}
REGISTER_LAYER_CREATOR(ReLU, get_relu_layer);

template <typename Dtype>
std::shared_ptr<Layer<Dtype> > get_tanh_layer(const LayerParameter& param){
    if(TIGER == param.backend()){
	return std::shared_ptr<Layer<Dtype> > (new TanhLayer<Dtype>(param));
    }
    else if(CUDNN == param.backend()){
	return std::shared_ptr<Layer<Dtype> > (new CuDNNTanhLayer<Dtype>(param));
    } 
    else{
	LOG(INFO) << "Layer " << param.name() << "has unknow backend";
    }

}

REGISTER_LAYER_CREATOR(Tanh, get_tanh_layer);

template <typename Dtype>
std::shared_ptr<Layer<Dtype> > get_softmax_layer(const LayerParameter& param){
    if(TIGER == param.backend()){
	return std::shared_ptr<Layer<Dtype> > (new SoftmaxLayer<Dtype>(param));
    }
    else if(CUDNN == param.backend()){
	return std::shared_ptr<Layer<Dtype> > (new CuDNNSoftmaxLayer<Dtype>(param));
    }
    else{
	LOG(INFO) << "Layer " << param.name() << "has unkonwn backend";
    }
}

REGISTER_LAYER_CREATOR(Softmax, get_softmax_layer);













}
