#ifndef TIGER_LAYERS_CUDNN_SIGMOID_LAYER_HPP
#define TIGER_LAYERS_CUDNN_SIGMOID_LAYER_HPP
#include "tiger/utils/cudnn.hpp"
#include "tiger/layers/neuron/sigmoid_layer.hpp"
#include "tiger/tiger.pb.h"

namespace tiger{

template <typename Dtype>
class CuDNNSigmoidLayer : public SigmoidLayer<Dtype>{
public:
    explicit CuDNNSigmoidLayer(const LayerParameter& param) : 
	SigmoidLayer<Dtype>(param),
	handles_setup_(false){}
    virtual ~CuDNNSigmoidLayer();

    virtual void layer_setup(const vector<Blob<Dtype>* >& bottom, 
	    const vector<Blob<Dtype>* >& top);
    
    virtual void reshape(const vector<Blob<Dtype>* >& bottom, 
	    const vector<Blob<Dtype>* >& top);

protected:
    virtual void forward_gpu(const vector<Blob<Dtype>* >& bottom, 
	    const vector<Blob<Dtype>* >& top) override;
    virtual void backward_gpu(const vector<Blob<Dtype>* >& top,
	    const vector<bool>& propagate_down, 
	    const vector<Blob<Dtype>* >& bottom) override;
    
    bool handles_setup_;
    cudnnHandle_t handle_;
    cudnnTensorDescriptor_t bottom_desc_;
    cudnnTensorDescriptor_t top_desc_;
    cudnnActivationDescriptor_t activ_desc_;

};



}

#endif
