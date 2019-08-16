#ifndef TIGER_LAYERS_CUDNN_RELU_LAYER_HPP
#define TIGER_LAYERS_CUDNN_RELU_LAYER_HPP
#include "tiger/layers/neuron/relu_layer.hpp"
#include "tiger/utils/cudnn.hpp"

namespace tiger{

template <typename Dtype>
class CuDNNReLULayer : public ReLULayer<Dtype>{
public:
    explicit CuDNNReLULayer(const LayerParameter& param) : 
	ReLULayer<Dtype>(param),
	handles_setup_(false){}
    virtual ~CuDNNReLULayer();
    virtual void layer_setup(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
    virtual void reshape(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
protected:
    virtual void forward_gpu(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
    virtual void backward_gpu(const vector<Blob<Dtype>* >& top,
	    const vector<bool>& propagate_down,
	    const vector<Blob<Dtype>* >& bottom);

    bool handles_setup_;
    cudnnHandle_t handle_;
    cudnnTensorDescriptor_t bottom_desc_;
    cudnnTensorDescriptor_t top_desc_;
    cudnnActivationDescriptor_t activ_desc_;
};

}

#endif
