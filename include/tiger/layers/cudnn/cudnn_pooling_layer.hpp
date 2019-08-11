#ifndef TIGER_LAYERS_CUDNN_POOLING_LAYER_HPP
#define TIGER_LAYERS_CUDNN_POOLING_LAYER_HPP

#include "tiger/layers/conv/pooling_layer.hpp"
#include "tiger/utils/cudnn.hpp"
namespace tiger{
template <typename Dtype>
class CuDNNPoolingLayer : public PoolingLayer<Dtype>{
public:
    explicit CuDNNPoolingLayer(const LayerParameter& param) : 
	PoolingLayer<Dtype>(param),
	handles_setup_(false){}
    
    virtual void layer_setup(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
    virtual void reshape(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
    virtual ~CuDNNPoolingLayer();
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
    cudnnPoolingDescriptor_t pooling_desc_;
    cudnnPoolingMode_t mode_;
};
}

#endif
