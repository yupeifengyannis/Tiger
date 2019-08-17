#ifndef TIGER_LAYERS_CUDNN_CUDNN_SOFTMAX
#define TIGER_LAYERS_CUDNN_CUDNN_SOFTMAX

#include "tiger/utils/cudnn.hpp"
#include "tiger/layers/neuron/softmax_layer.hpp"
#include "tiger/tiger.pb.h"

namespace tiger{

template <typename Dtype>
class CuDNNSoftmaxLayer : public SoftmaxLayer<Dtype>{
public:
    explicit CuDNNSoftmaxLayer(const LayerParameter& param) : 
	SoftmaxLayer<Dtype>(param),
	handles_setup_(false){}
    virtual void layer_setup(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
    virtual void reshape(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
    virtual ~CuDNNSoftmaxLayer();
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
};

}

#endif
