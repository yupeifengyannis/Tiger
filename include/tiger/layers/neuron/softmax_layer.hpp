#ifndef TIGER_LAYERS_NEURON_SOFTMAX_LAYER_HPP
#define TIGER_LAYERS_NEURON_SOFTMAX_LAYER_HPP

#include "tiger/blob.hpp"
#include "tiger/layer.hpp"
#include "tiger/tiger.pb.h"


namespace tiger{

template <typename Dtype>
class SoftmaxLayer : public Layer<Dtype>{
public:
    explicit SoftmaxLayer(const LayerParameter& layer_param) : 
	Layer<Dtype>(layer_param){}
    virtual void layer_setup(const vector<Blob<Dtype>* >& bottom, 
	    const vector<Blob<Dtype>* >& top){}
    virtual void reshape(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);

    virtual inline const char* type() const{
	return "Softmax";
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

    int outer_num_;
    int inner_num_;
    int softmax_axis_;
    Blob<Dtype> sum_multiplier_;
    Blob<Dtype> scale_;
};


}


#endif
