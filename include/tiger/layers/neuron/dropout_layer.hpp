#ifndef TIGER_LAYERS_NEURON_DROPOUT_LAYER_HPP
#define TIGER_LAYERS_NEURON_DROPOUT_LAYER_HPP

#include "tiger/layers/neuron/neuron_layer.hpp"
#include "tiger/tiger.pb.h"

namespace tiger{

// 关于dropout有一点需要注意一下，就是如果我们以一定的比例来
// 将某一些神经元放弃了，那么我们需要在剩下的神经元的权重参数
// 乘以相应的比例，从而来保证期望值是相同的

template <typename Dtype>
class DropoutLayer : public NeuronLayer<Dtype>{
public:
    explicit DropoutLayer(const LayerParameter& param) : 
	NeuronLayer<Dtype>(param){}
    
    virtual void layer_setup(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
    
    virtual void reshape(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);

    virtual inline const char* type() const{
	return "Dropout";
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
    
    Blob<unsigned int> rand_vec_;
    Dtype threshold_;
    Dtype scale_;
    unsigned int uint_thres_;

};
}



#endif
