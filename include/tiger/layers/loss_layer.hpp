#ifndef TIGER_LAYERS_LOSS_LAYER_HPP
#define TIGER_LAYERS_LOSS_LAYER_HPP

#include "tiger/layer.hpp"
#include "tiger/tiger.pb.h"

namespace tiger{

template <typename Dtype>
class LossLayer : public Layer<Dtype>{
public:
    explicit LossLayer(const LayerParameter& param) : 
	Layer<Dtype>(param){}
    
    virtual void layer_setup(const vector<Blob<Dtype>* >& bottom, 
	    const vector<Blob<Dtype>* >& top) override;
    virtual void reshape(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top) override;

    virtual int exact_num_bottom_blobs() const{
	// bottom[0]存放的是预测值，而bottom[1]存放的是label值
	return 2;
    }
    virtual int exact_num_top_blobs() const{
	return 1;
    } 
protected:
    inline virtual void forward_cpu(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top){}
    inline virtual void forward_gpu(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top){}
    inline virtual void backward_cpu(const vector<Blob<Dtype>* >& top,
	    const vector<bool>& propagate_down,
	    const vector<Blob<Dtype>* >& bottom){}
    inline virtual void backward_gpu(const vector<Blob<Dtype>* >& top,
	    const vector<bool>& propagate_down,
	    const vector<Blob<Dtype>* >& bottom){}
};

}


#endif
