#ifndef TIGER_LAYERS_LOSS_LAYER_HPP
#define TIGER_LAYERS_LOSS_LAYER_HPP

#include "tiger/layer.hpp"
#include "tiger/tiger.pb.h"

namespace tiger{

template <typename Dtype>
class LossLayer : public Layer<Dtype>{
public:
    explicit LossLayer(const LayerParamter& param) : 
	Layer<Dtype>(param){}
    
    virtual void layer_setup(const vector<Blob<Dtype>* >& bottom, 
	    const vector<Blob<Dtype>* >& top) override;
    virtual void reshape(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top) override;

    virtual void int exact_num_bottom_blobs() const{
	return 2;
    }
    virtual void int exact_num_top_blobs() const{
	return 1;
    }

}

}


#endif
