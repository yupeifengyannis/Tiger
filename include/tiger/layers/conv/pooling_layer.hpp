#ifndef TIGER_LAYERS_CONV_POOLING_LAYER_HPP
#define TIGER_LAYERS_CONV_POOLING_LAYER_HPP

#include "tiger/layer.hpp"
#include "tiger/blob.hpp"
#include "tiger/tiger.pb.h"

namespace tiger{

template <typename Dtype>
class PoolingLayer : public Layer<Dtype>{
public:
    explicit PoolingLayer(const LayerParameter& param) : 
	Layer<Dtype>(param){}
    virtual void layer_setup(const vector<Blob<Dtype>* >& bottom, 
	    const vector<Blob<Dtype>* >& top);
    virtual void reshape(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
    virtual inline const char* type()const{
	return "Pooling";
    }

protected:
    virtual void forward_cpu(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
    virtual void forward_gpu(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
    virtual void backward_cpu(const vector<Blob<Dtype>* >& top,
	    const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom);
    virtual void backward_gpu(const vector<Blob<Dtype>* >& top,
	    const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom);
    
    int kernel_h_;
    int kernel_w_;
    int stride_h_;
    int stride_w_;
    int pad_h_;
    int pad_w_;
    int channels_;
    int height_;
    int width_;
    int pooled_height_;
    int pooled_width_;
    Blob<Dtype> rand_idx_;
    Blob<int> max_idx_;
    PoolingParameter_RoundMode round_mode_;
};

}


#endif
