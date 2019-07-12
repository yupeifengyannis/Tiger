#ifndef LAYERS_BASE_DATA_LAYER_HPP
#define LAYERS_BASE_DATA_LAYER_HPP


#include "layer.hpp"

namespace tiger{

template <typename Dtype>
class BaseDataLayer : public Layer<Dtype>{
public:
    explicit BaseDataLayer(const LayerParameter& param);

    virtual void layer_setup(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
    
    virtual void data_layer_setup(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top){}
    
    virtual void reshape(const vector<Blob<Dtype>* >& bottom, 
	    const vector<Blob<Dtype>* >& top){}

    virtual void backward_cpu(const vector<Blob<Dtype>* >& top, 
	    const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom){}

    virtual void backward_gpu(const vector<Blob<Dtype>* >& top,
	    const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom){}
protected:


};


}



#endif

