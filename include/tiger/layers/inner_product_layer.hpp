#ifndef TIGER_LAYERS_INNER_PRODUCT_LAYER_HPP
#define TIGER_LAYERS_INNER_PRODUCT_LAYER_HPP

#include "tiger/layer.hpp"

namespace tiger{

// 如果是函数重载的话建议使用关键字

template <typename Dtype>
class InnerProductLayer : public Layer<Dtype>{
public:
    explicit InnerProductLayer(const LayerParameter& param) : 
	Layer<Dtype>(param){}
    virtual void layer_setup(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top) ;
    virtual void reshape(const vector<Blob<Dtype>* >& bottom, 
	    const vector<Blob<Dtype>* >& top) ;

    virtual inline const char* type() const {
	return "InnerProduct";
    }

    virtual inline int exact_num_bottom_blobs() const {
	return 1;
    }

    virtual inline int exact_num_top_blobs() const {
	return 1;
    }
protected:
    virtual void forward_cpu(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top) ;
    virtual void forward_gpu(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top) ;
   
    virtual void backward_cpu(const vector<Blob<Dtype>* >& top,
	    const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom) ;
    
    virtual void backward_gpu(const vector<Blob<Dtype>* >& top,
	    const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom) ;

    int M_;
    int N_;
    int K_;
    bool bias_term_;
    Blob<Dtype> bias_multiplier_;
    bool transpose_;
};

}



#endif
