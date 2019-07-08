#ifndef LAYER_HPP
#define LAYER_HPP

#include <memory>
#include <vector>
#include "common.hpp"
#include "blob.hpp"
#include "tiger.pb.h"

namespace tiger{

template <typename Dtype>
class Layer{
public:
    explicit Layer(const LayerParameter& param)
	: layer_param_(param){
	    phase_ = param.phase();
	    if(layer_param_.blobs_size() > 0){
		blobs_.resize(layer_param_.blobs_size());
		for(int i = 0; i < layer_param_.blobs_size(); i++){
		    blobs_[i].reset(new Blob<Dtype>());
		    blobs_[i]->FromProto(layer_param_.blobs(i));
		}
	    }
	}
    virtual ~Layer(){}

    virtual void layer_setup(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top){}
    
    virtual void reshape(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top) = 0;
    
    inline Dtype forward(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
    
    inline Dtype backward(const vector<Blob<Dtype>* >& top,
	    const vector<bool>& propagate_down,
	    const vector<Blob<Dtype>* >& bottom);

    vector<shared_ptr<Blob<Dtype> > >& blobs(){
	return blobs_;
    }



protected:
    LayerParameter layer_param_;
    Phase phase_;
    std::vector<std::shared_ptr<Blob<Dtype> > > blobs_;
    std::vector<bool> param_propagate_down_;
    vector<Dtype> loss_;

    virtual void forward_cpu(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top) = 0;

    virtual void forward_gpu(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top) = 0;

    virtual void backward_cpu(const vector<Blob<Dtype>* >& top,
	    const vector<Blob<Dtype>* >& bottom) = 0;
    
    virtual void backward_gpu(const vector<Blob<Dtype>* >& top,
	    const vector<Blob<Dtype>* >& bottom) = 0;


};


}


#endif
