#ifndef TIGER_LAYERS_NEURON_LAYER
#define TIGER_LAYERS_NEURON_LAYER


#include "tiger/layer.hpp"

namespace tiger{


template <typename Dtype>
class NeuronLayer : public Layer<Dtype>{
public:
    explicit NeuronLayer(const LayerParameter& param) : 
	Layer<Dtype>(param){}
    virtual void reshape(const vector<Blob<Dtype>* >& bottom, 
	    const vector<Blob<Dtype>* >& top) override;

    virtual inline int exact_num_bottom_blobs() const override {
	return 1;
    }
    virtual inline int exact_num_top_blobs() const override{
	return 1;
    }

};


}



#endif
