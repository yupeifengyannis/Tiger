#ifndef TIGER_LAYERS_DATA_BASE_DATA_LAYER_HPP
#define TIGER_LAYERS_DATA_BASE_DATA_LAYER_HPP

#include <memory>
#include "tiger/layer.hpp"
#include "tiger/tiger.pb.h"
#include "tiger/data_transformer.hpp"
#include "tiger/utils/blocking_queue.hpp"

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
    TransformationParameter transform_param_;
    std::shared_ptr<DataTransformer<Dtype> > data_transformer_;
    bool output_label_;
};

template <typename Dtype>
class BasePrefetchingDataLayer : public BaseDataLayer<Dtype>{
public:
    explicit BasePrefetchingDataLayer(const LayerParameter& param);
    void layer_setup(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
    virtual void forward_cpu(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
    virtual void forward_gpu(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
protected:
    virtual void load_batch(Batch<Dtype>* batch) = 0;
    vector<std::shared_ptr<Batch<Dtype> > > prefetch_;
    tiger::BlockingQueue<Batch<Dtype>* > prefetch_free_;
    tiger::BlockingQueue<Batch<Dtype>* > prefetch_full_;
    Batch<Dtype>* prefetch_current_;
    Blob<Dtype> transformed_data_;
};


}

#endif

