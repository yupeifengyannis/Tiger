#ifndef TIGER_FILLER_HPP
#define TIGER_FILLER_HPP

#include "tiger/blob.hpp"
#include "tiger/syncedmem.hpp"
#include "tiger/utils/math_function.hpp"
#include "tiger/tiger.pb.h"

namespace tiger{

template <typename Dtype>
class Filler{
public:
    explicit Filler(const FillerParameter& param) : 
	filler_param_(param){}
    virtual ~Filler(){}
    virtual void fill_data(Blob<Dtype>* blob) = 0;
protected:
	FillerParameter filler_param_;
};


template <typename Dtype>
class ConstantFiller : public Filler<Dtype>{
    explicit ConstantFiller(const FillerParameter& param) : 
	Filler<Dtype>(param){}
    virtual void filld_data(Blob<Dtype>* blob) override{
	Dtype* data = blob->mutable_cpu_data();
	const int count = blob->count();
	const Dtype value = this->filler_param_.value();
	for(int i = 0; i < count; i++){
	    data[i] = value;
	}
	CHECK_EQ(this->filler_param_.sparse(), -1) << 
	    "sparsity not supported by this filler";
    }
};



}


#endif
