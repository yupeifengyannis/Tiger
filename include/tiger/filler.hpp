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
public:
    explicit ConstantFiller(const FillerParameter& param) : 
	Filler<Dtype>(param){}
    virtual void fill_data(Blob<Dtype>* blob){
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

template <typename Dtype>
class SerialFiller : public Filler<Dtype>{
public:
    explicit SerialFiller(const FillerParameter& param) : 
	Filler<Dtype>(param){}
    virtual void fill_data(Blob<Dtype>* blob){
	Dtype* data = blob->mutable_cpu_data();
	const int count = blob->count();
	for(int i = 0; i < count; i++){
	    data[i] = i;
	}
    }
};


template <typename Dtype>
inline Filler<Dtype>* get_filler(const FillerParameter& filler_param){
    if("constant" == filler_param.type()){
	return new ConstantFiller<Dtype>(filler_param);
    }
    else if("serial" == filler_param.type()){
	return new SerialFiller<Dtype>(filler_param);
    }
    else{
	//TODO()
	LOG(FATAL) << "filler type " << filler_param.type() << " is not existed";
    }
    
}

}


#endif
