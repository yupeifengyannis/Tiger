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
    
    /// \brief setup这个函数应该是外界创建layer的接口
    void setup(const vector<Blob<Dtype>*> & bottom, 
	    const vector<Blob<Dtype>* >& top){
	// 检查输入输出的blobs的个数
	check_blob_counts(bottom, top);
	// 主要用于解析LayerParameter的参数，由子类重载解析 
	layer_setup(bottom, top);
	// 根据相应的参数来确定相关blob的维度
	reshape(bottom, top);

    }

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

    const LayerParameter& layer_param() const{
	return layer_param_;
    }

    inline Dtype loss(const int top_index) const{
	return (loss_.size() > top_index) ? loss_[top_index] : Dtype(0);
    }

    inline void set_loss(const int top_index, const Dtype value){
	if(loss_.size() <= top_index){
	    loss_.resize(top_index + 1, Dtype(0));
	}
	loss_[top_index] = value;
    }
    
    /// \brief 子类可以重载该函数来给出该类的类型
    virtual inline const char* type() const{
	return "";
    }
    
    /// \brief 返回输入的blob的确切blob个数
    virtual inline int exact_num_bottom_blobs() const{
	return -1;
    }
    
    /// \brief 返回输入blob的最小个数
    virtual inline int min_num_bottom_blobs() const{
	return -1;
    }
    
    /// \brief 返回输入blob的最大个数
    virtual inline int max_num_bottom_blobs() const{
	return -1;
    }

    /// \brief 返回输出blobs的确切个数
    virtual inline int exact_num_top_blobs() const{
	return -1;
    }
    
    /// \brief 返回输出blobs的最小个数
    virtual inline int min_num_top_blobs() const{
	return -1;
    }

    /// \brief 返回输出blbos的最大个数
    virtual inline int max_num_top_blobs() const{
	return -1;
    }
    /// \brief 返回bottom和top的blobs是否相等
    virtual inline bool eaqual_num_bottom_top_blobs() const{
	return false;
    }

    inline bool param_propagate_down(const int param_id){
	return (param_propagate_down_.size() > param_id) ?
	    param_propagate_down_[param_id] : false;
    }
    
    inline void set_param_propagate_down(const int param_id, const bool value){
	if(param_propagate_down_.size() <= param_id){
	    param_propagate_down_.resize(param_id + 1, true);
	}
	param_propagate_down_[param_id] = value;
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
    
    /// \brief 该函数就是用来检查我们输入和输出的blobs的个数是否和
    /// 设定的一致。
    virtual void check_blob_counts(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top){
	if(exact_num_bottom_blobs() >= 0){
	    CHECK_EQ(exact_num_bottom_blobs(), bottom.size()) << 
		"输入的bottom的blobs的个数和我们设定的不符合";
	}
	// TODO(需要检查其他的blob的情况)
    }

    inline void set_loss_weights(const vector<Blob<Dtype>* >& top){
	int num_loss_weights = layer_param_.loss_weight_size();
	if(num_loss_weights){
	    CHECK_EQ(top.size(), num_loss_weights) << 
		"loss weights的个数必须和top的个数一致";
	}
	for(int top_id = 0; top_id < top.size(); top_id++){
	    Dtype loss_weight = layer_param_.loss_weight(top_id);
	    if(loss_weight == Dtype(0)){
		continue;
	    }
	}
	// TODO(需要完成学习参数的0\1权重参数设置)
    }

};


}


#endif
