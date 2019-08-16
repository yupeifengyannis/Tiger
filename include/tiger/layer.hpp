#ifndef TIGER_LAYER_HPP
#define TIGER_LAYER_HPP

#include <memory>
#include <vector>
#include "tiger/common.hpp"
#include "tiger/blob.hpp"
#include "tiger/tiger.pb.h"
#include "tiger/layer_factory.hpp"
#include "tiger/utils/math_function.hpp"

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
		    blobs_[i]->from_proto(layer_param_.blobs(i));
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
	// 设置相关loss的权重
	set_loss_weights(top);
    }

    virtual void layer_setup(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top){}

    virtual void reshape(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top) = 0;

    inline Dtype forward(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);

    inline void backward(const vector<Blob<Dtype>* >& top,
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
	    const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom) = 0;

    virtual void backward_gpu(const vector<Blob<Dtype>* >& top,
	    const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom) = 0;

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
	// 这里一般只有loss层才会去设置该层的损失参数
	int num_loss_weights = layer_param_.loss_weight_size();
	if(num_loss_weights){
	    CHECK_EQ(top.size(), num_loss_weights) << 
		"loss weights的个数必须和top的个数一致";
	    for(int top_id = 0; top_id < top.size(); top_id++){
		Dtype loss_weight = layer_param_.loss_weight(top_id);
		if(loss_weight == Dtype(0)){
		    continue;
		}
		set_loss(top_id, loss_weight);
		const int count = top[top_id]->count();
		Dtype* loss_multiplier = top[top_id]->mutable_cpu_diff();
		tiger_set(count, loss_weight, loss_multiplier);
	    }
	}
    }

};

template <typename Dtype>
inline Dtype Layer<Dtype>::forward(const vector<Blob<Dtype>* >& bottom, 
	const vector<Blob<Dtype>* >& top){
    Dtype loss = 0;
    if(Tiger::mode() == Tiger::CPU){
	forward_cpu(bottom, top);
    }
    else{
	forward_gpu(bottom, top);
    }
    for(unsigned int top_id = 0; top_id < top.size(); ++top_id){
	if(!this->loss(top_id)){
	    continue;
	}
	const int count = top[top_id]->count();
	const Dtype* data = top[top_id]->cpu_data();
	// 前面已经将所有的loss weight设置到top blob的cpu_diff中。
	// 在实际反向传播的时候也不会去用到这个cpu_diff的，因为
	// 我们实际上在loss层的位置放置了一个diff_成员变量来用于
	// 方向传播
	const Dtype* loss_weights = top[top_id]->cpu_diff();
	Dtype blob_loss = 0;
	tiger_gpu_dot(count, data, loss_weights, &blob_loss);
	loss += blob_loss;
    }
    return loss;
}

template <typename Dtype>
inline void Layer<Dtype>::backward(const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom){
    if(Tiger::mode() == Tiger::CPU){
	backward_cpu(top, propagate_down, bottom);
    }
    else{
	backward_gpu(top, propagate_down, bottom);
    }
}






}


#endif
