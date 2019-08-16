#ifndef TIGER_NET_HPP
#define TIGER_NET_HPP

#include <string>
#include <memory>
#include <vector>
#include <map>
#include <set>

#include "tiger/common.hpp"
#include "tiger/tiger.pb.h"
#include "tiger/layer.hpp"
#include "tiger/blob.hpp"

namespace tiger{
template <typename Dtype>
class Net{
public:
    explicit Net(const NetParameter& param);
    virtual ~Net(){}

    void init(const NetParameter& param);
    const std::vector<Blob<Dtype>* >& forward(Dtype* loss = nullptr);
    void backward();

protected:
    void append_top(const NetParameter& param, const int layer_id,
	    const int top_id, std::set<string>* available_blobs,
	    std::map<string, int>* blob_name_to_idx);    
    
    int append_bottom(const NetParameter& param, const int layer_id,
	    const int top_id, std::set<string>* available_blobs,
	    std::map<string, int>* blob_name_to_idx);    
    
    void append_param(const NetParameter& param, const int layer_id,
	    const int param_id);
protected:
    std::string name_;
    Phase phase_;
    // 层的相关信息就是有下面四个成员变量来维护
    std::vector<std::shared_ptr<Layer<Dtype> > > layers_;
    std::vector<std::string> layer_names_;
    std::vector<bool> layer_need_backward_;
    std::map<std::string, int> layer_names_index_;
    // 每一层之间我们都是要经过数据传递，而这个数据传递的
    // blobs都是由下面四个变量来进行维护
    std::vector<std::shared_ptr<Blob<Dtype> > > blobs_;
    std::vector<std::string> blob_names_;
    std::vector<bool> blob_need_backward_;
    std::map<std::string, int> blob_names_index_;
    // 每层的输入数据
    std::vector<std::vector<Blob<Dtype>* > > bottom_vecs_;
    std::vector<std::vector<int> > bottom_id_vecs_;
    std::vector<std::vector<bool> > bottom_need_backward_;
    // 每层的输出数据
    std::vector<std::vector<Blob<Dtype>* > > top_vecs_;
    std::vector<std::vector<int> > top_id_vecs_;
    // 每层需要学习的权重参数
    std::vector<std::shared_ptr<Blob<Dtype> > > params_;
    std::vector<Blob<Dtype>* > learnable_params_;
    vector<int> learnable_index_;
    vector<float> params_lr_;
    vector<float> params_weight_decay_;
    size_t memory_used_;

};

}
#endif

