#include <algorithm>
#include "tiger/net.hpp"
#include "tiger/layer_factory.hpp"
namespace tiger{


static void show_map(std::map<std::string, int>& blob_name_to_idx){
    std::vector<std::pair<std::string, int> > vec(blob_name_to_idx.begin(), 
    blob_name_to_idx.end());
    std::sort(vec.begin(), vec.end(), [](std::pair<string, int>& lhs,
	std::pair<string, int>& rhs){
	    return lhs.second < rhs.second;
	    });
    for(auto item : vec){
	LOG(INFO) << item.first << "\t" << item.second;
    }
}

template <typename Dtype>
Net<Dtype>::Net(const NetParameter& param){

    init(param);
}
template <typename Dtype>
void Net<Dtype>::init(const NetParameter& in_param){
    NetParameter param = in_param;
    phase_ = param.state().phase();
    name_ = param.name();
    memory_used_ = 0;
    bottom_vecs_.resize(param.layer_size());
    bottom_id_vecs_.resize(param.layer_size());
    top_vecs_.resize(param.layer_size());
    top_id_vecs_.resize(param.layer_size());
    bottom_need_backward_.resize(param.layer_size());
    std::set<std::string> available_blobs;
    std::map<std::string, int> blob_name_to_idx;

    for(int layer_id = 0; layer_id < param.layer_size(); ++layer_id){
	if(!param.layer(layer_id).has_phase()){
	    param.mutable_layer(layer_id)->set_phase(phase_);
	}
	const LayerParameter layer_param = param.layer(layer_id);
	layers_.push_back(LayerRegistry<Dtype>::create_layer(layer_param));
	layer_names_.push_back(layer_param.name());
	LOG(INFO) << "creating layer: " << layer_param.name();

	// 需要添加层的bottom信息
	for(int bottom_id = 0; bottom_id < layer_param.bottom_size(); ++bottom_id){
	    const int blob_id = append_bottom(param, layer_id, bottom_id, 
	    &available_blobs, &blob_name_to_idx);
	}
	LOG(INFO) << "before append top blobs";
	show_map(blob_name_to_idx);
	int num_top = layer_param.top_size();
	for(int top_id = 0; top_id < num_top; ++top_id){
	    append_top(param, layer_id, top_id, &available_blobs, &blob_name_to_idx);
	}
	
	layers_[layer_id]->setup(bottom_vecs_[layer_id], top_vecs_[layer_id]);
	LOG(INFO) << "after append_top blobs ";
	show_map(blob_name_to_idx);
    }

    LOG(INFO) << "show blob names: ";
    for(auto item : blob_names_){
	std::cout << item << " ";
    }
    std::cout << std::endl;
}

template <typename Dtype>
void Net<Dtype>::append_top(const NetParameter& param, const int layer_id,
    const int top_id, std::set<string>* available_blobs,
    std::map<string, int>* blob_name_to_idx){
    
    std::shared_ptr<LayerParameter> layer_param(
	    new LayerParameter(param.layer(layer_id)));
    // 一般数据输入层是有两个top
    const std::string blob_name = (layer_param->top_size() > top_id) ?
	layer_param->top(top_id) : "(automatic)";
    
    if(blob_name_to_idx && layer_param->bottom_size() > top_id && 
	    blob_name == layer_param->bottom(top_id)){
	LOG(INFO) << blob_name << "----->" << blob_name << " (in-place)";
	// 如果是in-place计算，则不需要开辟新的内存空间了，否则
	// 我们像下面一下还需要new一个Blob的内存空间
	top_vecs_[layer_id].push_back(blobs_[(*blob_name_to_idx)[blob_name]].get());
	top_id_vecs_[layer_id].push_back((*blob_name_to_idx)[blob_name]);
    }
    else if(blob_name_to_idx && 
	    blob_name_to_idx->find(blob_name) != blob_name_to_idx->end()){
	LOG(FATAL) << blob_name << "produced by multiple sources";
    }
    else{

	std::shared_ptr<Blob<Dtype> > blob_pointer(new Blob<Dtype>());
	const int blob_id = blobs_.size();
	blobs_.push_back(blob_pointer);
	blob_names_.push_back(blob_name);
	blob_need_backward_.push_back(false);
	if(blob_name_to_idx){
	    (*blob_name_to_idx)[blob_name] = blob_id;
	}
	top_id_vecs_[layer_id].push_back(blob_id);
	top_vecs_[layer_id].push_back(blob_pointer.get());
    }
    if(available_blobs){
	available_blobs->insert(blob_name);
    }
}  

template <typename Dtype>
int Net<Dtype>::append_bottom(const NetParameter& param, const int layer_id,
const int bottom_id, std::set<string>* available_blobs,
    std::map<string, int>* blob_name_to_idx){
    
    std::shared_ptr<LayerParameter> layer_param(
	    new LayerParameter(param.layer(layer_id)));
    const std::string blob_name = layer_param->bottom(bottom_id);
    if(blob_name_to_idx->find(blob_name) == blob_name_to_idx->end()){
	LOG(FATAL) << "unknown bottom blob: " << blob_name << " ( layer name is  "
	    << layer_param->name() << " )";
    }
    const int blob_id = (*blob_name_to_idx)[blob_name];
    bottom_vecs_[layer_id].push_back(blobs_[blob_id].get());
    bottom_id_vecs_[layer_id].push_back(blob_id);
    bool need_backward = blob_need_backward_[blob_id];
    return blob_id;
} 


template <typename Dtype> 
void Net<Dtype>::append_param(const NetParameter& param, const int layer_id,
    const int param_id){

}

template <typename Dtype>
Dtype Net<Dtype>::forward_from_to(int start, int end){
    CHECK_GE(start, 0);
    CHECK_LT(end, this->layers_.size());
    Dtype loss = 0;
    for(int i = start; i <= end; ++i){
	Dtype layer_loss = layers_[i]->forward(bottom_vecs_[i], top_vecs_[i]);
	// 尽管每一层都有计算loss，但是一般如果不是最后层是不去真正计算loss的。
	loss += layer_loss;
    }
    return loss;
}

template <typename Dtype>
void Net<Dtype>::backward_from_to(int start, int end){
    CHECK_GE(end, 0);
    CHECK_LT(start, this->layers_.size());
    for(int i = start; i >= end; --i){
	if(this->layer_need_backward_[i]){
	    layers_[i]->backward(top_vecs_[i], bottom_need_backward_[i],
		    bottom_vecs_[i]);
	}
    }
}

template class Net<float>;
template class Net<double>;
}
