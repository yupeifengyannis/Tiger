#ifndef LAYER_FACTORY_HPP
#define LAYER_FACTORY_HPP

#include <map>
#include <memory>
#include <vector>
#include "tiger.pb.h"
#include "common.hpp"

namespace tiger{

/// \brief采用用了前置申明的方法
template <typename Dtype>
class Layer;

/// \biref Layer类的注册表
template <typename Dtype>
class LayerRegistry{
public:
    /// \brief 定义了函数指针类型
    /// \param LayerParameter 层的参数
    /// std::shared_ptr<Layer<Dtype> > 返回一个类的实例 
    typedef std::shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter&);
    typedef std::map<string, Creator> CreatorRegistry;
    
    static CreatorRegistry& registry(){
	static CreatorRegistry* g_registry_ = new CreatorRegistry();
	return *g_registry_;
    }
    
    static void add_creator(const string& type, Creator creator){
	CreatorRegistry& registry_table = registry();
	CHECK_EQ(registry_table.count(type), 0) << 
	    "Layer type " << type << "already registered";
	registry_table[type] = creator;
    }

    static std::shared_ptr<Layer<Dtype> > create_layer(const LayerParameter& param){
	const string& type = param.type();
	CreatorRegistry& registry_table = registry();
	CHECK_EQ(registry_table.count(type), 1) << "unknown layer type: " << type << 
	    " (known types: " << layer_type_string() << ")";
	return registry_table[type](param);
    }

    static std::vector<string> layer_type_list(){
	CreatorRegistry& registry_table = registry();
	vector<string> layer_types;
	for(auto item : registry_table){
	    layer_types.push_back(item.first);
	}
	return layer_types;
    } 

private:
    LayerRegistry(){}
    static string layer_type_string(){
	vector<string> layer_types = layer_type_list();
	string layer_type_str;
	for(auto iter = layer_types.begin(); iter != layer_types.end(); iter++){
	    if(iter != layer_types.begin()){
		layer_type_str += ", ";
	    }
	    layer_type_str += *iter;
	}
	return layer_type_str;
    }
};

template <typename Dtype>
class LayerRegisterer{
public:
    typedef std::shared_ptr<Layer<Dtype> > (*Creator)(const LayerParameter);
    LayerRegisterer(const string& type, Creator creator){
	LayerRegistry<Dtype>::add_creator(type, creator);
    }
};

#define REGISTER_LAYER_CREATOR(type, creator)\
    static LayerRegisterer<float> g_creator_f##type(#type, creator<float>)\
    static LayerRegisterer<double> g_creator_g##type(#type, creator<double>)\

#define REGISTER_LAYER_CLASS(type)\
    template <typename Dtype>\
    std::shared_ptr<Layer<Dtype> > creator_##type##layer(const LayerParameter& param){\
	return std::shared_ptr<Layer<Dtype> >(new type##Layer<Dtype>(param));\
    }\
    REGISTER_LAYER_CREATOR(type, creator_##type##layer)

}



#endif
