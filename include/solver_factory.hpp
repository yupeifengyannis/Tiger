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
class Solver;

/// \biref Solver类的注册表
template <typename Dtype>
class SolverRegistry{
public:
    /// \brief 定义了函数指针类型
    /// \param SolverParameter 层的参数
    /// std::shared_ptr<Solver<Dtype> > 返回一个类的实例 
    typedef std::shared_ptr<Solver<Dtype> > (*Creator)(const SolverParameter&);
    typedef std::map<string, Creator> CreatorRegistry;
    
    static CreatorRegistry& registry(){
	static CreatorRegistry* g_registry_ = new CreatorRegistry();
	return *g_registry_;
    }
    
    static void add_creator(const string& type, Creator creator){
	CreatorRegistry& registry_table = registry();
	CHECK_EQ(registry_table.count(type), 0) << 
	    "Solver type " << type << "already registered";
	registry_table[type] = creator;
    }

    static std::shared_ptr<Solver<Dtype> > create_solver(const SolverParameter& param){
	const string& type = param.type();
	CreatorRegistry& registry_table = registry();
	CHECK_EQ(registry_table.count(type), 1) << "unknown solver type: " << type << 
	    " (known types: " << solver_type_string() << ")";
	return registry_table[type](param);
    }

    static std::vector<string> solver_type_list(){
	CreatorRegistry& registry_table = registry();
	vector<string> solver_types;
	for(auto item : registry_table){
	    solver_types.push_back(item.first);
	}
	return solver_types;
    } 

private:
    SolverRegistry(){}
    static string solver_type_string(){
	vector<string> solver_types = solver_type_list();
	string solver_type_str;
	for(auto iter = solver_types.begin(); iter != solver_types.end(); iter++){
	    if(iter != solver_types.begin()){
		solver_type_str += ", ";
	    }
	    solver_type_str += *iter;
	}
	return solver_type_str;
    }
};

template <typename Dtype>
class SolverRegisterer{
public:
    typedef std::shared_ptr<Solver<Dtype> > (*Creator)(const SolverParameter);
    SolverRegisterer(const string& type, Creator creator){
	SolverRegistry<Dtype>::add_creator(type, creator);
    }
};

#define REGISTER_LAYER_CREATOR(type, creator)\
    static SolverRegisterer<float> g_creator_f##type(#type, creator<float>)\
    static SolverRegisterer<double> g_creator_g##type(#type, creator<double>)\

#define REGISTER_LAYER_CLASS(type)\
    template <typename Dtype>\
    std::shared_ptr<Solver<Dtype> > creator_##type##solver(const SolverParameter& param){\
	return std::shared_ptr<Solver<Dtype> >(new type##Solver<Dtype>(param));\
    }\
    REGISTER_LAYER_CREATOR(type, creator_##type##solver)

}



#endif
