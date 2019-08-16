#include "tiger/solver.hpp"

namespace tiger{

template <typename Dtype>
void Solver<Dtype>::set_action_function(ActionCallBack func){
    action_request_function_ = func;
}

template <typename Dtype>
SolverAction::Enum Solver<Dtype>::get_requested_action(){
    if(action_request_function_){
	// 通过回调函数来返回相应的结果
	return action_request_function_();
    }
    return SolverAction::NONE;
}

template <typename Dtype>
Solver<Dtype>::Solver(const SolverParameter& param) : 
    net_(),
    requested_early_exit_(false){
	init(param);
    }

template <typename Dtype>
void Solver<Dtype>::init(const SolverParameter& param){
    LOG(INFO) << param.DebugString();
    init_train_net();
    init_test_net();
    iter_ = 0;
    current_step_ = 0;
}

template <typename Dtype>
void Solver<Dtype>::init_train_net(){
    NetParameter net_param;
    NetState net_state;
    net_state.set_phase(TRAIN);
    net_param.mutable_state()->CopyFrom(net_state);
    net_.reset(new Net<Dtype>(net_param));
}


}
