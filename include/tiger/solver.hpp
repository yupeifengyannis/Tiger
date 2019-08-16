#ifndef TIGER_SOLVER_HPP
#define TIGER_SOLVER_HPP
#include <memory>
#include <functional>
#include "tiger/tiger.pb.h"
#include "tiger/net.hpp"

namespace tiger{

namespace SolverAction{
enum Enum{
    NONE = 0,
    STOP = 1,
    SNAPSHOT = 2
};
}

typedef std::function<SolverAction::Enum()> ActionCallBack;

template <typename Dtype>
class Solver{
public:
    explicit Solver(const SolverParameter& solver_param);
    void init(const SolverParameter& solver_param);
    void init_train_net();
    void init_test_net();
    void set_action_function(ActionCallBack func);
    SolverAction::Enum get_requested_action();
    
    virtual void solve_net(const char* resume_file = nullptr);
    void solve_net(const std::string& resume_file){
	solve_net(resume_file.c_str());
    }
    
    void step(int iters);
    void restore(const char* resume_file);

    void snapshot();
    virtual ~Solver(){}

    inline std::shared_ptr<Net<Dtype> > net(){
	return net_;
    }
    inline std::vector<std::shared_ptr<Net<Dtype> > > tset_nets(){
	return test_nets_;
    }
    inline int iter(){
	return iter_;
    }

    virtual inline const char* type() const{
	return "";
    }
    virtual void apply_update() = 0;

protected:
    std::string snapshot_filename(const std::string& ext);
    std::string snapshot_to_binary_proto();
    
    virtual void snapshot_solver_state(const std::string& model_filename) = 0;
    virtual void restore_solver_state_from_binary_proto(const std::string& state_file) = 0;


protected:
    SolverParameter solver_param_;
    int iter_;
    int current_step_;
    std::shared_ptr<Net<Dtype> > net_;
    std::vector<std::shared_ptr<Net<Dtype> > > test_nets_;
    std::vector<Dtype> loss_vec_;
    Dtype smooth_loss_;
    ActionCallBack action_request_function_;
    bool requested_early_exit_;
};

}
#endif
