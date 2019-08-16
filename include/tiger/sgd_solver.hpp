#ifndef TIGER_SGD_SOLVER_HPP
#define TIGER_SGD_SOLVER_HPP
#include <memory>
#include <vector>
#include "tiger/solver.hpp"
#include "tiger/blob.hpp"

namespace tiger{

template <typename Dtype>
class SGDSolver : public Solver<Dtype>{
public:
    explicit SGDSolver(const SolverParameter& solver_param) : 
	Solver<Dtype>(solver_param){
	    pre_solve();
	}
    virtual inline const char* type() const{
	return "SGD";
    }
    
    virtual void apply_update();
protected:
    void pre_solve();
    virtual void normalize(int param_id);
    virtual void regularize(int param_id);
    virtual void snapshot_solver_state(const std::string& model_filename);
    virtual void restore_solver_state_from_binary_proto(const std::string& state_file);
protected:
    std::vector<std::shared_ptr<Blob<Dtype> > > history_;
    std::vector<std::shared_ptr<Blob<Dtype> > > update_;
    std::vector<std::shared_ptr<Blob<Dtype> > > temp_;
};




}



#endif
