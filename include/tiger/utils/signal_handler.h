#ifndef TIGER_UTILS_SIGNAL_HANDLER_H
#define TIGER_UTILS_SIGNAL_HANDLER_H

#include "tiger/solver.hpp"

namespace tiger{

/// \biref SignalHandler这个类主要是用于获取信号的
class SignalHandler{
public:
    SignalHandler(SolverAction::Enum SIGINT_action, SolverAction::Enum SIGHUP_action);
    ~SignalHandler();
    /// \brief 利用回调函数返回一个检测信号的函数
    ActionCallBack get_action_function();
private:
    SolverAction::Enum check_for_signals() const;
    SolverAction::Enum SIGINT_action_;///< SIGINT信号
    SolverAction::Enum SIGHUP_action_;///< SIGHUP信号
    
};

}

#endif
