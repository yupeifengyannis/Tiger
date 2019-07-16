
#include "signal.h"

#include <glog/logging.h>
#include "tiger/utils/signal_handler.h"

namespace tiger{

static volatile sig_atomic_t got_sigint = false;
static volatile sig_atomic_t got_sighup = false;
static bool already_hooked_up = false;

void signal_handler(int signal){
    switch(signal){
	case SIGINT:
	    got_sigint = true;
	    break;
	case SIGHUP:
	    got_sighup = true;
	    break;
    }
}

void hookup_handler(){
    if(already_hooked_up){
	LOG(FATAL) << "signal handler is already hooked up";	
    }
    
    already_hooked_up = true;

    struct sigaction sa;
    // 绑定信号处理函数
    sa.sa_handler = &signal_handler;
    // 如果系统调用函数被信号打断，则信号处理完之后会重启系统调用函数
    sa.sa_flags = SA_RESTART;
    // 在信号处理的时候屏蔽一切信号
    sigfillset(&sa.sa_mask);
    if(sigaction(SIGHUP, &sa, NULL) == -1){
	LOG(FATAL) << "can't install SIGHUP handler";
    }
    if(sigaction(SIGINT, &sa, NULL) == -1){
	LOG(FATAL) << "can't install SIGINT handler";
    }

}

void unhook_handler(){
    if(already_hooked_up){
	struct sigaction sa;
	sa.sa_handler = SIG_DFL;
	sa.sa_flags = SA_RESTART;
	sigfillset(&sa.sa_mask);
	if(sigaction(SIGHUP, &sa, NULL) == -1){
	    LOG(FATAL) << "can't uninstall SIGHUP handler";
	}
	if(sigaction(SIGINT, &sa, NULL) == -1){
	    LOG(FATAL) << "can't uninstall SGIINT handler";
	}
	already_hooked_up = false;
    }
}

bool got_sigint_signal(){
    bool result = got_sigint;
    got_sigint = false;
    return result;
}

bool got_sighup_signal(){
    bool result = got_sighup;
    got_sighup = false;
    return result;
}


SignalHandler::SignalHandler(SolverAction::Enum SIGINT_action, 
	SolverAction::Enum SIGHUP_action) : 
    SIGINT_action_ (SIGINT_action),
    SIGHUP_action_ (SIGHUP_action){
	hookup_handler();
    }

SignalHandler::~SignalHandler(){
    unhook_handler();
}


SolverAction::Enum SignalHandler::check_for_signals() const{
    if(got_sigint_signal()){
	return SIGINT_action_;
    }
    if(got_sighup_signal()){
	return SIGHUP_action_;
    }
    return SolverAction::Enum::NONE;
}

ActionCallBack SignalHandler::get_action_function(){
    return std::bind(&SignalHandler::check_for_signals, this);
}







}

