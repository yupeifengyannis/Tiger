#include <iostream>
#include "tiger/utils/signal_handler.h"
using namespace std;
using namespace tiger;

int main(){
    tiger::SignalHandler signal_handler(SolverAction::STOP, SolverAction::SNAPSHOT);
    ActionCallBack get_signal = signal_handler.get_action_function();
    while(1){
	if(get_signal() == SolverAction::STOP){
	    cout << "捕捉到SIGINT信号" << endl;
	}
    }
}
