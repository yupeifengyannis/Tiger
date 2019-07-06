#ifndef SOLVER_HPP
#define SOLVER_HPP

#include <functional>

namespace tiger{

namespace SolverAction{
enum Enum{
    NONE = 0,
    STOP = 1,
    SNAPSHOT = 2
};
}

typedef std::function<SolverAction::Enum()> ActionCallBack;


}


#endif

