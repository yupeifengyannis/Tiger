#ifndef COMMON_HPP
#define COMMON_HPP

#include <string>

#include "utils/device_alternate.hpp"

using namespace std;
// TODO(Tiger单例类还有好多东西要去实现)

namespace tiger{
class Tiger{
public:
    ~Tiger();
    static Tiger& get();
    enum Brew{
	CPU,
	GPU
    };

    inline static Brew mode(){
	return get().mode_;
    }
protected:
    Brew mode_;

private:
    Tiger();
};
}




#endif
