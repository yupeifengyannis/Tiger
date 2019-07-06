#include <memory>
#include "common.hpp"

namespace tiger{

std::shared_ptr<Tiger> instance_;

Tiger::Tiger() : 
    mode_(Tiger::CPU){
    
    }

Tiger::~Tiger(){

}

Tiger& Tiger::get(){
    if(!instance_.get()){
	instance_.reset(new Tiger());
    }
    return *instance_;
}

}
