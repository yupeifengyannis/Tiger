#ifndef TIGER_TEST_MAIN_HPP
#define TIGER_TEST_MAIN_HPP


#include <glog/logging.h>
#include "gtest/gtest.h"
#include "tiger/common.hpp"
//int main(int argc, char** argv);

namespace tiger{

template <typename TypeParam>
class MultiDeviceTest : public testing::Test{
public:
    typedef typename TypeParam::Dtype Dtype;
protected:
    MultiDeviceTest(){
	Tiger::set_mode(TypeParam::device);
    }
    virtual ~MultiDeviceTest(){}
};

typedef testing::Types<float, double> TestDtypes;

template <typename TypeParam>
struct CPUDevice{
    typedef TypeParam Dtype;
    static const Tiger::Brew device = Tiger::CPU;
};

template <typename Dtype>
class CPUDeviceTest : public MultiDeviceTest<CPUDevice<Dtype> >{
};

#ifdef CPU_ONLY
typedef testing::Types<CPUDevice<float>,
    CPUDevice<double> > TestDtypesAndDevices;
#else

template <typename TypeParam>
struct GPUDevice{
    typedef TypeParam Dtype;
    static const Tiger::Brew device = Tiger::GPU;
};

template <typename Dtype>
class GPUDeviceTest : public MultiDeviceTest<GPUDevice<Dtype> >{
};

typedef testing::Types<CPUDevice<float>, CPUDevice<float>,
	GPUDevice<float>, GPUDevice<double> > TestDtypesAndDevices;

}
#endif


#endif
