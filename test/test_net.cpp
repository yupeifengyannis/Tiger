#include <string>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include "tiger/net.hpp"
#include "tiger/tiger.pb.h"
#include "tiger/utils/io.hpp"
#include "tiger/net.hpp"

using namespace std;
using namespace tiger;

template <typename Dtype>
void test_net(const string& net_path){
    NetParameter net_param;
    tiger::read_proto_from_text_file(net_path, &net_param);
    LOG(INFO) << net_param.DebugString(); 
    Net<Dtype> net(net_param);
}

int main(int argc, char** argv){
    const string net_path = "/home/yupefieng/Documents/dl_framework/Tiger/test.prototxt";
    test_net<float>(net_path);
}





