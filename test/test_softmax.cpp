#include "tiger/blob.hpp"
#include "tiger/layer_factory.hpp"
#include "tiger/layers/cudnn/cudnn_softmax.hpp"
#include "tiger/layers/neuron/softmax_layer.hpp"
#include "tiger/tiger.pb.h"
#include "tiger/common.hpp"
using namespace tiger;

template <typename Dtype>
void test_softmax(){
    LayerParameter layer_param;
    layer_param.set_type("Softmax");
    vector<int> shape_vec{1,1,1,4};
    Blob<Dtype> bottom_data(shape_vec);
}

int main(){
    test_softmax<float>();    
}

