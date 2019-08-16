#include "tiger/layer_factory.hpp"
#include "tiger/layers/neuron/dropout_layer.hpp"
#include "tiger/tiger.pb.h"

using namespace tiger;
template <typename Dtype>
void test_dropout_layer(){
    DropoutParameter drop_param;
    drop_param.set_dropout_ratio(1);
    LayerParameter layer_param;
    layer_param.set_type("Dropout");
}

int main(){
}
