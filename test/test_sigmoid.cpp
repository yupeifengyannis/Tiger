#include <glog/logging.h>
#include <iostream>
#include "tiger/layers/sigmoid_layer.hpp"
#include "tiger/layers/cudnn_sigmoid_layer.hpp"
#include "tiger/tiger.pb.h"
#include "tiger/common.hpp"

using namespace tiger;
void test_sigmoid_reshape(Layer<float>* layer, vector<Blob<float>* >& bottom_vec,
	vector<Blob<float>* >& top_vec){
    LOG(INFO) << "before layer setup";
    LOG(INFO) << "bottom_data.shape_string() is " << bottom_vec[0]->shape_string();
    LOG(INFO) << "top_data.shape_string() is " << top_vec[0]->shape_string();
    layer->setup(bottom_vec, top_vec); 
    LOG(INFO) << "after layer setup";
    LOG(INFO) << "bottom_data.shape_string() is " << bottom_vec[0]->shape_string();
    LOG(INFO) << "top_data.shape_string() is " << top_vec[0]->shape_string();
}

void test_sigmoid_forward(Layer<float>* layer, const vector<Blob<float>* >& bottom_vec,
	const vector<Blob<float>* >& top_vec){
    Tiger::set_mode(Tiger::GPU);
    layer->forward(bottom_vec, top_vec);
    const float* data = top_vec[0]->cpu_data();
    int count = top_vec[0]->count();
    for(int i = 0; i < count; i++){
	std::cout << data[i] << " ";
    }
    std::cout << std::endl;
    
}

void test_sigmoid_backward(Layer<float>* layer, const vector<Blob<float>* >& top_vec,
	const vector<bool>& propagate_down, const vector<Blob<float>* >& bottom_vec){
    Tiger::set_mode(Tiger::GPU);
    float* data_diff = top_vec[0]->mutable_cpu_diff();
    int count = top_vec[0]->count();
    for(int i = 0; i < count; i++){
	data_diff[i] = 2;
    }
    layer->backward(top_vec, propagate_down, bottom_vec);
    LOG(INFO) << "after backward_cpu";
    const float* bottom_diff = bottom_vec[0]->cpu_diff();
    for(int i = 0; i < count; i++){
	std::cout << bottom_diff[i] << " ";
    }
    std::cout << std::endl;
}

int main(){
    LayerParameter layer_param;
    std::vector<int> shape_data{1,1,1,4};
    Blob<float> bottom_data(shape_data);
    Blob<float> top_data;
    std::vector<Blob<float>* > bottom_vec;
    bottom_vec.push_back(&bottom_data);
    std::vector<Blob<float>* > top_vec;
    top_vec.push_back(&top_data);
    std::shared_ptr<Layer<float> > sigmoid_layer;
    // sigmoid_layer.reset(new CuDNNSigmoidLayer<float>(layer_param));
    sigmoid_layer.reset(new SigmoidLayer<float>(layer_param));
    int count = bottom_data.count();
    float* data = bottom_data.mutable_cpu_data();
    for(int i = 0; i < count; i++){
	data[i] = 1;
    }
    test_sigmoid_reshape(&*sigmoid_layer, bottom_vec, top_vec);
    test_sigmoid_forward(&*sigmoid_layer, bottom_vec, top_vec);
    std::vector<bool> propagate_down{true};
    test_sigmoid_backward(&*sigmoid_layer, top_vec,
	    propagate_down, bottom_vec);
}
