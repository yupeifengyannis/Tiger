#include "tiger/blob.hpp"
#include "tiger/layers/relu_layer.hpp"
#include "tiger/layers/cudnn_relu_layer.hpp"
#include "tiger/filler.hpp"
#include "tiger/tiger.pb.h"
#include "tiger/common.hpp"

using namespace tiger;

template <typename Dtype>
void test_setup(Layer<Dtype>* layer, const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    layer->setup(bottom, top);
}
template <typename Dtype>
void test_forward(Layer<Dtype>* layer, const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    layer->forward(bottom, top);
    const Dtype* top_data = top[0]->cpu_data();
    const int count = top[0]->count();
    LOG(INFO) << "finish forward";
    for(int i = 0; i < count; i++){
	std::cout << top_data[i] << " ";
    }
    std::cout << std::endl;
}

template <typename Dtype>
void test_backward(Layer<Dtype>* layer, const vector<Blob<Dtype>* >& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>* >& bottom){
    Dtype* top_diff = top[0]->mutable_cpu_diff();
    const int count = top[0]->count();
    for(int i = 0; i < count; i++){
	top_diff[i] = 2;
    }
    layer->backward(top, propagate_down, bottom);
    LOG(INFO) << "finish backward";
    const Dtype* bottom_diff = bottom[0]->cpu_diff();
    for(int i = 0; i < count; i++){
	std::cout << bottom_diff[i] << " ";
    }
    std::cout << std::endl;
}

template <typename Dtype>
void test(){
    Tiger::set_mode(Tiger::GPU);    
    LayerParameter layer_param;
    ReLUParameter* relu_param = layer_param.mutable_relu_param();
    relu_param->set_negative_slop(0);
    FillerParameter filler_param;
    filler_param.set_type("serial");
    std::shared_ptr<Filler<Dtype> > filler(get_filler<Dtype>(filler_param));
    std::vector<int> shape_vec{1,1,1,4};
    Blob<Dtype> bottom_data(shape_vec);
    filler->fill_data(&bottom_data);
    Blob<Dtype> top_data;
    std::vector<Blob<Dtype>* > bottom_vec;
    std::vector<Blob<Dtype>* > top_vec;
    bottom_vec.push_back(&bottom_data);
    top_vec.push_back(&top_data);
    std::vector<bool> propagate_down{true}; 
    std::shared_ptr<Layer<Dtype> > layer;
    layer.reset(new CuDNNReLULayer<Dtype>(layer_param));
    test_setup<Dtype>(layer.get(), bottom_vec, top_vec);
    test_forward<Dtype>(layer.get(), bottom_vec, top_vec);
    test_backward<Dtype>(layer.get(), top_vec, propagate_down, bottom_vec);
}

int main(){
    test<float>();
}
