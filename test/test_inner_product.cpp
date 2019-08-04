#include <vector>
#include "tiger/blob.hpp"
#include "tiger/layers/inner_product_layer.hpp"
#include "tiger/tiger.pb.h"
using namespace tiger;

template <typename Dtype>
void test_layer_setup(std::vector<Blob<Dtype>* >& bottom_vec, 
	std::vector<Blob<Dtype>* >& top_vec, const LayerParameter layer_param){
    LOG(INFO) << layer_param.DebugString();
    std::shared_ptr<Layer<Dtype> > layer(new InnerProductLayer<Dtype>(layer_param));
    layer->setup(bottom_vec, top_vec);
    std::vector<std::shared_ptr<Blob<Dtype> > > blobs_vec = layer->blobs();
    LOG(INFO) << "check weight blobs ";
    LOG(INFO) << blobs_vec[0]->shape_string();
    std::vector<int> weight_shape = blobs_vec[0]->shape();
    const Dtype* weight_data = blobs_vec[0]->cpu_data();
    for(int i = 0; i < weight_shape[0]; i++){
	for(int j = 0; j < weight_shape[1]; j++){
	    std::cout << weight_data[i * weight_shape[1] + j] << " ";  
	}
	std::cout << std::endl;
    }
    LOG(INFO) << "check bias blobs ";
    LOG(INFO) << blobs_vec[1]->shape_string();
    std::vector<int> bias_shape = blobs_vec[1]->shape();
    const Dtype* bias_data = blobs_vec[1]->cpu_data();
    for(int i = 0; i < bias_shape[0]; i++){
	std::cout << bias_data[i] << std::endl;
    }
} 

template <typename Dtype>
void test_layer_reshape(std::vector<Blob<Dtype>* >& bottom_vec, 
	std::vector<Blob<Dtype>* >& top_vec, const LayerParameter& layer_param){
    std::shared_ptr<Layer<Dtype> > layer(new InnerProductLayer<Dtype>(layer_param));
    layer->setup(bottom_vec, top_vec);
    LOG(INFO) << top_vec[0]->shape_string();
}


int main(){
    std::vector<Blob<float>* > bottom_vec;
    bottom_vec.push_back(new Blob<float>(std::vector<int>{2,2,2,2}));
    std::vector<Blob<float>* > top_vec;
    top_vec.push_back(new Blob<float>());
    LayerParameter layer_param;
    InnerProductParameter* inner_param = layer_param.mutable_inner_param();
    inner_param->set_num_output(10);
    FillerParameter* weight_param = inner_param->mutable_weight_filler();
    weight_param->set_type("constant");
    weight_param->set_value(10);
    FillerParameter* bias_param = inner_param->mutable_bias_filler();
    bias_param->set_type("constant");
    bias_param->set_value(12);
    // test_layer_setup(bottom_vec, top_vec, layer_param);
    test_layer_reshape(bottom_vec, top_vec, layer_param);
}
