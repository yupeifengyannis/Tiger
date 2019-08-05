#include <vector>
#include "tiger/blob.hpp"
#include "tiger/layers/inner_product_layer.hpp"
#include "tiger/tiger.pb.h"
#include "tiger/common.hpp"
#include "tiger/filler.hpp"
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

template <typename Dtype>
void test_forward_backward(){
    // create bottom_vec
    std::vector<Blob<Dtype>* > bottom_vec;
    std::shared_ptr<Blob<Dtype> > bottom_blob(new Blob<Dtype>(std::vector<int>{2,2,2,2}));
    bottom_vec.push_back(bottom_blob.get());
    FillerParameter bottom_filler_param;
    bottom_filler_param.set_type("constant");
    bottom_filler_param.set_value(10);
    std::shared_ptr<Filler<Dtype> > bottom_filler(get_filler<Dtype>(bottom_filler_param));
    bottom_filler->fill_data(bottom_blob.get());
    
    // create top_vec
    std::shared_ptr<Blob<Dtype> > top_blob(new Blob<Dtype>());
    std::vector<Blob<Dtype>* > top_vec;
    top_vec.push_back(top_blob.get());

    // create layerparamer
    LayerParameter layer_param;
    InnerProductParameter* inner_param = layer_param.mutable_inner_param();
    inner_param->set_num_output(10);
    FillerParameter* weight_param = inner_param->mutable_weight_filler();
    weight_param->set_type("serial");
    FillerParameter* bias_param = inner_param->mutable_bias_filler();
    bias_param->set_type("serial");
    
    // create inner product layer
    std::shared_ptr<Layer<Dtype> > layer(new InnerProductLayer<Dtype>(layer_param));
    layer->setup(bottom_vec, top_vec);
    Tiger::set_mode(Tiger::GPU);
    layer->forward(bottom_vec, top_vec);
    // show top_blob data
    std::vector<int> top_shape = top_vec[0]->shape();
    const Dtype* top_data = top_vec[0]->cpu_data();
    for(int i = 0; i < top_shape[0]; i++){
	for(int j = 0; j < top_shape[1]; j++){
	    std::cout << top_data[i * top_shape[1] + j] << " ";
	}
	std::cout << std::endl;
    }
    std::cout << std::endl;
}

template <typename Dtype>
void test_setup_reshape(){
    std::vector<Blob<Dtype>* > bottom_vec;
    bottom_vec.push_back(new Blob<Dtype>(std::vector<int>{2,2,2,2}));
    std::vector<Blob<Dtype>* > top_vec;
    top_vec.push_back(new Blob<Dtype>());
    LayerParameter layer_param;
    InnerProductParameter* inner_param = layer_param.mutable_inner_param();
    inner_param->set_num_output(10);
    FillerParameter* weight_param = inner_param->mutable_weight_filler();
    weight_param->set_type("constant");
    weight_param->set_value(10);
    FillerParameter* bias_param = inner_param->mutable_bias_filler();
    bias_param->set_type("constant");
    bias_param->set_value(12);
    test_layer_setup(bottom_vec, top_vec, layer_param);
    test_layer_reshape(bottom_vec, top_vec, layer_param);
}
int main(){
    // test_setup_reshape<float>();
    // test_setup_reshape<double>();
    test_forward_backward<float>();
}

