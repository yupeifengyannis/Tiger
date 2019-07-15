#include <string>
#include <vector>
#include "layers/data_layer.hpp"
#include "utils/io.hpp"
#include "blob.hpp"

using namespace tiger;

int main(){
    std::string path = "./test_data/data_layer.prototxt";
    LayerParameter data_layer_param;
    tiger::read_proto_from_text_file_or_die(path, &data_layer_param);
    DataLayer<float> data_layer(data_layer_param);
    
    std::vector<Blob<float>* > bottom;
    std::vector<Blob<float>* > top;
    Blob<float>* blob_ptr = new Blob<float>();
    top.push_back(blob_ptr);
    
    data_layer.setup(bottom, top);



}

