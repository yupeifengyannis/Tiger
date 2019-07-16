#include <memory>
#include <string>
#include "tiger/utils/io.hpp"
#include "tiger/utils/leveldb.hpp"
#include "tiger/data_transformer.hpp"
#include "tiger/tiger.pb.h"
using namespace std;
using namespace tiger;

int main(){
    string source = "/home/yupefieng/Documents/dl_framework/Tiger/test_data/mnist";
    std::shared_ptr<LevelDB> db(new LevelDB());
    db->open(source, Mode::READ); 
    std::shared_ptr<LevelDBCursor> cursor(db->new_cursor());
    Datum datum;
    tiger::TransformationParameter param;
    DataTransformer<float> data_transformer(param, tiger::TEST);
    datum.ParseFromString(cursor->value());
    LOG(INFO) << "datum encoded is " << datum.encoded();
    cv::Mat mat = decode_datum_to_mat(datum);
    
    vector<int> shape = data_transformer.infer_blob_shape(mat);
    Blob<float> blob(shape);
    LOG(INFO) << blob.shape_string();
    while(cursor->valid()){
	datum.ParseFromString(cursor->value());
	cv::Mat tmp_mat = tiger::transform_datum_to_mat(datum);
	cv::resize(tmp_mat, tmp_mat, cv::Size(320, 320));
	cv::imshow("img", tmp_mat);
	cv::waitKey(1);
	cursor->next();
    }
}




