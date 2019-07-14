
#include "layers/data_layer.hpp"
#include "tiger.pb.h"

namespace tiger{

template <typename Dtype>
DataLayer<Dtype>::DataLayer(const LayerParameter& param) : 
    BasePrefetchingDataLayer<Dtype>(param),
    offset_(0){
	db_.reset(new LevelDB());
	// open level db dataset for reading
	db_->open(param.data_param().source(), Mode::READ);
	cursor_.reset(db_->new_cursor());
    }

template <typename Dtype>
void DataLayer<Dtype>::data_layer_setup(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    const int batch_size = this->layer_param_.data_param().batch_size();
    Datum datum;
    datum.ParseFromString(cursor_->value());
    
}

template <typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch){
    
}



}
