#include <opencv2/opencv.hpp>
#include "utils/io.hpp"
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
    cv::Mat img_mat = tiger::transform_datum_to_mat(datum);
    vector<int> top_shape = this->data_transformer_->infer_blob_shape(img_mat);
    this->transformed_data_.reshape(top_shape);
    LOG(INFO) << this->transformed_data_.shape_string(); 
    top_shape[0] = batch_size;
    top[0]->reshape(top_shape);
    for(int i = 0; i < this->prefetch_.size(); i++){
	this->prefetch_[i]->data_.reshape(top_shape);
    }
    if(this->output_label_){
	vector<int> label_shape(1, batch_size);
	top[1]->reshape(label_shape);
	for(int i = 0; i < this->prefetch_.size(); i++){
	    this->prefetch_[i]->label_.reshape(label_shape);
	}
    }
}

template <typename Dtype>
void DataLayer<Dtype>::next(){
    cursor_->next();
    if(!cursor_->valid()){
	LOG(INFO) << "restarting data prefetchig from start";
	cursor_->seek_to_first();
    }
    offset_++;
}

template <typename Dtype>
void DataLayer<Dtype>::load_batch(Batch<Dtype>* batch){
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());
    int batch_size = this->layer_param_.data_param().batch_size();
    
    Datum datum;
    cv::Mat img_mat;
    for(int item_id = 0; item_id < batch_size; item_id++){
	datum.ParseFromString(cursor_->value());
	img_mat = tiger::transform_datum_to_mat(datum);
	if(item_id == 0){
	    img_mat = tiger::transform_datum_to_mat(datum); 
	    vector<int> top_shape = this->data_transformer_->infer_blob_shape(img_mat);
	    this->transformed_data_.reshape(top_shape);
	    top_shape[0] = batch_size;
	    batch->data_.reshape(top_shape);
	}
	int offset = batch->data_.offset(item_id);
	Dtype* top_data = batch->data_.mutable_cpu_data();
	this->transformed_data_.set_cpu_data(top_data + offset);
	this->data_transformer_->transform(img_mat, &(this->transformed_data_));
	if(this->output_label_){
	    Dtype* top_label = batch->label_.mutable_cpu_data();
	    top_label[item_id] = datum.label();
	}
	next();
    }

}


template class DataLayer<float>;
template class DataLayer<double>;


}
