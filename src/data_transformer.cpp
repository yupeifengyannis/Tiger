#include "data_transformer.hpp"
#include "utils/io.hpp"
namespace tiger{

template <typename Dtype>
DataTransformer<Dtype>::DataTransformer(const TransformationParameter& param, Phase phase) : 
    param_(param), phase_(phase){
	init_rand();
	if(this->param_.has_mean_file()){
	    CHECK_EQ(param_.mean_value_size(), 0) << 
		"can't specify mean_file and mean value at same time";
	    const string& mean_file = param.mean_file();
	    BlobProto blob_proto;
	    tiger::read_proto_from_binary_file_or_die(mean_file, &blob_proto);
	    data_mean_.from_proto(blob_proto);
	}
	if(param_.mean_value_size() > 0){
	    CHECK(param_.has_mean_file() == false) << 
		"can't specify mean_file and mean value at same time";
	    for(int c = 0; c < param_.mean_value_size(); c++){
		mean_values_.push_back(param_.mean_value(c));
	    }
	}
    }

template <typename Dtype>
void DataTransformer<Dtype>::transform(const vector<cv::Mat>& mat_vector,
	Blob<Dtype>* transformed_blob){
    const int mat_num = mat_vector.size();
    const int num = transformed_blob->num();
    const int channels = transformed_blob->channels();
    const int height = transformed_blob->height();
    const int width = transformed_blob->width();
    
    CHECK_GT(mat_num, 0) << "mat vector is empty";
    CHECK_EQ(mat_num, num) << 
	"the size of mat_vector must be equals to transformed_blob->num()";
    vector<int> blob_vec{1, channels, height, width};
    Blob<Dtype> uni_blob(blob_vec);
    for(int item_id = 0; item_id < mat_num; item_id++){
	int offset = transformed_blob->offset(item_id);
	uni_blob.set_cpu_data(transformed_blob->mutable_cpu_data() + offset);
	transform(mat_vector[item_id], &uni_blob);
    }
}

template <typename Dtype>
void DataTransformer<Dtype>::transform(const cv::Mat& cv_img, 
	Blob<Dtype>* transformed_blob){
    const int crop_size = param_.crop_size();
    const int img_channels = cv_img.channels();
    const int img_height = cv_img.rows;
    const int img_width = cv_img.cols;
    
    const int blob_channels = transformed_blob->channels();
    const int blob_height = transformed_blob->height();
    const int blob_width = transformed_blob->width();
    const int blob_nums = transformed_blob->num();

    CHECK_EQ(blob_channels, img_channels);
    CHECK_LE(blob_height, img_height);
    CHECK_LE(blob_width, img_width);
    CHECK_GE(blob_nums, 1);
    
    CHECK(cv_img.depth() == CV_8U) << "image data type must be unsigned byte";
    
    const Dtype scale = param_.scale();
    const bool do_mirror = param_.mirror() && rand(2); 
    const bool has_mean_values = mean_values_.size() > 0;
    const bool has_mean_file = param_.has_mean_file();

    CHECK_GT(img_channels, 0);
    CHECK_GE(img_height, crop_size);
    CHECK_GE(img_width, crop_size);
    
    Dtype* mean = NULL;
    if(has_mean_file){
	CHECK_EQ(img_channels, data_mean_.channels());
	CHECK_EQ(img_height, data_mean_.height());
	CHECK_EQ(img_width, data_mean_.width());
	mean = data_mean_.mutable_cpu_data();
	LOG(INFO) << "has mean files";
    }
    
    if(has_mean_values){
	CHECK(mean_values_.size() == 1 || mean_values_.size() == img_channels) << 
	    "specify either 1 mean_value or as many as channels: " << img_channels;
	if(img_channels > 1 && mean_values_.size() == 1){
	    for(int c = 1; c < img_channels; c++){
		mean_values_.push_back(mean_values_[0]);
	    }
	}
	LOG(INFO) << "has mean values";
    }
    
    int h_off = 0;
    int w_off = 0;
    cv::Mat cv_cropped_img = cv_img;
    if(crop_size){
	CHECK_EQ(crop_size, blob_height);
	CHECK_EQ(crop_size, blob_width);
	if(phase_ == TRAIN){
	    h_off = rand(img_height - crop_size + 1);
	    w_off = rand(img_width - crop_size + 1);
	}
	else{
	    h_off = (img_height - crop_size) / 2;
	    w_off = (img_width - crop_size) / 2;
	}
	cv::Rect roi(w_off, h_off, crop_size, crop_size);
	cv_cropped_img = cv_img(roi);
    }
    else{
	CHECK_EQ(img_height, blob_height);
	CHECK_EQ(img_width, blob_width);
    }
    
    CHECK(cv_cropped_img.data);
    
    Dtype* transformed_data = transformed_blob->mutable_cpu_data();

    int top_index = 0;
    // 在opencv中图片的存储的顺序为HWC
    for(int h = 0; h < blob_height; h++){
	const uchar* ptr = cv_cropped_img.ptr<uchar>(h);
	int img_index = 0;
	for(int w = 0; w < blob_width; w++){
	    for(int c = 0; c < blob_channels; c++){
		if(do_mirror){
		    top_index = (c * blob_height + h) * blob_width + (blob_width - 1 - w);
		}
		else{
		    top_index = (c * blob_height + h) * blob_width + w;
		}
		Dtype pixel = static_cast<Dtype>(ptr[img_index++]);
		if(has_mean_file){
		    LOG(INFO) << "has_mean_file";
		    int mean_index = (c * img_height + h_off + h) * img_width + w_off + w;
		    transformed_data[top_index] = (pixel - mean[mean_index]) * scale;
		}
		else{
		    if(has_mean_values){
			transformed_data[top_index] = 
			    (pixel - mean_values_[c]) * scale;
		    }
		    else{
			transformed_data[top_index] = pixel * scale;
		    }
		}
	    }
	}
    }
}


template <typename Dtype>
vector<int> DataTransformer<Dtype>::infer_blob_shape(const cv::Mat& cv_img){
    const int crop_size = param_.crop_size();
    const int img_channels = cv_img.channels();
    const int img_height = cv_img.rows;
    const int img_width = cv_img.cols;
    CHECK_GT(img_channels, 0);
    CHECK_GE(img_height, crop_size);
    CHECK_GE(img_width, crop_size);
    vector<int> shape(4);
    shape[0] = 1;
    shape[1] = img_channels;
    shape[2] = (crop_size) ? crop_size : img_height;
    shape[3] = (crop_size) ? crop_size : img_width;
    return shape;
}

template <typename Dtype>
vector<int> DataTransformer<Dtype>::infer_blob_shape(const vector<cv::Mat>& mat_vector){
    const int num = mat_vector.size();
    CHECK_GT(num, 0);
    vector<int> shape = infer_blob_shape(mat_vector[0]);
    shape[0] = num;
    return shape;
}

template <typename Dtype>
void DataTransformer<Dtype>::init_rand(){
    const bool needs_rand = param_.mirror() || 
	(phase_ == TRAIN && param_.crop_size());
    if(needs_rand){
	Generator rng_gen;
	const unsigned int rng_seed = (*rng_gen.rng())();
	rng_.reset(new Tiger::RNG(rng_seed));
    }
    else{
	rng_.reset();
    }
}

template <typename Dtype>
int DataTransformer<Dtype>::rand(int n){
    CHECK(rng_);
    CHECK_GT(n, 0);
    rng_t* rng = static_cast<rng_t*>(rng_->generator());
    return ((*rng)() % n);
}

template class DataTransformer<float>;
template class DataTransformer<double>;

}

