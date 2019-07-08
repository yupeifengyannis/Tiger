#include "blob.hpp"


namespace tiger{

template <typename Dtype>
void Blob<Dtype>::reshape(const vector<int>& shape){
    CHECK_LE(shape.size(), k_max_blob_axes);
    count_ = 1;
    shape_.resize(shape.size());
    if(!shape_data_ || shape_data_->size() < shape.size() * sizeof(int)){
	shape_data_.reset(new SyncedMemory(shape.size() * sizeof(int)));
    }
    int* shape_data = static_cast<int*>(shape_data_->mutable_cpu_data());
    for(unsigned int i = 0; i < shape.size(); i++){
	CHECK_GE(shape[i], 0);
	if(count_ != 0){
	    CHECK_LE(shape[i], INT_MAX / count_) << "blob size exceeds INT_MAX";
	}
	count_ *= shape[i];
	shape_[i] = shape[i];
	shape_data[i] = shape[i];
    }
    if(count_ > capacity_){
	capacity_ = count_;
	data_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
	diff_.reset(new SyncedMemory(capacity_ * sizeof(Dtype)));
    }
}


template <typename Dtype>
void Blob<Dtype>::reshape(const BlobShape& shape){
    CHECK_LE(shape.dim_size(), k_max_blob_axes);
    vector<int> shape_vec(shape.dim_size());
    for(int i =0; i < shape.dim_size(); i++){
	shape_vec[i] = shape.dim(i);
    }
    reshape(shape_vec);
}

template <typename Dtype>
void Blob<Dtype>::reshape_like(const Blob<Dtype>& other){
    reshape(other.shape());
}

template <typename Dtype>
Blob<Dtype>::Blob(const vector<int>& shape) : 
    capacity_(0){
	reshape(shape);
    }

template <typename Dtype>
const int* Blob<Dtype>::gpu_shape() const{
    CHECK(shape_data_);
    return (const int*)shape_data_->gpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_data() const{
    CHECK(data_);
    return (const Dtype*)data_->cpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_cpu_data(Dtype* data){
    CHECK(data);
    size_t size = count_ * sizeof(Dtype);
    if(data_->size() != size){
	data_.reset(new SyncedMemory(size));
	diff_.reset(new SyncedMemory(size));
    }
    data_->set_cpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_data() const{
    CHECK(data_);
    return (const Dtype*)data_->gpu_data();
}

template <typename Dtype>
void Blob<Dtype>::set_gpu_data(Dtype* data){
    CHECK(data);
    size_t size = count_ * sizeof(Dtype);
    if(data_->size() != size){
	data_.reset(new SyncedMemory(size));
	diff_.reset(new SyncedMemory(size));
    }
    data_->set_gpu_data(data);
}

template <typename Dtype>
const Dtype* Blob<Dtype>::cpu_diff() const{
    CHECK(diff_);
    return (const Dtype*)diff_->cpu_data();
}

template <typename Dtype>
const Dtype* Blob<Dtype>::gpu_diff() const{
    CHECK(diff_);
    return (const Dtype*)diff_->gpu_data();
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_data(){
    CHECK(data_);
    return static_cast<Dtype*>(data_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_cpu_diff(){
    CHECK(diff_);
    return static_cast<Dtype*>(diff_->mutable_cpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_data(){
    CHECK(data_);
    return static_cast<Dtype*>(data_->mutable_gpu_data());
}

template <typename Dtype>
Dtype* Blob<Dtype>::mutable_gpu_diff(){
    CHECK(diff_);
    return static_cast<Dtype*>(diff_->mutable_gpu_data());
}

template <typename Dtype>
void Blob<Dtype>::share_data(const Blob<Dtype>& other){
    CHECK_EQ(count_, other.count());
    data_ = other.data();
}

template <typename Dtype>
void Blob<Dtype>::share_diff(const Blob<Dtype>& other){
    CHECK_EQ(count_, other.count());
    diff_ = other.diff();
}

template <typename Dtype>
bool Blob<Dtype>::shape_equals(const BlobProto& other){
    if(other.has_num() || other.has_channels() || 
	    other.has_height() || other.has_channels()){
	return shape_.size() <= 4 && 
	    shape(0) == other.num() && 
	    shape(1) == other.channels() && 
	    shape(2) == other.height() && 
	    shape(3) == other.width();
    }
    vector<int> other_shape(other.shape().dim_size());
    for(int i = 0; i < other.shape().dim_size(); i++){
	other_shape[i] = other.shape().dim(i);
    }
    return shape_ == other_shape;
}



template <typename Dtype>
void Blob<Dtype>::from_proto(const BlobProto& proto, bool is_reshape){
    if(is_reshape){
	vector<int> shape;
	if(proto.has_num() || proto.has_channels() || 
		proto.has_height() || proto.has_width()){
	    shape.resize(4);
	    shape[0] = proto.num();
	    shape[1] = proto.channels();
	    shape[2] = proto.height();
	    shape[3] = proto.width();
	}
	else{
	    shape.resize(proto.shape().dim_size());
	    for(int i = 0; i < proto.shape().dim_size(); i++){
		shape[i] = proto.shape().dim(i);
	    }
	}
	reshape(shape);
    }
    else{
	CHECK(shape_equals(proto)) << "shape mismatch";
    }
    // 拷贝数据
    Dtype* data_vec = mutable_cpu_data();
    if(proto.double_data_size() > 0){
	CHECK_EQ(count_, proto.double_data_size());
	for(int i = 0; i < count_; i++){
	    data_vec[i] = proto.double_data(i);
	}
    }
    else{
	CHECK_EQ(count_, proto.data_size());
	for(int i = 0; i < count_; i++){
	    data_vec[i] = proto.data(i);
	}
    }
    
    Dtype* diff_vec = mutable_cpu_diff();
    if(proto.double_diff_size() > 0){
	CHECK_EQ(count_, proto.double_diff_size());
	for(int i = 0; i < count_; i++){
	    diff_vec[i] = proto.double_diff(i);
	}
    }
    else{
	CHECK_EQ(count_, proto.diff_size());
	for(int i = 0; i < count_; i++){
	    diff_vec[i] = proto.diff(i);
	}
    }

}










}
