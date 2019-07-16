#ifndef TIGER_BLOB_HPP
#define TIGER_BLOB_HPP

#include <vector>
#include <memory>
#include <opencv2/opencv.hpp>
#include "tiger/common.hpp"
#include "tiger/syncedmem.hpp"
#include "tiger/tiger.pb.h"

const int k_max_blob_axes = 32;

namespace tiger{
template <typename Dtype>
class Blob{
public:
    Blob() : 
	data_(),
	diff_(),
	count_(0),
	capacity_(0){}
    explicit Blob(const vector<int>& shape);
    void reshape(const vector<int>& shape);
    void reshape(const BlobShape& shape);
    void reshape_like(const Blob& other);
    
    inline string shape_string()const{
	string ret;
	for(int i = 0; i < shape_.size(); i++){
	    ret += to_string(shape_[i]) + " ";
	}
	ret += "(" + to_string(count_) + ")";
	return ret;
    }
    
    inline const vector<int>& shape() const{
	return shape_;
    }
    
    inline int shape(int index) const{
	return shape_[index];
    }
    inline int num_axes() const{
	return shape_.size();
    }
    inline int count() const{
	return count_;
    }
    
    inline int count(int start_axis, int end_axis) const{
	CHECK_LE(start_axis, end_axis);
	CHECK_GE(start_axis, 0);
	CHECK_GE(end_axis, 0);
	CHECK_LE(end_axis, num_axes());
	int count = 1;
	for(int i = start_axis; i < end_axis; i++){
	    count *= shape(i);
	}
	return count;
    }
    
    inline int count(int start_axis){
	return count(start_axis, num_axes());
    }

    inline int num() const{
	return shape(0);
    }
    inline int channels() const{
	return shape(1);
    }
    inline int height() const{
	return shape(2);
    }
    inline int width() const{
	return shape(3);
    }
    
    inline int offset(const int n, const int c = 0, const int h = 0, const int w = 0) const{
	CHECK_GE(n, 0);
	CHECK_LE(n, num());
	CHECK_GE(c, 0);
	CHECK_LE(c, channels());
	CHECK_GE(h, 0);
	CHECK_LE(h, height());
	CHECK_GE(w, 0);
	CHECK_LE(w, width());
	return ((n * channels() + c) * height() + h) * width() + w;
    }
    
    inline int offset(const vector<int>& indices) const{
	CHECK_LE(indices.size(), num_axes());
	int offset = 0;
	for(int i = 0; i < num_axes(); i++){
	    offset *= shape(i);
	    if(indices.size() > 1){
		CHECK_GE(indices[i], 0);
		CHECK_LE(indices[i], shape(i));
		offset += indices[i];
	    }
	}
	return offset;
    }
    
    void copy_from(const Blob<Dtype>& src, bool copy_diff = false, bool reshape = false);

    inline Dtype data_at(const int n, const int c, const int h, const int w) const{
	return cpu_data()[offset(n, c, h, w)];
    }
    
    inline Dtype diff_at(const int n, const int c, const int h, const int w) const{
	return cpu_diff()[offset(n, c, h, w)];
    }
    
    inline Dtype data_at(const vector<int>& indices) const{
	return cpu_data()[offset(indices)];
    }

    inline Dtype diff_at(const vector<int>& indices) const{
	return cpu_diff()[offset(indices)];
    }

    inline const std::shared_ptr<SyncedMemory>& data() const{
	CHECK(data_);
	return data_;
    }

    inline const std::shared_ptr<SyncedMemory>& diff() const{
	CHECK(diff_);
	return diff_;
    }

    const Dtype* cpu_data()const;
    void set_cpu_data(Dtype* data);
    const int* gpu_shape() const;
    const Dtype* gpu_data() const;
    void set_gpu_data(Dtype* data);
    const Dtype* cpu_diff() const;
    const Dtype* gpu_diff() const;
    Dtype* mutable_cpu_data();
    Dtype* mutable_gpu_data();
    Dtype* mutable_cpu_diff();
    Dtype* mutable_gpu_diff();
    void update();
    void from_proto(const BlobProto& proto, bool reshape = true);
    void to_proto(BlobProto* proto, bool write_diff = false) const;
    void to_mat_vec(vector<cv::Mat>& output_mat);
    
    Dtype asum_data() const;
    Dtype asum_diff() const;
    Dtype sumsq_data() const;
    Dtype sumsq_diff() const;
    
    void scale_data(Dtype scale_factor);
    void scale_diff(Dtype scale_factor);

    void share_data(const Blob& other);

    void share_diff(const Blob& other);
    bool shape_equals(const BlobProto& other);

protected:
    std::shared_ptr<SyncedMemory> data_;///<存的是feature和label的数据
    std::shared_ptr<SyncedMemory> diff_;///<存的是误差
    std::shared_ptr<SyncedMemory> shape_data_;///<存的是形状数据
    std::vector<int> shape_;
    int count_;
    int capacity_;

};


}



#endif
