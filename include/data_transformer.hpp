#ifndef DATA_TRANSFORMER_HPP
#define DATA_TRANSFORMER_HPP

#include <vector>
#include <opencv2/opencv.hpp>
#include "tiger.pb.h"
#include "blob.hpp"

namespace tiger{

template <typename Dtype>
class DataTransformer{
public:
    explicit DataTransformer(const TransformationParameter& param, Phase phase);
    virtual ~DataTransformer(){}
     
    void init_rand();
    
    void transform(const vector<cv::Mat>& mat_vector, Blob<Dtype>* transformed_blob);
    
    void transform(const cv::Mat& cv_img, Blob<Dtype>* transform_blob);
    
    void transform(Blob<Dtype>* input_blob, Blob<Dtype>* transformed_blob);
    
    vector<int> infer_blob_shape(const vector<cv::Mat>& mat_vector);

    vector<int> infer_blob_shape(const cv::Mat& cv_img);
    
protected:
    virtual int rand(int n);
    TransformationParameter param_;
    Phase phase_;
    std::shared_ptr<Tiger::RNG> rng_;
    Blob<Dtype> data_mean_;
    std::vector<Dtype> mean_values_;

};

}

#endif
