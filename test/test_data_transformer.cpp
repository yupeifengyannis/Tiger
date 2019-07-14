#include <vector>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <gflags/gflags.h>
#include "data_transformer.hpp"
#include "utils/io.hpp"
#include "tiger.pb.h"

using namespace tiger;

DEFINE_string(path, "", "transformation prototxt path");

void test_transform(const TransformationParameter& param){
    DataTransformer<float> data_transformer(param, tiger::TRAIN);
    cv::VideoCapture cap(0);
    cv::Mat tmp_mat;
    cap >> tmp_mat;
    std::vector<int> blob_shape = data_transformer.infer_blob_shape(tmp_mat);
    LOG(INFO) << blob_shape.size();
    Blob<float> transformed_blob(blob_shape); 	
    int index = 0;
    while(1){
	cv::Mat frame;
	cap >> frame;
	cv::imshow("origin video", frame);
	data_transformer.transform(frame, &transformed_blob);
	std::vector<cv::Mat> ret_vec;
	transformed_blob.to_mat_vec(ret_vec);
	cv::imshow("transformed video", ret_vec[0]);
	cv::waitKey(1);
    }
}

int main(int argc, char** argv){
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    TransformationParameter param;
    tiger::read_proto_from_text_file_or_die(FLAGS_path, &param);
    LOG(INFO) << param.DebugString();
    test_transform(param);

}
