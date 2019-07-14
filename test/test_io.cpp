#include <boost/filesystem.hpp>
#include <opencv2/opencv.hpp>
#include <glog/logging.h>
#include "tiger.pb.h"
#include "common.hpp"
#include "utils/io.hpp"

using namespace tiger;
namespace fs = boost::filesystem;
void test_write_proto_to_text_file(const string& filename){
    tiger::SolverParameter solver_param;
    solver_param.set_net("net");
    solver_param.set_train_net("train_net");
    solver_param.set_test_net("test_net");
    LOG(INFO) << solver_param.DebugString();
    tiger::write_proto_to_text_file(solver_param, filename); 
}

void test_read_proto_from_text_file(const string& filename){
    tiger::SolverParameter solver_param;
    tiger::read_proto_from_text_file_or_die(filename, &solver_param);
    LOG(INFO) << solver_param.DebugString();

}

void test_write_proto_to_binary_file(const string& filename, 
	const string& binary_name){
    tiger::SolverParameter solver_param;
    tiger::read_proto_from_text_file_or_die(filename, &solver_param);
    tiger::write_proto_to_binary_file(solver_param, binary_name);
}

void test_read_proto_from_binary_file(const string& binary_name){
    tiger::SolverParameter solver_param;
    tiger::read_proto_from_binary_file_or_die(binary_name, &solver_param);
    LOG(INFO) << solver_param.DebugString();
}

void test_read_image_to_mat(const string file_name){
    cv::Mat img = read_image_to_mat(file_name);
    cv::imshow("test img", img);
    cv::waitKey();
}



int main(){ 
    const string file_name = "./test_data/horse.jpg";
    test_read_image_to_mat(file_name);
}






