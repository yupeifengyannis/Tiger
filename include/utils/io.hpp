#ifndef UTILS_IO_HPP
#define UTILS_IO_HPP

#include <google/protobuf/message.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "common.hpp"
#include "tiger.pb.h"


namespace tiger{

using ::google::protobuf::Message;

/// \brief 从文本文件中读取prtobuf
bool read_proto_from_text_file(const char* filename, Message* proto);

inline bool read_proto_from_text_file(const string& filename, Message* proto){
    return read_proto_from_text_file(filename.c_str(), proto);
}

inline void read_proto_from_text_file_or_die(const string& filename, Message* proto){
    CHECK(read_proto_from_text_file(filename, proto));
}

/// \brief 将protobuf写到文本文件中
void write_proto_to_text_file(const Message& proto, const char* filename);

inline void write_proto_to_text_file(const Message& proto, const string& filename){
    return write_proto_to_text_file(proto, filename.c_str());
}

/// \brief 从二进制文件中读取protobuf
bool read_proto_from_binary_file(const char* filename, Message* proto);

inline bool read_proto_from_binary_file(const string& filename, Message* proto){
    return read_proto_from_binary_file(filename.c_str(), proto);
}

inline void read_proto_from_binary_file_or_die(const string& filename, Message* proto){
    CHECK(read_proto_from_binary_file(filename, proto));
}

void write_proto_to_binary_file(const Message& proto, const char* filename);

inline void write_proto_to_binary_file(const Message& proto, const string filename){
    return write_proto_to_binary_file(proto, filename.c_str());
}

cv::Mat transform_datum_to_mat(const Datum& datum);

cv::Mat read_image_to_mat(const string& file_name, const int height = 0, 
    const int width = 0, const bool is_color = true);


cv::Mat decode_datum_to_mat(const Datum& datum);








}


#endif
