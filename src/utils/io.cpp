#include <sys/stat.h>
#include <fcntl.h>
#include <memory>
#include <fstream>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>


#include "utils/io.hpp"

const int k_proto_read_bytes_limit = INT_MAX;

namespace tiger{

using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::ZeroCopyOutputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::CodedOutputStream;


bool read_proto_from_text_file(const char* filename, Message* proto){
    int fd = open(filename, O_RDONLY);
    CHECK_NE(fd, -1) << "file not found " << filename;
    FileInputStream* input = new FileInputStream(fd);
    bool success = google::protobuf::TextFormat::Parse(input, proto);
    close(fd);
    delete input;
    return success;
}

void write_proto_to_text_file(const Message& proto, const char* filename){
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    FileOutputStream* output = new FileOutputStream(fd);
    CHECK(google::protobuf::TextFormat::Print(proto, output));
    delete output;
    close(fd);
}

bool read_proto_from_binary_file(const char* filename, Message* proto){
    int fd = open(filename, O_RDONLY);
    CHECK_NE(fd, -1) << "file not found " << filename;
    ZeroCopyInputStream* raw_input = new FileInputStream(fd);
    CodedInputStream* coded_input = new CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(k_proto_read_bytes_limit, 536870912);
    bool success = proto->ParseFromCodedStream(coded_input);
    close(fd);
    delete raw_input;
    delete coded_input;
    return success;
}

void write_proto_to_binary_file(const Message& proto, const char* filename){
    fstream output(filename, ios::out | ios::trunc | ios::binary);
    CHECK(proto.SerializeToOstream(&output));
}


}
