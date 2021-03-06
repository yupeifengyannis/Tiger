#include <fstream>
#include <memory>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include "tiger/utils/leveldb.hpp"
#include "tiger/tiger.pb.h"

using namespace tiger;

DEFINE_string(backend, "leveldb", "backend for storing the results");
DEFINE_string(image_filename, "", "image file path");
DEFINE_string(label_filename, "", "label file path");
DEFINE_string(db_path, "", "leveldb dataset path");


// 进行大小端转换
uint32_t swap_endian(uint32_t val) {
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}


void conver_dataset(const char* image_filename, const char* label_filename,
	const char* db_path, const string& db_backend){
    std::ifstream image_file(image_filename, std::ios::in | std::ios::binary);
    std::ifstream label_file(label_filename, std::ios::in | std::ios::binary);
    CHECK(image_file) << "unable to open file " << image_filename;
    CHECK(label_file) << "unable to open file " << label_filename;
    
    uint32_t magic;
    uint32_t num_items;
    uint32_t num_labels;
    uint32_t rows;
    uint32_t cols;

    image_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    LOG(INFO) << "image file magic is " << magic;
    CHECK_EQ(magic, 2051) << "incorrect image file magic";
    label_file.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    LOG(INFO) << "label file magic is " << magic;
    CHECK_EQ(magic, 2049) << "incorrect label file magic";
    image_file.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    LOG(INFO) << "image item number is " << num_items;
    label_file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = swap_endian(num_labels);
    LOG(INFO) << "label item number is " << num_labels;
    CHECK_EQ(num_labels, num_items);
    image_file.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    LOG(INFO) << "rows is " << rows;
    image_file.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);
    LOG(INFO) << "cols is " << cols;
    std::shared_ptr<tiger::LevelDB> db(new tiger::LevelDB);
    db->open(db_path, tiger::NEW);
    std::shared_ptr<tiger::LevelDBTransaction> txn(db->new_transaction());
    
    char label;
    char* pixels = new char[rows * cols];
    int count = 0;
    string value;

    Datum datum;
    datum.set_channels(1);
    datum.set_height(rows);
    datum.set_width(cols);

    LOG(INFO) << "a total of " << num_items << "items";
    LOG(INFO) << "rows: " << rows << "cols: " << cols;
    for(int item_id = 0; item_id < num_items; item_id++){
	image_file.read(pixels, rows * cols);
	label_file.read(&label, 1);
	datum.set_data(pixels, rows * cols);
	datum.set_label(label);
	string key_str = to_string(item_id);
	datum.SerializeToString(&value);
	
	txn->put(key_str, value);
	
	if(++count % 1000 == 0){
	    txn->commit();
	}
    }
    if(count % 1000 != 0){
	txn->commit();
    }
    delete [] pixels;
    db->close();

}

int main(int argc, char** argv){
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    const string& db_backend = FLAGS_backend;
    const string& image_filename = FLAGS_image_filename;
    const string& label_filename = FLAGS_label_filename;
    const string& db_path = FLAGS_db_path;
    conver_dataset(image_filename.c_str(), label_filename.c_str(), 
	    db_path.c_str(), db_backend);
}
