#include <map>
#include <set>
#include <fstream>
#include <cmath>
#include <boost/filesystem.hpp>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>

#include "nlohmann/json.hpp"
#include "tiger.pb.h"
#include "utils/leveldb.hpp"
#include "utils/io.hpp"


using namespace std;
using namespace tiger;
using json = nlohmann::json;
namespace fs = boost::filesystem;

DEFINE_string(image_path, "", "bdd100k image path");
DEFINE_string(label_path, "", "bdd100k label path");
DEFINE_string(db_path, "", "bdd100k db path");

int get_id(std::string& name){
    if("bike" == name){
	return 0;
    }
    else if("bus" == name){
	return 1;
    }
    else if("car" == name){
	return 2;
    }
    else if("motor" == name){
	return 3;
    }
    else if("person" == name){
	return 4;
    }
    else if("rider" == name){
	return 5;
    }
    else if("traffic light" == name){
	return 6;
    }
    else if("traffic sign" == name){
	return 7;
    }
    else if("train" == name){
	return 8;
    }
    else if("truck" == name){
	return 9;
    }
    else{
	LOG(FATAL) << name << "is not existed";
    }
}


string get_file_name(const std::string file_path){
    int start_pos = 0;
    for(auto iter = file_path.rbegin(); iter != file_path.rend(); iter++){
	if('/' == *iter){
	    start_pos = file_path.size() - (iter - file_path.rbegin());
	    break;
	}
    }
    return file_path.substr(start_pos, file_path.size());
}

std::map<string, string> get_image_file_path(const std::string& image_path){
    CHECK(fs::exists(image_path)) << image_path << " is not existed!"; 
    std::map<string, string> image_file_table;
    fs::path image_dir(image_path);
    fs::directory_iterator end;
    string key;
    string value;
    for(fs::directory_iterator iter(image_dir); iter != end; iter++){
	if(fs::is_regular(*iter)){
	    value = iter->path().string();
	    key = get_file_name(value);
	    image_file_table.insert(make_pair(key, value));
	}
    }
    return image_file_table;
}

void show_image(std::map<string, string>image_file_table){
    cv::Mat img_mat;
    for(auto item : image_file_table){
	img_mat = cv::imread(item.second);
	cv::imshow("image", img_mat);
	cv::waitKey(1);
    }
}

void convert_dataset(const std::string& image_path, const std::string& label_path,
	const std::string& db_path){
    std::map<string, string> image_file_table = get_image_file_path(image_path);
    std::shared_ptr<LevelDB> db(new LevelDB());
    db->open(db_path, Mode::NEW);
    std::shared_ptr<LevelDBTransaction> txn(db->new_transaction());
    std::ifstream raw_label(label_path);
    json json_label = json::parse(raw_label);
    Datum datum;
    int count = 0;
    for(unsigned int i = 0; i < 200; i++){
	datum.Clear();
	json json_img = json_label[i];
	json json_object = json_img["labels"];
	std::string jpg_name = json_img["name"];
	cv::Mat img_mat = cv::imread(image_file_table[jpg_name]);
	// datum.set_channels(img_mat.channels());
	// datum.set_height(img_mat.rows);
	// datum.set_width(img_mat.cols);
	// datum.set_data(img_mat.data, img_mat.cols * img_mat.rows * img_mat.channels());
	mat_to_datum(img_mat, &datum);
	for(auto item : json_object){
	    if(item.find("box2d") != item.end()){
		string label_name = item["category"];
		json box2d = item["box2d"];
		float x1 = box2d["x1"];
		float x2 = box2d["x2"];
		float y1 = box2d["y1"];
		float y2 = box2d["y2"];
		float height = std::abs(y2 - y1);
		float width = std::abs(x1 - x2);
		float center_x = (x1 + x2) / 2;
		float center_y = (y1 + y2) / 2;
		ObjectLabel* object_label = datum.add_object_label();
		object_label->set_x1(x1);
		object_label->set_x2(x2);
		object_label->set_y1(y1);
		object_label->set_y2(y2);
		object_label->set_heigth(height);
		object_label->set_width(width);
		object_label->set_center_x(center_x);
		object_label->set_center_y(center_y);
		object_label->set_name(label_name);
		object_label->set_id(get_id(label_name));
	    }
	}
	string value;
	string key = jpg_name;
	datum.SerializeToString(&value);
	txn->put(key, value);
	if(++count % 10 == 0){
	    txn->commit();
	}
    }
    if(count % 10 != 0){
	txn->commit();
    }
    db->close();
}

int main(int argc, char** argv){
    ::gflags::ParseCommandLineFlags(&argc, &argv, true);
    convert_dataset(FLAGS_image_path, FLAGS_label_path, FLAGS_db_path); 
}



