#include <memory>
#include <string>

#include "utils/io.hpp"
#include "utils/leveldb.hpp"
#include "data_transformer.hpp"
#include "tiger.pb.h"

using namespace std;
using namespace tiger;

int main(){
    string source = "/home/yupefieng/Documents/dl_framework/Tiger/test_data/bdd100k";
    std::shared_ptr<LevelDB> db(new LevelDB());
    db->open(source, Mode::READ);
    std::shared_ptr<LevelDBCursor> cursor(db->new_cursor());
    Datum datum;
    while(cursor->valid()){
	datum.ParseFromString(cursor->value());
	cv::Mat img_mat = tiger::transform_datum_to_mat(datum);
	int label_size = datum.object_label_size();
	for(int i = 0; i < label_size; i++){
	    ObjectLabel label = datum.object_label(i);
	    cv::Rect rect(label.x1(), label.y1(), label.width(), label.heigth());
	    cv::rectangle(img_mat, rect, cv::Scalar(0, 0, 255), 1, cv::LINE_8, 0); 
	    cv::putText(img_mat, label.name(), cv::Point(label.x2(), label.y2()),
		cv::FONT_HERSHEY_TRIPLEX, 1, cv::Scalar(0, 0, 255));
	}
	cv::imshow("img", img_mat);
	cv::waitKey();
	cursor->next();
    }

}





