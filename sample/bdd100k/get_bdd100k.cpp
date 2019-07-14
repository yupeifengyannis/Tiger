#include <memory>
#include <string>

#include "utils/io.hpp"
#include "utils/leveldb.hpp"
#include "data_transformer.hpp"

using namespace std;
using namespace tiger;

int main(){
    string source = "/home/yupefieng/Documents/dl_framework/Tiger/test_data/bdd100k";
    std::shared_ptr<LevelDB> db(new LevelDB());
    db->open(source, Mode::READ);
    std::shared_ptr<LevelDBCursor> cursor(db->new_cursor());
    Datum datum;
    datum.ParseFromString(cursor->value());
    cv::Mat img_mat = tiger::transform_datum_to_mat(datum);
    cv::imshow("img", img_mat);
    cv::waitKey();

}





