#include "tiger/utils/leveldb.hpp"

namespace tiger{

void LevelDB::open(const string& source, Mode mode){
    leveldb::Options options;
    options.block_size = 65536;
    options.write_buffer_size = 26843546;
    options.max_open_files = 100;
    options.error_if_exists = mode == NEW;
    options.create_if_missing = mode != READ;
    leveldb::Status status = leveldb::DB::Open(options, source, &db_);
    CHECK(status.ok()) << "Failed to open leveldb" << source << 
	status.ToString();
    LOG(INFO) << "Opened leveldb" << source;
}
    


}
