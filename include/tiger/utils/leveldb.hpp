#ifndef UTILS_LEVELDB_HPP
#define UTILS_LEVELDB_HPP


#include <memory>
#include <string>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include "common.hpp"

namespace tiger{

enum Mode{
    READ,
    WRITE,
    NEW
};


class LevelDBCursor{
public:
    explicit LevelDBCursor(leveldb::Iterator* iter) : 
	iter_(iter){
	    seek_to_first();
	    CHECK(iter_->status().ok()) << iter_->status().ToString();
	}

    ~LevelDBCursor(){
	delete iter_;
    }

    void seek_to_first(){
	iter_->SeekToFirst();
    }

    void next(){
	iter_->Next();
    }

    string key(){
	return iter_->key().ToString();
    }

    string value(){
	return iter_->value().ToString();
    }

    bool valid(){
	return iter_->Valid();
    }
private:
    leveldb::Iterator* iter_;
};

class LevelDBTransaction{
public:
    explicit LevelDBTransaction(leveldb::DB* db) : 
	db_(db){
	    CHECK_NOTNULL(db_);
	}
    void put(const string& key, const string& value){
	batch_.Put(key, value);
    }
    
    void commit(){
	leveldb::Status status = db_->Write(leveldb::WriteOptions(), &batch_);
	CHECK(status.ok()) << "Failed to write batch to leveldb " << status.ToString();
    }

private:
    leveldb::DB* db_;
    leveldb::WriteBatch batch_;

};

class LevelDB{
public:
    LevelDB() : 
	db_(NULL){}
    void open(const string& source, Mode mode);
    void close(){
	if(db_ != NULL){
	    delete db_;
	    db_ = NULL;
	}
    }
    
    LevelDBCursor* new_cursor(){
	return new LevelDBCursor(db_->NewIterator(leveldb::ReadOptions()));
    }
    
    LevelDBTransaction* new_transaction(){
	return new LevelDBTransaction(db_);
    }


private:
    leveldb::DB* db_;
};

}



#endif
