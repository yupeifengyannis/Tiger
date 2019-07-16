#ifndef TIGER_LAYERS_DATA_LAYER_HPP
#define TIGER_LAYERS_DATA_LAYER_HPP
#include <memory>
#include "tiger/layers/base_data_layer.hpp"
#include "tiger/utils/leveldb.hpp"

namespace tiger{
template <typename Dtype>
class DataLayer : public BasePrefetchingDataLayer<Dtype>{
public:
    explicit DataLayer(const LayerParameter& param);
    virtual ~DataLayer(){}
    virtual void data_layer_setup(const vector<Blob<Dtype>* >& bottom, 
	    const vector<Blob<Dtype>* >& top);
    virtual inline const char* type() const {
	return "Data";
    }
    virtual inline int exact_num_bottom_blobs() const{
	return 0;
    }
    virtual inline int min_top_blobs() const{
	return 1;
    }
    virtual inline int max_top_blobs() const{
	return 2;
    }
protected:
    void next();
    void skip();
    virtual void load_batch(Batch<Dtype>* batch);
    std::shared_ptr<LevelDB> db_;
    std::shared_ptr<LevelDBCursor> cursor_;
    uint64_t offset_;
};
}


#endif
