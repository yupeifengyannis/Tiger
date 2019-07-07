#include "blob.hpp"


namespace tiger{

template <typename Dtype>
void Blob<Dtype>::reshape(const vector<int>& shape){
    CHECK_LE(shape.size(), k_max_blob_axes);
    count_ = 1;
    shape_.resize(shape.size());

}

}
