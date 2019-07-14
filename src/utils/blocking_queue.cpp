#include <glog/logging.h>
#include "blob.hpp"
#include "utils/blocking_queue.hpp"

namespace tiger{

template <typename Dtype>
BlockingQueue<Dtype>::BlockingQueue() : 
    sync_(new Sync()){
    }

template <typename Dtype>
void BlockingQueue<Dtype>::push(const Dtype& t){
    std::unique_lock<std::mutex> lock(sync_->mutex_);   
    queue_.push(t);
    sync_->condition_.notify_one();
}

template <typename Dtype>
bool BlockingQueue<Dtype>::try_pop(Dtype* t){
    std::unique_lock<std::mutex> lock(sync_->mutex_);
    if(queue_.empty()){
	return false;
    }
    *t = queue_.front();
    queue_.pop();
    return true;
}

template <typename Dtype>
Dtype BlockingQueue<Dtype>::pop(const string& log_on_wait){
    std::unique_lock<std::mutex> lock(sync_->mutex_);
    while(queue_.empty()){
	if(!log_on_wait.empty()){
	    LOG_EVERY_N(INFO, 1000) << log_on_wait;
	}
	sync_->condition_.wait(lock);
    }
    Dtype t = queue_.front();
    queue_.pop();
    return t;
}

template <typename Dtype>
bool BlockingQueue<Dtype>::try_peek(Dtype* t){
    std::unique_lock<std::mutex> lock(sync_->mutex_);
    if(queue_.empty()){
	return false;
    }
    *t = queue_.front();
    return true;
}

template <typename Dtype>
Dtype BlockingQueue<Dtype>::peek(){
    std::unique_lock<std::mutex> lock(sync_->mutex_);
    while(queue_.empty()){
	sync_->condition_.wait(lock);
    }
    return queue_.front();
}

template <typename Dtype>
size_t BlockingQueue<Dtype>::size() const{
    std::unique_lock<std::mutex> lock(sync_->mutex_);
    return queue_.size();
}

template class BlockingQueue<Batch<float>*>;
template class BlockingQueue<Batch<double>*>;
template class BlockingQueue<int>;
}
