#ifndef TIGER_UTILS_BLOCKING_QUEUE_HPP
#define TIGER_UTILS_BLOCKING_QUEUE_HPP
#include <queue>
#include <mutex>
#include <string>
#include <condition_variable>
#include "tiger/blob.hpp"

using namespace std;
namespace tiger{

template <typename Dtype>
class Batch{
public:
    Blob<Dtype> data_;
    Blob<Dtype> label_;
};

class Sync{
public:
    mutable std::mutex mutex_;     
    std::condition_variable condition_;
};

template <typename Dtype>
class BlockingQueue{
public:
    explicit BlockingQueue();
    BlockingQueue(const BlockingQueue&) = delete;
    BlockingQueue& operator=(const BlockingQueue&) = delete;
    void push (const Dtype& t);
    bool try_pop(Dtype* t);
    Dtype pop(const string& log_on_wait = "");
    bool try_peek(Dtype* t);
    Dtype peek();
    size_t size()const;
protected:
    std::queue<Dtype> queue_; 
    std::shared_ptr<Sync> sync_;
};

}

#endif
