#include <glog/logging.h>
#include <thread>
#include "tiger/utils/blocking_queue.hpp"

using namespace tiger;
BlockingQueue<int> test_queue;

void producter_thread(int item){
    test_queue.push(item);
}

void consumer_thread(){
    int* ret = new int;
    while(1){
	if(test_queue.try_pop(ret)){
	    LOG(INFO) << *ret;
	}
    }
}

int main(){
    std::vector<std::thread> thread_vec;
    for(int i = 0; i < 10; i++){
	std::thread product(producter_thread, i);
	thread_vec.emplace_back(std::move(product));
    }
    std::thread consumer(consumer_thread);
    for(auto&& item : thread_vec){
	item.join();
    }
    consumer.join();
}




