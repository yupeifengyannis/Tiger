#include "tiger/layers/neuron_layer.hpp"

namespace tiger{

template <typename Dtype>
void NeuronLayer<Dtype>::reshape(const vector<Blob<Dtype>* >& bottom, 
	const vector<Blob<Dtype>* >& top){
    top[0]->reshape_like(*bottom[0]);
}

template class NeuronLayer<float>;
template class NeuronLayer<double>;



}
