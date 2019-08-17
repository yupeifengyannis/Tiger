#ifndef TIGER_LAYERS_CONV_BASE_CONV_LAYER_HPP
#define TIGER_LAYERS_CONV_BASE_CONV_LAYER_HPP

#include "tiger/layer.hpp"
#include "tiger/tiger.pb.h"

namespace tiger{

template <typename Dtype>
class BaseConvolutionLayer : public Layer<Dtype>{
public:
    explicit BaseConvolutionLayer(const LayerParameter& param) : 
	Layer<Dtype>(param){}
    virtual void layer_setup(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);

    virtual void reshape(const vector<Blob<Dtype>* >& bottom,
	    const vector<Blob<Dtype>* >& top);
protected:
    void forward_cpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output);
    void forward_cpu_bias(Dtype* output, const Dtype* bias);
    void backward_cpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights);
    void backward_cpu_bias(Dtype* bias, const Dtype* input);
#ifndef CPU_ONLY
    void forward_gpu_gemm(const Dtype* input, const Dtype* weights, Dtype* output);
    void forward_gpu_bias(Dtype* output, const Dtype* bias);
    void backward_gpu_gemm(const Dtype* input, const Dtype* output, Dtype* weights);
    void backward_gpu_bias(Dtype* bias, const Dtype* input);
#endif
    
    virtual bool reverse_dimensions() = 0;
    virtual void compute_output_shape() = 0;

protected:
    Blob<int> kernel_shape_;
    Blob<int> stride_;
    Blob<int> pad_;
    Blob<int> dilation_;
    Blob<int> conv_input_shape_;
    vector<int> col_buffer_shape_;
    vector<int> output_shape_;
    const vector<int>* bottom_shape_;

    int num_spatial_axes_;
    int bottom_dim_;
    int top_dim_;
    int channel_axis_;
    int num_;
    int channels_;
    int group_;
    int out_spatial_dim_;
    int weight_offset_;
    int num_output_;
    int bias_term_;
    int is_1x1_;

private:
    int num_kernels_im2col_;
    int num_kernels_col2im_;
    int conv_out_channels_;
    int conv_in_channels_;
    int conv_out_spatial_dim_;
    int kernel_dim_;
    int col_offset_;
    int output_offset_;
    
    Blob<Dtype> col_buffer_;
    Blob<Dtype> bias_multiplier_;


};


}


#endif
