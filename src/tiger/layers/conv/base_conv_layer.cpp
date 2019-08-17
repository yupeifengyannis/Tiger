#include "tiger/layers/conv/base_conv_layer.hpp"
#include "tiger/filler.hpp"

namespace tiger{

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::layer_setup(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    // 确定本层的相关参数
    ConvolutionParameter conv_param = this->layer_param_.conv_param();
    this->channel_axis_ = conv_param.axis();
    const int first_spatial_axis = channel_axis_ + 1;
    const int num_axes = bottom[0]->num_axes();
    this->num_spatial_axes_ = num_axes - first_spatial_axis;
    CHECK_GE(num_spatial_axes_, 0);
    vector<int> spatial_dim_blob_shape(1, num_spatial_axes_);
    // 确定kernel的参数
    this->kernel_shape_.reshape(spatial_dim_blob_shape);
    CHECK(conv_param.has_kernel_h() && conv_param.kernel_w()) << 
	"convolution parameter must have kernel_h and kernel_w";
    CHECK_EQ(num_spatial_axes_, 2) << 
	"num_spatial_axes must be 2";	
    int* kernel_data = kernel_shape_.mutable_cpu_data();
    kernel_data[0] = conv_param.kernel_h();
    kernel_data[1] = conv_param.kernel_w();
    // 确定stride的参数
    this->stride_.reshape(spatial_dim_blob_shape);
    CHECK(conv_param.has_stride_h() && conv_param.has_stride_w()) << 
	"convolution parameter must have stride_h and stride_w";
    int* stride_data = stride_.mutable_cpu_data();
    stride_data[0] = conv_param.stride_h();
    stride_data[1] = conv_param.stride_w();
    // 确定pad的参数
    this->pad_.reshape(spatial_dim_blob_shape);
    CHECK(conv_param.has_pad_h() && conv_param.has_pad_w()) << 
	"convolution parameter must have pad_h and pad_w";
    int* pad_data = pad_.mutable_cpu_data();
    pad_data[0] = conv_param.pad_h();
    pad_data[1] = conv_param.pad_w();
    // 确定dilation的参数
    this->dilation_.reshape(spatial_dim_blob_shape);
    CHECK(conv_param.has_dilation_h() && conv_param.has_dilation_w()) <<
	"convolution parameter must have dilation_h and dilation_w";
    int* dilation_data = dilation_.mutable_cpu_data();
    dilation_data[0] = conv_param.dilation_h();
    dilation_data[1] = conv_param.dilation_w();
    // 确定1x1的参数
    this->is_1x1_ = false;
    if(kernel_data[0] == 1 && kernel_data[1] == 1 && 
	    pad_data[0] == 1 && pad_data[1] == 1){
	is_1x1_ = true;
    }
    this->channels_ = bottom[0]->shape(channel_axis_);
    this->num_output_ = conv_param.num_output();
    CHECK_GT(num_output_, 0);
    this->group_ = conv_param.group();
    CHECK_EQ(channels_ % group_, 0);
    CHECK_EQ(num_output_ % group_, 0);
    
    if(reverse_dimensions()){
	this->conv_out_channels_ = channels_;
	this->conv_in_channels_ = num_output_;
    }
    else{
	this->conv_out_channels_ = num_output_;
	this->conv_in_channels_ = channels_;
    }
    
    vector<int> weight_shape(2);
    weight_shape[0] = conv_out_channels_;
    weight_shape[1] = conv_in_channels_ / group_;
    for(int i = 0; i < num_spatial_axes_; ++i){
	weight_shape.push_back(kernel_data[i]);
    }
    this->bias_term_ = conv_param.bias_term();
    vector<int> bias_shape(bias_term_, num_output_);
    if(this->blobs_.size() > 0){
	LOG(INFO) << "skipping parameter initialization";
    }
    else{
	if(bias_term_){
	    this->blobs_.resize(2);
	}
	else{
	    this->blobs_.resize(1);
	}
	this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
	std::shared_ptr<Filler<Dtype> > weight_filler;
	weight_filler.reset(get_filler<Dtype>(conv_param.weight_filler()));
	weight_filler->fill_data(this->blobs_[0].get());
	if(bias_term_){
	    this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
	    std::shared_ptr<Filler<Dtype> > bias_filler;
	    bias_filler.reset(get_filler<Dtype>(conv_param.bias_filler()));
	    bias_filler->fill_data(this->blobs_[1].get());
	}
    }
    
    this->kernel_dim_ = this->blobs_[0]->count(1);
    this->weight_offset_ = conv_out_channels_ * kernel_dim_ / group_;
    this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void BaseConvolutionLayer<Dtype>::reshape(const vector<Blob<Dtype>* >& bottom,
	const vector<Blob<Dtype>* >& top){
    const int first_spatial_axis = channel_axis_ + 1;
    CHECK_EQ(bottom[0]->num_axes(), first_spatial_axis + num_spatial_axes_) << 
	"bottom num_axes may not change";
    num_ = bottom[0]->count(0, channel_axis_);
    CHECK_EQ(bottom[0]->shape(channel_axis_), channels_);
    
}




template class BaseConvolutionLayer<float>;
template class BaseConvolutionLayer<double>;





}
