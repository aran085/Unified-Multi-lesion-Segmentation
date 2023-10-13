#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/multi_channel_bin_sigmoid_ce_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
void MultiChannelBinSigmoidCrossEntropyLossLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::LayerSetUp(bottom, top);
  sigmoid_bottom_vec_.clear();
  sigmoid_bottom_vec_.push_back(bottom[0]);
  sigmoid_top_vec_.clear();
  sigmoid_top_vec_.push_back(sigmoid_output_.get());
  sigmoid_layer_->SetUp(sigmoid_bottom_vec_, sigmoid_top_vec_);

  const MCBSCELossParameter  mcbsce_loss_param = this->layer_param_.mcbsce_loss_param();
  num_label_ = mcbsce_loss_param.num_label();
	key_ = mcbsce_loss_param.key();
}

template <typename Dtype>
void MultiChannelBinSigmoidCrossEntropyLossLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  outer_num_ = bottom[0]->shape(0);  // batch size
  inner_num_ = bottom[0]->count(1);  // instance size: |output| == |target|
  CHECK_EQ(bottom[0]->width()*bottom[0]->height(), bottom[1]->width()*bottom[1]->height()) <<
    "MULTICHANNEL_BIN_SIGMOID_CROSS_ENTROPY_LOSS layer inputs must have the same spatial dimension.";
  sigmoid_layer_->Reshape(sigmoid_bottom_vec_, sigmoid_top_vec_);
}

template <typename Dtype>
void MultiChannelBinSigmoidCrossEntropyLossLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  // The forward pass computes the sigmoid outputs.
  sigmoid_bottom_vec_[0] = bottom[0];
  sigmoid_layer_->Forward(sigmoid_bottom_vec_, sigmoid_top_vec_);
  // Compute the loss (negative log likelihood)
  // Stable version of loss computation from input data
  const Dtype* input_data = bottom[0]->cpu_data();
  const Dtype* target = bottom[1]->cpu_data();
  
  //lt
  const Dtype* sigmoid_output_data = sigmoid_output_->cpu_data();
  Dtype *temp_count_pos = new Dtype[num_label_];
  Dtype *temp_neg_count = new Dtype[num_label_];
  Dtype (*temp_count_neg)[5] = new Dtype[num_label_][5]; 
  Dtype (*temp_neg_loss)[5] = new Dtype[num_label_][5];
  Dtype *temp_pos_loss = new Dtype[num_label_];
  Dtype *temp_loss_neg = new Dtype[num_label_];
  Dtype loss_pos = 0;
  Dtype loss_neg = 0;
  Dtype count_pos = 0;
  Dtype count_neg = 0;
  
    //chu shi hua
  for(int i = 0; i < num_label_; i++){
	  temp_count_pos[i] = 0;
	  temp_neg_count[i] = 0;
	  temp_pos_loss[i] = 0;
	  temp_loss_neg[i] = 0;
	  for(int j = 0; j< 5;j++){
		  temp_count_neg[i][j] = 0;
		  temp_neg_loss[i][j] = 0;
	  }
  }
  int dim = bottom[0]->height()*bottom[0]->width();
    //jin xing tong ji
    for (int i = 0; i < num_label_; ++i) { /* loop over channels */
      for (int j = 0; j < dim; ++j) { /* loop over pixels */
	      int idx = i*dim+j;
		  Dtype temp = log(1 + exp(input_data[idx] - 2 * input_data[idx] * (input_data[idx] >= 0)));
      	if (target[j] == (i+1)) {
			temp_count_pos[i] = temp_count_pos[i] + key_;
			temp_pos_loss[i] -=input_data[idx] * (1 - (input_data