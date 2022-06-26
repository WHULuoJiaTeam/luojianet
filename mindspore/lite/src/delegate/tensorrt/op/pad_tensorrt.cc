/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <numeric>
#include <functional>
#include "src/delegate/tensorrt/op/pad_tensorrt.h"
#include "src/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
int PadTensorRT::IsSupport(const mindspore::schema::Primitive *primitive,
                           const std::vector<mindspore::MSTensor> &in_tensors,
                           const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE2 && in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  if (in_tensors_[1].Data() == nullptr) {
    MS_LOG(ERROR) << "invalid pad tensor for: " << op_name_;
    return RET_ERROR;
  }
  auto pad_primitive = this->GetPrimitive()->value_as_PadFusion();
  if (pad_primitive == nullptr) {
    MS_LOG(ERROR) << "convert PadFusion failed: " << op_name_;
    return RET_ERROR;
  }
  schema::PaddingMode padding_mode = pad_primitive->padding_mode();
  if (padding_mode != schema::PaddingMode::PaddingMode_CONSTANT) {
    MS_LOG(ERROR) << "Unsupported padding mode: " << schema::PaddingMode(padding_mode) << ", for op: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors[0].format() != Format::NHWC && in_tensors[0].format() != Format::NCHW) {
    MS_LOG(ERROR) << "Unsupported input tensor format of " << in_tensors[0].format();
    return RET_ERROR;
  }
  constant_value_ = pad_primitive->constant_value();
  return RET_OK;
}

int PadTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  mindspore::MSTensor &pad_tensor = in_tensors_[1];
  int element_cnt = std::accumulate(pad_tensor.Shape().begin(), pad_tensor.Shape().end(), 1, std::multiplies<int>());
  if (element_cnt != tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims * INPUT_SIZE2) {
    MS_LOG(ERROR) << "pad tensor cnt is invalid. cnt: " << element_cnt
                  << ", input tensor dims cnt: " << tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims;
    return RET_ERROR;
  }

  nvinfer1::ITensor *pad_input = tensorrt_in_tensors_[0].trt_tensor_;
  MS_LOG(DEBUG) << "before transpose "
                << GetTensorFormat(pad_input, tensorrt_in_tensors_[0].format_, tensorrt_in_tensors_[0].same_format_);
  if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
      tensorrt_in_tensors_[0].format_ == Format::NHWC) {
    // transpose: NHWC->NCHW
    nvinfer1::IShuffleLayer *transpose_layer_in = NHWC2NCHW(network, *tensorrt_in_tensors_[0].trt_tensor_);
    if (transpose_layer_in == nullptr) {
      MS_LOG(ERROR) << "transpose: NHWC->NCHW failed";
      return RET_ERROR;
    }
    transpose_layer_in->setName((op_name_ + "_transpose2NCHW").c_str());
    this->transpose_layer_ = transpose_layer_in;
    pad_input = transpose_layer_in->getOutput(0);
    MS_LOG(DEBUG) << "after transpose " << GetTensorFormat(pad_input, Format::NCHW, false);
  }

  // trt 6 only support 2D padding
  const int *padding_data = reinterpret_cast<const int *>(in_tensors_[1].Data().get());
  MS_ASSERT(padding_data);
  nvinfer1::IPaddingLayer *padding_layer = nullptr;
  if (element_cnt == index_NHWC_ * INPUT_SIZE2) {
    // only support pad at HW index
    int h_pre;
    int h_post;
    int w_pre;
    int w_post;
    if (SameDims(pad_input->getDimensions(), in_tensors_[0].Shape())) {
      // NCHW: 0: N_pre, 1: N_post, 2: C_pre, 3: C_post, 4: H_pre, 5: H_post, 6: W_pre, 7: W_post
      if (*padding_data != 0 || *(padding_data + 1) != 0 || *(padding_data + 2) != 0 || *(padding_data + 3) != 0) {
        MS_LOG(WARNING) << "tensorrt padding only support pad at HW index, unsupported padding value of: " << op_name_;
      }
      h_pre = 4;
      h_post = 5;
      w_pre = 6;
      w_post = 7;
    } else {
      // NHWC: 0: N_pre, 1: N_post, 2: H_pre, 3: H_post, 4: W_pre, 5: W_post, 6: C_pre, 7: C_post
      if (*padding_data != 0 || *(padding_data + 1) != 0 || *(padding_data + 6) != 0 || *(padding_data + 7) != 0) {
        MS_LOG(WARNING) << "tensorrt padding only support pad at HW index, unsupported padding value of: " << op_name_;
      }
      h_pre = 2;
      h_post = 3;
      w_pre = 4;
      w_post = 5;
    }
    nvinfer1::DimsHW prePadding{*(padding_data + h_pre), *(padding_data + w_pre)};
    nvinfer1::DimsHW postPadding{*(padding_data + h_post), *(padding_data + w_post)};
    MS_LOG(DEBUG) << op_name_ << " prePadding: " << prePadding.d[0] << ", " << prePadding.d[1]
                  << "; postPadding: " << postPadding.d[0] << ", " << postPadding.d[1];

    padding_layer = network->addPadding(*pad_input, prePadding, postPadding);
  } else {
    MS_LOG(ERROR) << "need check for pad_tensor dims: " << op_name_
                  << ", pad_tensor ElementNum: " << pad_tensor.ElementNum();
    return RET_ERROR;
  }
  if (padding_layer == nullptr) {
    MS_LOG(ERROR) << "add padding layer failed for " << op_name_;
    return RET_ERROR;
  }
  this->layer_ = padding_layer;
  padding_layer->setName(op_name_.c_str());
  padding_layer->getOutput(0)->setName((op_name_ + "_output").c_str());
  bool same_format = SameDims(padding_layer->getOutput(0)->getDimensions(), out_tensors_[0].Shape()) &&
                     SameDims(tensorrt_in_tensors_[0].trt_tensor_->getDimensions(), in_tensors_[0].Shape());
  this->AddInnerOutTensors(ITensorHelper{padding_layer->getOutput(0), Format::NCHW, same_format});
  MS_LOG(DEBUG) << "after transpose " << GetTensorFormat(tensorrt_out_tensors_[0]);
  return RET_OK;
}
}  // namespace mindspore::lite
