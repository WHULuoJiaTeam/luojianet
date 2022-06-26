/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/delegate/npu/op/max_pooling_npu.h"
#include "src/delegate/npu/npu_converter_utils.h"
namespace mindspore {
int MaxPoolingNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors) {
  auto pooling_prim = primitive->value_as_MaxPoolFusion();
  if (pooling_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  return RET_OK;
}

int MaxPoolingNPUOp::SetPoolingParam(const schema::MaxPoolFusion *pooling_prim) {
  pooling_->set_attr_mode(0);
  if (pooling_prim->global()) {
    pooling_->set_attr_global_pooling(pooling_prim->global());
  } else {
    CHECK_NULL_RETURN(pooling_prim->kernel_size());
    auto window_h = static_cast<int>(*(pooling_prim->kernel_size()->begin()));
    auto window_w = static_cast<int>(*(pooling_prim->kernel_size()->begin() + 1));
    pooling_->set_attr_window(ge::AttrValue::LIST_INT({window_h, window_w}));
  }
  CHECK_NULL_RETURN(pooling_prim->strides());
  auto stride_h = static_cast<int>(*(pooling_prim->strides()->begin()));
  auto stride_w = static_cast<int>(*(pooling_prim->strides()->begin() + 1));
  pooling_->set_attr_stride(ge::AttrValue::LIST_INT({stride_h, stride_w}));
  if (pooling_prim->pad_mode() == schema::PadMode_SAME) {
    pooling_->set_attr_pad_mode(PAD_SAME);
    pooling_->set_attr_pad({0, 0, 0, 0});
  } else if (pooling_prim->pad_mode() == schema::PadMode_VALID) {
    pooling_->set_attr_pad_mode(PAD_VALID);
    pooling_->set_attr_pad({0, 0, 0, 0});
  } else {
    pooling_->set_attr_pad_mode(0);
    CHECK_NULL_RETURN(pooling_prim->pad());
    auto pad_u = static_cast<int>(*(pooling_prim->pad()->begin() + PAD_UP));
    auto pad_d = static_cast<int>(*(pooling_prim->pad()->begin() + PAD_DOWN));
    auto pad_l = static_cast<int>(*(pooling_prim->pad()->begin() + PAD_LEFT));
    auto pad_r = static_cast<int>(*(pooling_prim->pad()->begin() + PAD_RIGHT));
    pooling_->set_attr_pad(ge::AttrValue::LIST_INT({pad_u, pad_d, pad_l, pad_r}));
  }

  if (pooling_prim->round_mode() == schema::RoundMode_FLOOR) {  // no use in cpu
    pooling_->set_attr_ceil_mode(0);
    pooling_->set_attr_data_mode(1);
  } else {
    pooling_->set_attr_ceil_mode(1);
    pooling_->set_attr_data_mode(0);
  }
  return RET_OK;
}

int MaxPoolingNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                          const std::vector<mindspore::MSTensor> &out_tensors) {
  pooling_ = new (std::nothrow) hiai::op::PoolingD(name_ + "_pooling");
  if (pooling_ == nullptr) {
    MS_LOG(ERROR) << "New pooling npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  auto pooling_prim = primitive->value_as_MaxPoolFusion();
  if (pooling_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  auto ret = SetPoolingParam(pooling_prim);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Set npu op parameter for convolution op " << name_ << " failed.";
    return RET_ERROR;
  }
  act_type_ = pooling_prim->activation_type();
  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    ret = SetActivation(pooling_, pooling_prim->activation_type());
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int MaxPoolingNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                                  const std::vector<mindspore::MSTensor> &out_tensors,
                                  const std::vector<ge::Operator *> &npu_inputs) {
  pooling_->set_input_x(*npu_inputs[0]);
  return RET_OK;
}

ge::Operator *MaxPoolingNPUOp::GetNPUOp() {
  if (act_type_ == schema::ActivationType_NO_ACTIVATION) {
    return pooling_;
  } else {
    return act_;
  }
}

MaxPoolingNPUOp::~MaxPoolingNPUOp() {
  if (pooling_ != nullptr) {
    delete pooling_;
    pooling_ = nullptr;
  }
}
}  // namespace mindspore
