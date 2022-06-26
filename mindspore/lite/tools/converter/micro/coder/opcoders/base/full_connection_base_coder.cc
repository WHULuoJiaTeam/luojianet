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

#include "coder/opcoders/base/full_connection_base_coder.h"

namespace mindspore::lite::micro {
FullConnectionBaseCoder::~FullConnectionBaseCoder() {
  fc_param_ = nullptr;
  filter_tensor_ = nullptr;
  bias_tensor_ = nullptr;
}

int FullConnectionBaseCoder::Init() {
  this->fc_param_ = reinterpret_cast<MatMulParameter *>(parameter_);
  filter_tensor_ = input_tensors_.at(kWeightIndex);
  MS_CHECK_PTR(filter_tensor_);
  if (input_tensors_.size() == kInputSize2) {
    bias_tensor_ = input_tensors_.at(kBiasIndex);
    MS_CHECK_PTR(bias_tensor_);
    MS_CHECK_PTR(bias_tensor_->data());
  }
  return RET_OK;
}
}  // namespace mindspore::lite::micro
