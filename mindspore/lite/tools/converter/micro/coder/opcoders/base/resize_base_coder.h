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

#ifndef MINDSPORE_LITE_MICRO_CODER_OPCODERS_RESIZE_BASE_CODER_H_
#define MINDSPORE_LITE_MICRO_CODER_OPCODERS_RESIZE_BASE_CODER_H_

#include <vector>
#include <memory>
#include "coder/opcoders/op_coder.h"
#include "nnacl/resize_parameter.h"

namespace mindspore::lite::micro {
class ResizeBaseCoder : public OperatorCoder {
 public:
  ResizeBaseCoder(const std::vector<Tensor *> &in_tensors, const std::vector<Tensor *> &out_tensors,
                  const Model::Node *node, size_t node_index, Target target)
      : OperatorCoder(in_tensors, out_tensors, node, node_index, target) {}

  ~ResizeBaseCoder() override = default;

  int Init();

 protected:
  int method_{0};
  int new_height_{0};
  int new_width_{0};
  int coordinate_transform_mode_{0};
  bool preserve_aspect_ratio_{false};
  bool const_shape_{false};

 private:
  int CheckParameters();
  int CheckInputsOuputs();
};
}  // namespace mindspore::lite::micro
#endif  // MINDSPORE_LITE_MICRO_CODER_OPCODERS_RESIZE_BASE_CODER_H_
