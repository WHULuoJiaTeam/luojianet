/**
 * Copyright 2021, 2022 LuoJiaNET Research and Development Group, Wuhan University
 * Copyright 2021, 2022 Huawei Technologies Co., Ltd
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

#include "op_tiling/op_tiling_utils.h"
#include <string>
#include "graph/ge_tensor.h"
#include "graph/utils/attr_utils.h"
#include "op_tiling/op_tiling_constants.h"

namespace optiling {
void AddNameToTensordesc(ge::OpDescPtr &op_desc) {
  if (!op_desc->HasAttr(ATTR_NAME_OP_INFER_DEPENDS)) {
    return;
  }
  size_t input_size = op_desc->GetAllInputsSize();
  for (size_t i = 0; i < input_size; i++) {
    ge::GeTensorDescPtr tensor_desc_ptr = op_desc->MutableInputDesc(i);
    if (tensor_desc_ptr == nullptr) {
      continue;
    }
    tensor_desc_ptr->SetName(op_desc->GetInputNameByIndex(i));
  }
}

void ReplaceEmptyShapeOfTensorDesc(ge::OpDescPtr &op_desc, std::vector<int32_t> &indexes) {
  size_t input_size = op_desc->GetAllInputsSize();
  for (size_t i = 0; i < input_size; ++i) {
    ge::GeTensorDescPtr tensor_desc_ptr = op_desc->MutableInputDesc(i);
    if (tensor_desc_ptr == nullptr) {
      continue;
    }
    if (tensor_desc_ptr->MutableShape().IsScalar()) {
      indexes.push_back(i);
      tensor_desc_ptr->MutableShape().SetDimNum(1);
      tensor_desc_ptr->MutableShape().SetDim(0, 1);
    }
  }

  size_t output_size = op_desc->GetOutputsSize();
  for (size_t i = 0; i < output_size; ++i) {
    ge::GeTensorDescPtr tensor_desc_ptr = op_desc->MutableOutputDesc(i);
    if (tensor_desc_ptr == nullptr) {
      continue;
    }
    if (tensor_desc_ptr->MutableShape().IsScalar()) {
      indexes.push_back(-1 - i);
      tensor_desc_ptr->MutableShape().SetDimNum(1);
      tensor_desc_ptr->MutableShape().SetDim(0, 1);
    }
  }
}

void RecoveryEmptyShapeOfTensorDesc(ge::OpDescPtr &op_desc, const std::vector<int32_t> &indexes) {
  for (const int32_t &index : indexes) {
    ge::GeTensorDescPtr tensor_desc_ptr;
    if (index >= 0) {
      tensor_desc_ptr = op_desc->MutableInputDesc(index);
    } else {
      tensor_desc_ptr = op_desc->MutableOutputDesc(std::abs(index) - 1);
    }
    if (tensor_desc_ptr == nullptr) {
      continue;
    }
    tensor_desc_ptr->MutableShape().SetDimNum(0);
  }
}
}  // namespace optiling
