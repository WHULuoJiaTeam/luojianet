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

#include "host_kernels/kernel_utils.h"

#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"

namespace {
const int kDimensionShapeIndex = 0;
const int kDimensionDimsIndex = 1;
const size_t kDimensionNodeInputSize = 2;
}  // namespace

namespace ge {
Status KernelUtils::CheckDimensionNodeInfo(const NodePtr &node_ptr) {
  if (node_ptr == nullptr) {
    GELOGE(FAILED, "parameter is null.");
    return FAILED;
  }
  auto input_nodes = node_ptr->GetInDataNodes();
  if (input_nodes.size() != kDimensionNodeInputSize) {
    GELOGW("op:%s type: %s, dimension input size must be %zu, but get %zu inputs", node_ptr->GetName().c_str(),
           node_ptr->GetType().c_str(), kDimensionNodeInputSize, input_nodes.size());
    return NOT_CHANGED;
  }

  NodePtr dim_node = input_nodes.at(kDimensionDimsIndex);
  if (dim_node == nullptr) {
    GELOGE(PARAM_INVALID, "dim node is nullptr");
    return PARAM_INVALID;
  }

  std::vector<ConstGeTensorPtr> const_ge_tensor = OpDescUtils::GetWeights(dim_node);
  if (const_ge_tensor.empty()) {
    GELOGE(PARAM_INVALID, "dim node must be const op");
    return PARAM_INVALID;
  }
  const ConstGeTensorPtr &input_dim = const_ge_tensor.at(0);
  if (input_dim->GetData().size() == 0) {
    GELOGE(PARAM_INVALID, "dim data size is 0");
    return PARAM_INVALID;
  }

  return SUCCESS;
}

bool KernelUtils::CheckFormatSupported(const NodePtr &node_ptr) {
  if (node_ptr == nullptr) {
    GELOGE(FAILED, "parameter is null.");
    return false;
  }
  OpDescPtr op_desc = node_ptr->GetOpDesc();
  if (op_desc == nullptr) {
    GELOGE(FAILED, "op_desc is null");
    return false;
  }
  const auto &input_desc = op_desc->MutableInputDesc(kDimensionShapeIndex);
  GE_CHECK_NOTNULL_EXEC(input_desc, return false);
  Format fmt = input_desc->GetFormat();
  if (fmt == FORMAT_NC1HWC0 || fmt == FORMAT_FRACTAL_Z) {
    GELOGW("invalid format, fmt: %s", TypeUtils::FormatToSerialString(fmt).c_str());
    return false;
  }

  return true;
}

bool KernelUtils::CheckSizeForTransOp(const ge::ConstGeTensorPtr &const_weight_ptr,
                                      const ge::OpDescPtr &op_desc_ptr) {
  if (const_weight_ptr == nullptr || op_desc_ptr == nullptr) {
    GELOGE(FAILED, "parameter invalid");
    return false;
  }
  auto data_size = const_weight_ptr->GetData().GetSize();
  const auto &input_desc = op_desc_ptr->MutableInputDesc(0);
  GE_CHECK_NOTNULL_EXEC(input_desc, return false);
  DataType data_type = input_desc->GetDataType();
  GeShape data_shape = input_desc->GetShape();
  Format data_format = input_desc->GetFormat();
  auto shape_size = input_desc->GetShape().GetShapeSize();
  int64_t cal_size = 0;

  auto ret = TensorUtils::CalcTensorMemSize(data_shape, data_format, data_type, cal_size);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "CalcTensorMemSize failed");
    return false;
  }

  uint32_t length = 1;
  if (!TypeUtils::GetDataTypeLength(data_type, length)) {
    GELOGE(PARAM_INVALID, "Input datatype %d is not support .", data_type);
    return false;
  }

  GELOGI("Const real value Size:%zu, op_desc Shape Size:%ld, data_type:%s.", data_size, cal_size,
         TypeUtils::DataTypeToSerialString(data_type).c_str());
  if (shape_size != 0) {
    // Standard tensor
    if (data_size != static_cast<size_t>(cal_size) || data_size == 0) {
      GELOGW("Const input data size is not equal with tensor desc shape");
      return false;
    }
  } else if (data_shape.GetDimNum() != 0) {
    // Empty tensor, has zero in shape vector
    if (data_size != 0) {
      GELOGW("Const input data size is not equal with tensor desc shape");
      return false;
    }
  } else {
    // Scalar tensor, has only one element in tensor
    if (length != 0 && (data_size / static_cast<size_t>(length) != 1)) {
      GELOGW("Const input data size is not equal with tensor desc shape");
      return false;
    }
  }

  return true;
}

bool KernelUtils::IsUnknownShape(const ge::GeShape &shape) {
  vector<int64_t> dims = shape.GetDims();
  for (auto dim : dims) {
    if (dim < 0) {
      GELOGW("Shape kernel recoginze unknown shape.Ignore shape kernel.");
      return true;
    }
  }
  return false;
}
}  // namespace ge
