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

#include "host_kernels/floordiv_kernel.h"

#include <cfloat>

#include <memory>
#include <set>

#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "host_kernels/kernel_utils.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const size_t kFloorDivInputX = 0;
const size_t kFloorDivInputY = 1;
const size_t kFloorDivTensorShapeIsEmpty = 0;
const size_t kFloorDivInputSize = 2;
const std::set<DataType> kFloorDivSupportedType = {DT_FLOAT,  DT_DOUBLE, DT_UINT8, DT_INT8,
                                                   DT_UINT16, DT_INT16,  DT_INT32, DT_INT64};
}  // namespace
Status FloorDivKernel::FloorDivCheck(const OpDescPtr &op_desc_ptr,
                                     const std::vector<ge::ConstGeTensorPtr> &input) const {
  // check input size
  if (op_desc_ptr == nullptr) {
    GELOGW("Input opdesc is nullptr.");
    return PARAM_INVALID;
  }
  if (input.size() != kFloorDivInputSize) {
    GELOGW("Unexpected FloorDiv node, node input size: %zu, node name: %s", input.size(),
           op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }

  // check dims of x and y
  ConstGeTensorPtr x_tensor = input.at(kFloorDivInputX);
  ConstGeTensorPtr y_tensor = input.at(kFloorDivInputY);
  GE_CHECK_NOTNULL(x_tensor);
  GE_CHECK_NOTNULL(y_tensor);
  if (x_tensor->GetTensorDesc().GetShape().GetDimNum() != kFloorDivTensorShapeIsEmpty &&
      y_tensor->GetTensorDesc().GetShape().GetDimNum() != kFloorDivTensorShapeIsEmpty) {
    // x and y are not scalars
    vector<int64_t> x_dims = x_tensor->GetTensorDesc().GetShape().GetDims();
    vector<int64_t> y_dims = y_tensor->GetTensorDesc().GetShape().GetDims();
    if (x_dims.size() != y_dims.size()) {
      GELOGW("FloorDivKernel dims of x and y do not match, node name: %s", op_desc_ptr->GetName().c_str());
      return PARAM_INVALID;
    } else {
      for (size_t i = 0; i < x_dims.size(); ++i) {
        if (x_dims[i] != y_dims[i]) {
          GELOGW("FloorDivKernel dims of x and y do not match, node name: %s", op_desc_ptr->GetName().c_str());
          return PARAM_INVALID;
        }
      }
    }
  }

  // check data type
  DataType x_data_dtype = x_tensor->GetTensorDesc().GetDataType();
  DataType y_data_dtype = y_tensor->GetTensorDesc().GetDataType();
  if (x_data_dtype != y_data_dtype) {
    GELOGW("FloorDivKernel data type of x and y do not match, x data type is %s, but y data type is %s, node name: %s.",
           TypeUtils::DataTypeToSerialString(x_data_dtype).c_str(),
           TypeUtils::DataTypeToSerialString(y_data_dtype).c_str(), op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }
  if (kFloorDivSupportedType.find(x_data_dtype) == kFloorDivSupportedType.end()) {
    GELOGW("FloorDivKernel data type %s not support, node name: %s",
           TypeUtils::DataTypeToSerialString(x_data_dtype).c_str(), op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }

  // check data
  if (x_tensor->GetData().size() == 0 || y_tensor->GetData().size() == 0) {
    GELOGW("FloorDivKernel data size of inputs is 0, node name: %s", op_desc_ptr->GetName().c_str());
    return PARAM_INVALID;
  }

  return SUCCESS;
}

void FloorDivKernel::ShapeCal(const std::vector<ge::ConstGeTensorPtr> &input, GeTensorPtr output_ptr) {
  vector<int64_t> output_dims;
  size_t x_dim = input.at(kFloorDivInputX)->GetTensorDesc().GetShape().GetDimNum();
  size_t y_dim = input.at(kFloorDivInputY)->GetTensorDesc().GetShape().GetDimNum();
  if (x_dim >= y_dim) {
    output_dims = input.at(kFloorDivInputX)->GetTensorDesc().GetShape().GetDims();
  } else {
    output_dims = input.at(kFloorDivInputY)->GetTensorDesc().GetShape().GetDims();
  }
  output_ptr->MutableTensorDesc().SetShape(GeShape(output_dims));
}

template <typename T>
T FloorDivKernel::DivCal(const T &x_i, const T &y_i) {
  if ((x_i < static_cast<T>(0)) != (y_i < static_cast<T>(0))) {
    T abs_x_i = x_i < 0 ? -x_i : x_i;
    T abs_y_i = y_i < 0 ? -y_i : y_i;
    return static_cast<T>(static_cast<int32_t>(-(abs_x_i + abs_y_i - 1) / abs_y_i));
  } else {
    return static_cast<T>(static_cast<int32_t>(x_i / y_i));
  }
}

template <typename T>
bool FloorDivKernel::ZeroCheck(const T &element, DataType data_type) {
  bool result = false;
  if (data_type == DT_UINT8 || data_type == DT_INT8 || data_type == DT_UINT16 || data_type == DT_INT16 ||
      data_type == DT_INT32 || data_type == DT_INT64) {
    result = (element == 0);
  } else if (data_type == DT_FLOAT) {
    result = (fabs(element) < FLT_EPSILON);
  } else if (data_type == DT_DOUBLE) {
    result = (fabs(element) < DBL_EPSILON);
  }
  return result;
}

template <typename T>
Status FloorDivKernel::DataCalBroadcast(const T &x, const T &y, size_t num_x, size_t num_y, DataType data_type,
                                        GeTensorPtr output_ptr) {
  size_t data_num = (num_x > num_y) ? num_x : num_y;
  unique_ptr<T[]> buf(new (std::nothrow) T[data_num]());
  if (buf == nullptr) {
    GELOGE(MEMALLOC_FAILED, "new buf failed");
    return INTERNAL_ERROR;
  }

  if (num_x > num_y) {
    if (ZeroCheck<T>(y, data_type)) {
      GELOGE(PARAM_INVALID, "The divisor of FloorDiv can not be zero.");
      return PARAM_INVALID;
    }
    for (size_t i = 0; i < num_x; ++i) {
      buf[i] = DivCal<T>((&x)[i], y);
    }
  } else {
    for (size_t i = 0; i < num_y; ++i) {
      if (ZeroCheck<T>((&y)[i], data_type)) {
        GELOGE(PARAM_INVALID, "The divisor of FloorDiv can not be zero.");
        return PARAM_INVALID;
      }
      buf[i] = DivCal<T>(x, (&y)[i]);
    }
  }
  if (output_ptr->SetData(reinterpret_cast<uint8_t *>(buf.get()), data_num * sizeof(T)) != GRAPH_SUCCESS) {
    GELOGE(PARAM_INVALID, "set data failed");
    return PARAM_INVALID;
  }

  return SUCCESS;
}

template <typename T>
Status FloorDivKernel::DataCal(const std::vector<ConstGeTensorPtr> &input, GeTensorPtr output_ptr) {
  ConstGeTensorPtr x_tensor = input.at(kFloorDivInputX);
  ConstGeTensorPtr y_tensor = input.at(kFloorDivInputY);
  GE_CHECK_NOTNULL(x_tensor);
  GE_CHECK_NOTNULL(y_tensor);
  T *x = const_cast<T *>(reinterpret_cast<const T *>(x_tensor->GetData().GetData()));
  T *y = const_cast<T *>(reinterpret_cast<const T *>(y_tensor->GetData().GetData()));
  if (x == nullptr || y == nullptr) {
    GELOGE(PARAM_INVALID, "Input tensor is nullptr.");
    return PARAM_INVALID;
  }

  size_t data_num_x = x_tensor->GetData().size() / sizeof(T);
  size_t data_num_y = y_tensor->GetData().size() / sizeof(T);
  DataType data_type = x_tensor->GetTensorDesc().GetDataType();
  if (x_tensor->GetTensorDesc().GetShape().GetDimNum() == y_tensor->GetTensorDesc().GetShape().GetDimNum()) {
    // x and y are both scalars or vector, no need broadcast
    unique_ptr<T[]> buf(new (std::nothrow) T[data_num_x]());
    if (buf == nullptr) {
      GELOGE(MEMALLOC_FAILED, "new buf failed");
      return INTERNAL_ERROR;
    }

    for (size_t i = 0; i < data_num_x; ++i) {
      if (ZeroCheck<T>(y[i], data_type)) {
        GELOGE(PARAM_INVALID, "The divisor of FloorDiv can not be zero.");
        return PARAM_INVALID;
      }
      buf[i] = DivCal<T>(x[i], y[i]);
    }
    if (output_ptr->SetData(reinterpret_cast<uint8_t *>(buf.get()), data_num_x * sizeof(T)) != GRAPH_SUCCESS) {
      GELOGE(PARAM_INVALID, "set data failed");
      return PARAM_INVALID;
    }
  } else {
    // x-y is vector-scalar, need broadcast
    if (DataCalBroadcast<T>(*x, *y, data_num_x, data_num_y, data_type, output_ptr) != SUCCESS) {
      return PARAM_INVALID;
    }
  }
  return SUCCESS;
}

Status FloorDivKernel::ComputeByDataType(DataType data_type, const std::vector<ConstGeTensorPtr> &input,
                                         GeTensorPtr output_ptr) {
  Status ret;
  switch (data_type) {
    case DT_FLOAT:
      ret = DataCal<float>(input, output_ptr);
      break;
    case DT_DOUBLE:
      ret = DataCal<double>(input, output_ptr);
      break;
    case DT_UINT8:
      ret = DataCal<uint8_t>(input, output_ptr);
      break;
    case DT_INT8:
      ret = DataCal<int8_t>(input, output_ptr);
      break;
    case DT_UINT16:
      ret = DataCal<uint16_t>(input, output_ptr);
      break;
    case DT_INT16:
      ret = DataCal<int16_t>(input, output_ptr);
      break;
    case DT_INT32:
      ret = DataCal<int32_t>(input, output_ptr);
      break;
    case DT_INT64:
      ret = DataCal<int64_t>(input, output_ptr);
      break;
    default:
      GELOGW("FloorDivKernel does not support Data type:%s", TypeUtils::DataTypeToSerialString(data_type).c_str());
      return NOT_CHANGED;
  }
  return ret;
}

Status FloorDivKernel::Compute(const OpDescPtr op_desc_ptr, const std::vector<ConstGeTensorPtr> &input,
                               std::vector<GeTensorPtr> &v_output) {
  GELOGI("FloorDivKernel in");
  if (FloorDivCheck(op_desc_ptr, input) != SUCCESS) {
    GELOGW("FloorDivKernel input is invalid, failed to fold node.");
    return NOT_CHANGED;
  }

  // Index 0 can always gets a GeTensorDesc object from any OpDescPtr.
  auto output_tensor_desc = op_desc_ptr->GetOutputDesc(0);
  GeTensorPtr output_ptr = MakeShared<GeTensor>(output_tensor_desc);
  if (output_ptr == nullptr) {
    GELOGW("make_shared ge::GeTensor failed, node name %s.", op_desc_ptr->GetName().c_str());
    return NOT_CHANGED;
  }

  // calculate shape
  ShapeCal(input, output_ptr);

  // calculate data and data type
  DataType x_data_dtype = input.at(kFloorDivInputX)->GetTensorDesc().GetDataType();
  output_ptr->MutableTensorDesc().SetDataType(x_data_dtype);
  if (ComputeByDataType(x_data_dtype, input, output_ptr) != SUCCESS) {
    return NOT_CHANGED;
  }

  // print output tensor information, and will be deleted
  GELOGD("FloorDiv op %s output tensor data size is %zu", op_desc_ptr->GetName().c_str(), output_ptr->GetData().size());
  vector<int64_t> data_dims = output_ptr->GetTensorDesc().GetShape().GetDims();
  GELOGD("FloorDiv op %s output tensor dim size is %zu", op_desc_ptr->GetName().c_str(), data_dims.size());

  v_output.emplace_back(output_ptr);
  GELOGI("FloorDivKernel success.");
  return SUCCESS;
}
REGISTER_KERNEL(FLOORDIV, FloorDivKernel);
}  // namespace ge
