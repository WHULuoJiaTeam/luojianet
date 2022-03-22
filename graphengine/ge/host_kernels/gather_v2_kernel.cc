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

#include "host_kernels/gather_v2_kernel.h"

#include <memory>
#include <set>

#include "common/fp16_t.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "host_kernels/kernel_utils.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const size_t kGatherV2InputIndexZero = 0;
const size_t kGatherV2InputIndexOne = 1;
const size_t kGatherV2InputIndexTwo = 2;
const size_t kGatherV2InputIndexThree = 3;
const size_t kGatherV2DimOne = 1;
const size_t kGatherV2InpotNum = 3;
const size_t kMaxIndicatesDims = 1;  // only support scalar and 1 dims indicates_
const std::set<DataType> supported_type = {DT_FLOAT16, DT_DOUBLE, DT_INT8,   DT_INT16,  DT_INT16, DT_INT32,
                                           DT_INT64,   DT_UINT8,  DT_UINT16, DT_UINT32, DT_UINT64};
const int64_t DIM_AXIS_0 = 0;
const int64_t DIM_AXIS_1 = 1;
const int64_t DIM_AXIS_2 = 2;
const int64_t DIM_AXIS_3 = 3;
}  // namespace
template <typename T>
Status GatherV2Kernel::ProcessAxis0(ConstGeTensorPtr tensor_x, GeTensorPtr output) {
  Status ret = SUCCESS;
  T *data_ptr_x = reinterpret_cast<T *>(const_cast<unsigned char *>(tensor_x->GetData().data()));
  T *data_ptr_y = reinterpret_cast<T *>(const_cast<unsigned char *>(output->GetData().data()));
  // index is valid, and no bigger than kGatherV2InputIndexZero
  size_t output_size = output->GetData().size();
  for (int64_t i = 0; i < output->GetTensorDesc().GetShape().GetDim(kGatherV2InputIndexZero); i++) {
    T *data_ptr_x_tmp = data_ptr_x + indicates_[i] * xstride_[kGatherV2InputIndexZero];
    T *data_ptr_y_tmp = data_ptr_y + i * ystride_[kGatherV2InputIndexZero];
    size_t size = sizeof(T) * xstride_[kGatherV2InputIndexZero];
    if (data_ptr_y_tmp - data_ptr_y < 0) {
      GELOGE(PARAM_INVALID, "ptr_y - ptr_y_tmp less than zero");
      return PARAM_INVALID;
    }
    size_t offset_size = (data_ptr_y_tmp - data_ptr_y) * sizeof(T);
    auto ret_mem = memcpy_s(reinterpret_cast<void *>(data_ptr_y_tmp), output_size - offset_size,
                            reinterpret_cast<void *>(data_ptr_x_tmp), size);
    if (ret_mem != 0) {
      GELOGE(MEMALLOC_FAILED, "memcpy failed!");
      return MEMALLOC_FAILED;
    }
  }
  return ret;
}

template <typename T>
Status GatherV2Kernel::ProcessAxis1(ConstGeTensorPtr tensor_x, GeTensorPtr output) {
  Status ret = SUCCESS;
  T *data_ptr_x = reinterpret_cast<T *>(const_cast<unsigned char *>(tensor_x->GetData().data()));
  T *data_ptr_y = reinterpret_cast<T *>(const_cast<unsigned char *>(output->GetData().data()));
  // index is valid, and no bigger than kGatherV2InputIndexOne
  size_t output_size = output->GetData().size();
  for (int64_t i = 0; i < output->GetTensorDesc().GetShape().GetDim(kGatherV2InputIndexZero); i++) {
    T *data_ptr_x_i = data_ptr_x + i * xstride_[kGatherV2InputIndexZero];
    T *data_ptr_y_i = data_ptr_y + i * ystride_[kGatherV2InputIndexZero];
    for (int64_t j = 0; j < output->GetTensorDesc().GetShape().GetDim(kGatherV2InputIndexOne); j++) {
      T *data_ptr_x_tmp = data_ptr_x_i + indicates_[j] * xstride_[kGatherV2InputIndexOne];
      T *data_ptr_y_tmp = data_ptr_y_i + j * ystride_[kGatherV2InputIndexOne];
      size_t size = sizeof(T) * xstride_[kGatherV2InputIndexOne];
      if (data_ptr_y_tmp - data_ptr_y < 0) {
        GELOGE(PARAM_INVALID, "ptr_y - ptr_y_tmp less than zero");
        return PARAM_INVALID;
      }
      size_t offset_size = (data_ptr_y_tmp - data_ptr_y) * sizeof(T);
      auto ret_mem = memcpy_s(reinterpret_cast<void *>(data_ptr_y_tmp), output_size - offset_size,
                              reinterpret_cast<void *>(data_ptr_x_tmp), size);
      if (ret_mem != 0) {
        GELOGE(MEMALLOC_FAILED, "memcpy failed!");
        return MEMALLOC_FAILED;
      }
    }
  }
  return ret;
}

template <typename T>
Status GatherV2Kernel::ProcessAxis2(ConstGeTensorPtr tensor_x, GeTensorPtr output) {
  Status ret = SUCCESS;
  T *data_ptr_x = reinterpret_cast<T *>(const_cast<unsigned char *>(tensor_x->GetData().data()));
  T *data_ptr_y = reinterpret_cast<T *>(const_cast<unsigned char *>(output->GetData().data()));
  // index is valid, and no bigger than kGatherV2InputIndexTwo
  size_t output_size = output->GetData().size();
  for (int64_t i = 0; i < output->GetTensorDesc().GetShape().GetDim(kGatherV2InputIndexZero); i++) {
    T *data_ptr_x_i = data_ptr_x + i * xstride_[kGatherV2InputIndexZero];
    T *data_ptr_y_i = data_ptr_y + i * ystride_[kGatherV2InputIndexZero];
    for (int64_t j = 0; j < output->GetTensorDesc().GetShape().GetDim(kGatherV2InputIndexOne); j++) {
      T *data_ptr_x_j = data_ptr_x_i + j * xstride_[kGatherV2InputIndexOne];
      T *data_ptr_y_j = data_ptr_y_i + j * ystride_[kGatherV2InputIndexOne];
      for (int64_t m = 0; m < output->GetTensorDesc().GetShape().GetDim(kGatherV2InputIndexTwo); m++) {
        T *data_ptr_x_tmp = data_ptr_x_j + indicates_[m] * xstride_[kGatherV2InputIndexTwo];
        T *data_ptr_y_tmp = data_ptr_y_j + m * ystride_[kGatherV2InputIndexTwo];
        size_t size = sizeof(T) * xstride_[kGatherV2InputIndexTwo];
        if (data_ptr_y_tmp - data_ptr_y < 0) {
          GELOGE(PARAM_INVALID, "ptr_y - ptr_y_tmp less than zero");
          return PARAM_INVALID;
        }
        size_t offset_size = (data_ptr_y_tmp - data_ptr_y) * sizeof(T);
        auto ret_mem = memcpy_s(reinterpret_cast<void *>(data_ptr_y_tmp), output_size - offset_size,
                                reinterpret_cast<void *>(data_ptr_x_tmp), size);
        if (ret_mem != 0) {
          GELOGE(MEMALLOC_FAILED, "memcpy failed!");
          return MEMALLOC_FAILED;
        }
      }
    }
  }
  return ret;
}

template <typename T>
Status GatherV2Kernel::ProcessAxis3(ConstGeTensorPtr tensor_x, GeTensorPtr output) {
  Status ret = SUCCESS;
  T *data_ptr_x = reinterpret_cast<T *>(const_cast<unsigned char *>(tensor_x->GetData().data()));
  T *data_ptr_y = reinterpret_cast<T *>(const_cast<unsigned char *>(output->GetData().data()));
  // index is valid, and no bigger than kGatherV2InputIndexThree
  size_t output_size = output->GetData().size();
  for (int64_t i = 0; i < output->GetTensorDesc().GetShape().GetDim(kGatherV2InputIndexZero); i++) {
    T *data_ptr_x_i = data_ptr_x + i * xstride_[kGatherV2InputIndexZero];
    T *data_ptr_y_i = data_ptr_y + i * ystride_[kGatherV2InputIndexZero];
    for (int64_t j = 0; j < output->GetTensorDesc().GetShape().GetDim(kGatherV2InputIndexOne); j++) {
      T *data_ptr_x_j = data_ptr_x_i + j * xstride_[kGatherV2InputIndexOne];
      T *data_ptr_y_j = data_ptr_y_i + j * ystride_[kGatherV2InputIndexOne];
      for (int64_t m = 0; m < output->GetTensorDesc().GetShape().GetDim(kGatherV2InputIndexTwo); m++) {
        T *data_ptr_x_m = data_ptr_x_j + m * xstride_[kGatherV2InputIndexTwo];
        T *data_ptr_y_m = data_ptr_y_j + m * ystride_[kGatherV2InputIndexTwo];
        for (int64_t n = 0; n < output->GetTensorDesc().GetShape().GetDim(kGatherV2InputIndexThree); n++) {
          T *data_ptr_x_tmp = data_ptr_x_m + indicates_[n] * xstride_[kGatherV2InputIndexThree];
          T *data_ptr_y_tmp = data_ptr_y_m + n * ystride_[kGatherV2InputIndexThree];
          size_t size = sizeof(T) * xstride_[kGatherV2InputIndexThree];
          if (data_ptr_y_tmp - data_ptr_y < 0) {
            GELOGE(PARAM_INVALID, "ptr_y - ptr_y_tmp less than zero");
            return PARAM_INVALID;
          }
          size_t offset_size = (data_ptr_y_tmp - data_ptr_y) * sizeof(T);
          auto ret_mem = memcpy_s(reinterpret_cast<void *>(data_ptr_y_tmp), output_size - offset_size,
                                  reinterpret_cast<void *>(data_ptr_x_tmp), size);
          if (ret_mem != 0) {
            GELOGE(MEMALLOC_FAILED, "memcpy failed!");
            return MEMALLOC_FAILED;
          }
        }
      }
    }
  }
  return ret;
}

template <typename T>
Status GatherV2Kernel::GenData(const int64_t data_num, ConstGeTensorPtr tensor_x, int64_t axis, GeTensorPtr output) {
  if (data_num <= 0) {
    return PARAM_INVALID;
  }
  if (!CheckInt64MulOverflow(data_num, sizeof(T))) {
    GELOGE(PARAM_INVALID, "Int64MulOverflow, data_num:%ld, type_len:%zu.", data_num, sizeof(T));
    return PARAM_INVALID;
  }

  std::unique_ptr<T[]> buf(new (std::nothrow) T[data_num]());
  if (buf == nullptr) {
    GELOGE(MEMALLOC_FAILED, "New sizeof(T) * data_num(%zu) memory failed", static_cast<size_t>(sizeof(T) * data_num));
    return MEMALLOC_FAILED;
  }
  GE_IF_BOOL_EXEC(
    output->SetData(reinterpret_cast<uint8_t *>(buf.get()), static_cast<size_t>(data_num * sizeof(T))) != GRAPH_SUCCESS,
    GELOGE(INTERNAL_ERROR, "set data failed");
    return INTERNAL_ERROR);

  Status ret = SUCCESS;
  switch (axis) {
    case DIM_AXIS_0:
      ret = ProcessAxis0<T>(tensor_x, output);
      break;
    case DIM_AXIS_1:
      ret = ProcessAxis1<T>(tensor_x, output);
      break;
    case DIM_AXIS_2:
      ret = ProcessAxis2<T>(tensor_x, output);
      break;
    case DIM_AXIS_3:
      ret = ProcessAxis3<T>(tensor_x, output);
      break;
    default:
      GELOGI("Only support 4 dims and below but input axis is %ld", axis);
      return NOT_CHANGED;
  }
  return ret;
}
Status GatherV2Kernel::CalcStride(std::vector<int64_t> &stride, std::vector<int64_t> dims) {
  if (stride.size() != dims.size() || dims.size() == 0) {
    return PARAM_INVALID;
  }
  int i = static_cast<int>(dims.size() - kGatherV2DimOne);
  stride[static_cast<size_t>(i)] = static_cast<int64_t>(kGatherV2DimOne);
  i--;
  while (i >= 0) {
    size_t index = static_cast<size_t>(i) + kGatherV2DimOne;
    if (!CheckInt64MulOverflow(stride[index], dims[index])) {
      GELOGE(PARAM_INVALID, "Int64MulOverflow, data_num(%ld) type_len(%ld)", stride[index], dims[index]);
      return PARAM_INVALID;
    }
    stride[static_cast<size_t>(i)] = stride[index] * dims[index];
    i--;
  }
  return SUCCESS;
}
Status GatherV2Kernel::Process(int64_t axis, DataType data_type, ConstGeTensorPtr input_tensor_ptr,
                               GeTensorPtr output_ptr) {
  Status ret = SUCCESS;
  int64_t data_num = output_ptr->GetTensorDesc().GetShape().GetShapeSize();
  switch (data_type) {
    case DT_FLOAT16:
      ret = GenData<fp16_t>(data_num, input_tensor_ptr, axis, output_ptr);
      break;
    case DT_DOUBLE:
      ret = GenData<double>(data_num, input_tensor_ptr, axis, output_ptr);
      break;
    case DT_INT8:
      ret = GenData<int8_t>(data_num, input_tensor_ptr, axis, output_ptr);
      break;
    case DT_INT16:
      ret = GenData<int16_t>(data_num, input_tensor_ptr, axis, output_ptr);
      break;
    case DT_INT32:
      ret = GenData<int32_t>(data_num, input_tensor_ptr, axis, output_ptr);
      break;
    case DT_INT64:
      ret = GenData<int64_t>(data_num, input_tensor_ptr, axis, output_ptr);
      break;
    case DT_UINT8:
      ret = GenData<uint8_t>(data_num, input_tensor_ptr, axis, output_ptr);
      break;
    case DT_UINT16:
      ret = GenData<uint16_t>(data_num, input_tensor_ptr, axis, output_ptr);
      break;
    case DT_UINT32:
      ret = GenData<uint32_t>(data_num, input_tensor_ptr, axis, output_ptr);
      break;
    case DT_UINT64:
      ret = GenData<uint64_t>(data_num, input_tensor_ptr, axis, output_ptr);
      break;
    default:
      GELOGI("GatherV2Kernel does not support this Data type:%s", TypeUtils::DataTypeToSerialString(data_type).c_str());
      return NOT_CHANGED;
  }
  return ret;
}
Status GatherV2Kernel::SaveIndicesByDataType(ConstGeTensorPtr indices_tensor_ptr, GeShape &x_shape,
                                             GeShape &indices_shape, DataType indices_data_type, size_t axis) {
  if (indices_data_type == DT_INT32) {
    auto indices_ptr = const_cast<int32_t *>(reinterpret_cast<const int32_t *>(indices_tensor_ptr->GetData().data()));
    for (int64_t i = 0; i < indices_shape.GetShapeSize(); i++) {
      if (*(indices_ptr + i) < 0 || *(indices_ptr + i) >= x_shape.GetDim(axis)) {
        GELOGW("indices %ld value is not in range [0, %ld).", i, x_shape.GetDim(axis));
        return NOT_CHANGED;
      }
      indicates_.push_back(*(indices_ptr + i));
    }
  } else {
    // int64
    auto indices_ptr = const_cast<int64_t *>(reinterpret_cast<const int64_t *>(indices_tensor_ptr->GetData().data()));
    for (int64_t i = 0; i < indices_shape.GetShapeSize(); i++) {
      if (*(indices_ptr + i) < 0 || *(indices_ptr + i) >= x_shape.GetDim(axis)) {
        GELOGW("indices %ld value is not in range [0, %ld).", i, x_shape.GetDim(axis));
        return NOT_CHANGED;
      }
      indicates_.push_back(*(indices_ptr + i));
    }
  }

  return SUCCESS;
}
Status GatherV2Kernel::Check(const OpDescPtr &op_desc_ptr, const vector<ConstGeTensorPtr> &input,
                             vector<GeTensorPtr> &v_output) const {
  if (op_desc_ptr == nullptr) {
    GELOGW("input opdesc is nullptr.");
    return NOT_CHANGED;
  }

  if (input.size() != kGatherV2InpotNum) {
    GELOGW("The number of input for GatherV2 must be %zu.", kGatherV2InpotNum);
    return NOT_CHANGED;
  }

  bool is_null = (input[kGatherV2InputIndexZero] == nullptr || input[kGatherV2InputIndexOne] == nullptr ||
                  input[kGatherV2InputIndexTwo] == nullptr);
  if (is_null) {
    GELOGW("some input is nullptr.");
    return NOT_CHANGED;
  }
  ConstGeTensorPtr tensor0 = input.at(kGatherV2InputIndexZero);
  ConstGeTensorPtr tensor1 = input.at(kGatherV2InputIndexOne);
  ConstGeTensorPtr tensor2 = input.at(kGatherV2InputIndexTwo);

  bool size_is_zero =
    ((tensor0->GetData().size() == 0) || (tensor1->GetData().size() == 0) || (tensor2->GetData().size() == 0));
  if (size_is_zero) {
    GELOGW("some input size is zero.");
    return NOT_CHANGED;
  }

  auto indices_shape = tensor1->GetTensorDesc().GetShape();
  auto axis_shape = tensor2->GetTensorDesc().GetShape();
  // axis must be scalar
  if (axis_shape.GetDimNum() != 0) {
    GELOGW("axis must be scalar but its shape is %zu", axis_shape.GetDimNum());
    return NOT_CHANGED;
  }
  auto axis_data_type = tensor2->GetTensorDesc().GetDataType();
  bool is_valid_axis_data_type = axis_data_type == DT_INT32 || axis_data_type == DT_INT64;
  if (!is_valid_axis_data_type) {
    GELOGW("axis datatype must be DT_INT32 or DT_INT64");
    return NOT_CHANGED;
  }

  // check indices data_type && dims && every element
  auto indices_data_type = tensor1->GetTensorDesc().GetDataType();
  bool is_valid_indices_data_type = indices_data_type == DT_INT32 || indices_data_type == DT_INT64;
  if (!is_valid_indices_data_type) {
    GELOGW("indices datatype must be DT_INT32 or DT_INT64.");
    return NOT_CHANGED;
  }
  if (indices_shape.GetDimNum() > kMaxIndicatesDims) {
    GELOGW("indices input only support 0 or 1 dims.");
    return NOT_CHANGED;
  }
  return SUCCESS;
}
void GatherV2Kernel::DebugPrint(int64_t axis, const GeShape &x_shape, const GeShape &indices_shape,
                                const std::vector<int64_t> &y_shape) {
  GELOGD("GatherV2Kernel axis:%ld x_shape:%zu indices_shape:%zu y_shape:%zu.", axis, x_shape.GetDimNum(),
         indices_shape.GetDimNum(), y_shape.size());
  for (size_t i = 0; i < x_shape.GetDimNum(); i++) {
    GELOGD("GatherV2Kernel x_shape[%zu]: %ld.", i, x_shape.GetDim(i));
  }
  for (size_t i = 0; i < indices_shape.GetDimNum(); i++) {
    GELOGD("GatherV2Kernel indices_shape[%zu]: %ld.", i, indices_shape.GetDim(i));
  }
  for (size_t i = 0; i < y_shape.size(); i++) {
    GELOGD("GatherV2Kernel y_shape[%zu]: %ld.", i, y_shape[i]);
  }
  for (auto ele : indicates_) {
    GELOGD("GatherV2Kernel indices:%ld.", ele);
  }
}

Status GatherV2Kernel::Compute(const OpDescPtr op_desc_ptr, const vector<ConstGeTensorPtr> &input,
                               vector<GeTensorPtr> &v_output) {
  GELOGI("Enter GatherV2Kernel Process.");
  Status ret = Check(op_desc_ptr, input, v_output);
  if (ret != SUCCESS) {
    GELOGW("param check failed");
    return NOT_CHANGED;
  }
  GELOGI("GatherV2Kernel[%s] start Process", op_desc_ptr->GetName().c_str());
  ConstGeTensorPtr tensor0 = input.at(kGatherV2InputIndexZero);
  ConstGeTensorPtr tensor1 = input.at(kGatherV2InputIndexOne);
  ConstGeTensorPtr tensor2 = input.at(kGatherV2InputIndexTwo);

  auto x_shape = tensor0->GetTensorDesc().GetShape();
  auto indices_shape = tensor1->GetTensorDesc().GetShape();

  auto axis_data_type = tensor2->GetTensorDesc().GetDataType();
  int64_t axis = axis_data_type == DT_INT32
                   ? *(const_cast<int32_t *>(reinterpret_cast<const int32_t *>(tensor2->GetData().data())))
                   : *(const_cast<int64_t *>(reinterpret_cast<const int64_t *>(tensor2->GetData().data())));
  axis = axis >= 0 ? axis : axis + x_shape.GetDimNum();
  // check axis value
  if (axis < 0 || (axis + 1) > static_cast<int64_t>(x_shape.GetDimNum())) {
    GELOGW("axis is invalid!");
    return NOT_CHANGED;
  }
  auto indices_data_type = tensor1->GetTensorDesc().GetDataType();
  ret = SaveIndicesByDataType(tensor1, x_shape, indices_shape, indices_data_type, static_cast<size_t>(axis));
  if (ret != SUCCESS) {
    GELOGW("Save indeices by data type failed!");
    return ret;
  }

  // check input data type
  auto x_data_type = tensor0->GetTensorDesc().GetDataType();
  if (supported_type.find(x_data_type) == supported_type.end()) {
    GELOGI("GatherV2Kernel does not support this Data type:%s.",
           TypeUtils::DataTypeToSerialString(x_data_type).c_str());
    return NOT_CHANGED;
  }
  // calc output shape
  std::vector<int64_t> y_shape;
  for (size_t i = 0; i < static_cast<size_t>(axis); i++) {
    y_shape.push_back(x_shape.GetDim(i));
  }
  for (size_t i = 0; i < indices_shape.GetDimNum(); i++) {
    y_shape.push_back(indices_shape.GetDim(i));
  }
  for (size_t i = static_cast<size_t>(axis) + 1; i < x_shape.GetDimNum(); i++) {
    y_shape.push_back(x_shape.GetDim(i));
  }

  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(0));
  if (output_ptr == nullptr) {
    GELOGW("make_shared ge::GeTensor failed, node name %s.", op_desc_ptr->GetName().c_str());
    return NOT_CHANGED;
  }
  output_ptr->MutableTensorDesc().SetShape(GeShape(y_shape));
  output_ptr->MutableTensorDesc().SetDataType(x_data_type);

  // added for debug
  DebugPrint(axis, x_shape, indices_shape, y_shape);

  // calc stride
  std::vector<int64_t> xstride(x_shape.GetDimNum());
  std::vector<int64_t> ystride(y_shape.size());
  xstride_ = xstride;
  ystride_ = ystride;
  auto ret_x = CalcStride(xstride_, x_shape.GetDims());
  auto ret_y = CalcStride(ystride_, y_shape);
  ret = (ret_x == SUCCESS && ret_y == SUCCESS) ? SUCCESS : NOT_CHANGED;
  if (ret != SUCCESS) {
    GELOGE(ret, "CalcStride Failed");
    return ret;
  }

  ret = Process(axis, x_data_type, tensor0, output_ptr);
  if (ret != SUCCESS) {
    GELOGE(ret, "GenData failed, data_type: %s", TypeUtils::DataTypeToSerialString(x_data_type).c_str());
    return ret;
  }

  GELOGI("GatherV2Kernel Process Success.");
  v_output.push_back(output_ptr);
  return SUCCESS;
}
REGISTER_KERNEL(GATHERV2, GatherV2Kernel);
}  // namespace ge
