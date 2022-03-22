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

#include "host_kernels/dynamic_stitch_kernel.h"

#include <securec.h>
#include <memory>

#include "common/fp16_t.h"
#include "framework/common/ge_inner_error_codes.h"
#include "common/math/math_util.h"
#include "framework/common/op/ge_op_utils.h"
#include "framework/common/types.h"
#include "framework/common/debug/ge_log.h"
#include "graph/utils/type_utils.h"
#include "inc/kernel_factory.h"

namespace ge {
namespace {
const int kDoubleAttrN = 2;
const int kFirstOutputDescIdx = 0;
const int kMergedShapeSecondDim = 1;
const size_t kNullTensorDimNum = 1;
const int64_t kNullTensorDimValue = 0;
const std::set<DataType> kSupportedTypeSet = {DT_INT8,  DT_UINT8, DT_INT16,   DT_UINT16, DT_INT32,
                                              DT_INT64, DT_BOOL,  DT_FLOAT16, DT_FLOAT,  DT_DOUBLE};
}  // namespace
Status DynamicStitchKernel::Compute(const OpDescPtr op_desc_ptr, const vector<ConstGeTensorPtr> &input,
                                    vector<GeTensorPtr> &v_output) {
  GELOGD("DynamicStitch Kernel in.");
  Status validate_ret = ValidateParams(op_desc_ptr, input);
  if (validate_ret != SUCCESS) {
    GELOGW("Dynamic stitch kernel params validate failed.");
    return NOT_CHANGED;
  }

  // OutputDesc size is not null, validated before
  GeTensorPtr output_ptr = MakeShared<GeTensor>(op_desc_ptr->GetOutputDesc(kFirstOutputDescIdx));
  if (output_ptr == nullptr) {
    GELOGW("Fail to malloc output.");
    return NOT_CHANGED;
  }
  Status ret = GenData(input, output_ptr);
  if (ret != SUCCESS) {
    GELOGW("Dynamic stitch folding failed.");
    return NOT_CHANGED;
  }
  v_output.push_back(output_ptr);
  GELOGD("Dynamic stitch end.");
  return SUCCESS;
}

Status DynamicStitchKernel::ValidateParams(const OpDescPtr &op_desc_ptr, const std::vector<ConstGeTensorPtr> &input) {
  if (op_desc_ptr == nullptr) {
    GELOGW("Input op_desc is nullptr.");
    return PARAM_INVALID;
  }
  if (op_desc_ptr->GetOutputsSize() == 0) {
    GELOGW("Current output_desc is empty.");
    return PARAM_INVALID;
  }
  // validate input
  // input[0]~input[N-1] is indices, input[N]~input[2N-1] is data
  if (input.empty()) {
    GELOGI("Input is empty. Ignore dynamic stitch kernel.");
    return NOT_CHANGED;
  }
  for (const auto &in : input) {
    if (in == nullptr) {
      GELOGW("input is nullptr.");
      return PARAM_INVALID;
    }
  }
  // validate attrs
  if (!(AttrUtils::GetInt(op_desc_ptr, ATTR_NAME_N, n_))) {
    GELOGW("Attr %s is not exist.", ATTR_NAME_N.c_str());
    return NOT_CHANGED;
  }
  // validate attr N and input.size
  if ((kDoubleAttrN * n_) > static_cast<int>(input.size())) {
    GELOGW("Input size %zu is not not match with attr %d. Ignore dynamic stitch kernel.", input.size(), n_);
    return NOT_CHANGED;
  }
  // validate supported datatype
  DataType data_type = input[n_]->GetTensorDesc().GetDataType();
  if (kSupportedTypeSet.find(data_type) == kSupportedTypeSet.end()) {
    GELOGW("Input data_type %s is not supported. Please check IR definition. Ignore dynamic stitch kernel.",
           TypeUtils::DataTypeToSerialString(data_type).c_str());
    return NOT_CHANGED;
  }
  return SUCCESS;
}

void DynamicStitchKernel::ComputeMergedShape(const vector<ConstGeTensorPtr> &input, GeShape &merged_shape) {
  // Safety note: index [1~2*n_] for input is valid, and all input is not null, validated in ValidateParams
  // merged.shape = [max(indices)] + step
  // 1. Compute merged first dim, which is the max index.
  int32_t merged_first_dim = 0;
  int64_t indices_shape_size = 0;
  for (int i = 0; i < n_; i++) {
    // shape is [] means scalar
    indices_shape_size =
      input[i]->GetTensorDesc().GetShape().GetDims().empty() ? 1 : input[i]->GetTensorDesc().GetShape().GetShapeSize();
    const int32_t *input_indices = reinterpret_cast<const int32_t *>(input[i]->GetData().data());
    for (int64_t j = 0; j < indices_shape_size; j++) {
      merged_first_dim = std::max(merged_first_dim, input_indices[j]);
    }
  }
  // 2. Compute step, which is follow : step = data[t].shape - indices[t].shape
  size_t indices_dim_num = input[0]->GetTensorDesc().GetShape().GetDimNum();
  GeShape data_shape = input[n_]->GetTensorDesc().GetShape();
  int64_t step = (data_shape.GetDimNum() == indices_dim_num) ? 0 : data_shape.GetDim(indices_dim_num);

  vector<int64_t> merged_dim_vec = {merged_first_dim + 1};
  if (step > 0) {
    merged_dim_vec.emplace_back(step);
    GELOGD("merged_shape is [ %d, %ld].", merged_first_dim, step);
  }
  merged_shape = GeShape(merged_dim_vec);
  GELOGD("merged_shape is [ %d ].", merged_first_dim);
}

Status DynamicStitchKernel::GenData(const vector<ConstGeTensorPtr> &input, GeTensorPtr &output_ptr) {
  // Safety note: index [1~2*n_] for input is valid, and all input is not null, validated in ValidateParams
  GeShape merged_shape;
  ComputeMergedShape(input, merged_shape);
  auto data_type = input[n_]->GetTensorDesc().GetDataType();

  // 1.calc output data size
  auto output_size = merged_shape.GetShapeSize();
  int64_t data_size = GetSizeByDataType(data_type);
  auto step = merged_shape.GetDim(kMergedShapeSecondDim);
  if (!CheckInt64MulOverflow(output_size, data_size) || !CheckInt64MulOverflow(step, data_size)) {
    GELOGW("Check int64 mul overflow failed. Output_size is %ld, data_size is %ld, step is %ld.", output_size,
           data_size, step);
    return NOT_CHANGED;
  }
  auto allowance = output_size * data_size;
  auto data_unit = step > 0 ? step * data_size : data_size;
  // 2.allocate memery for output
  std::unique_ptr<uint8_t[]> buf(new (std::nothrow) uint8_t[allowance]);
  if (buf == nullptr) {
    GELOGW("new buffer failed");
    return INTERNAL_ERROR;
  }
  // 3.copy data from input_data along with the sequence of input_indices
  Status stitch_ret = StitchDataFollowIndices(data_unit, input, allowance, buf);
  if (stitch_ret != SUCCESS) {
    GELOGW("Stitch data follow index failed.");
    return NOT_CHANGED;
  }

  output_ptr->MutableTensorDesc().SetDataType(data_type);
  output_ptr->MutableTensorDesc().SetShape(merged_shape);
  Status ret = output_ptr->SetData(buf.get(), allowance);
  if (ret != GRAPH_SUCCESS) {
    GELOGW("set data failed");
    return NOT_CHANGED;
  }
  return SUCCESS;
}

Status DynamicStitchKernel::StitchDataFollowIndices(int64_t data_unit, const vector<ConstGeTensorPtr> &input,
                                                    int64_t allowance, std::unique_ptr<uint8_t[]> &buf) {
  // Safety note: index [1~2*n_] for input is valid, and all input is not null, validated in ValidateParams
  int64_t dst_offset = 0;
  int64_t src_offset = 0;
  std::set<int32_t> indices_set;
  for (int i = 0; i < n_; i++) {
    GeShape indices_shape = input[i]->GetTensorDesc().GetShape();
    size_t indices_dim_num = indices_shape.GetDimNum();
    // skip null indices tensor
    if (indices_dim_num == kNullTensorDimNum && indices_shape.GetDim(0) == kNullTensorDimValue) {
      GELOGD("Input indices[%d] has null tensor, skip it.", i);
      continue;
    }
    auto indices_shape_size = indices_shape.GetShapeSize();
    // to normalize logic, assume scalar as vector with shape of [1].
    indices_shape_size = (indices_shape_size == 0) ? 1 : indices_shape_size;
    // all index for input is less than size of input
    const int32_t *input_indices = reinterpret_cast<const int32_t *>(input[i]->GetData().data());
    const uint8_t *input_data = input[i + n_]->GetData().data();
    for (int64_t j = 0; j < indices_shape_size; j++) {
      // if index repeated, need new data replace old data , so give more allowance
      if (indices_set.find(input_indices[j]) != indices_set.end()) {
        if (ge::CheckInt64AddOverflow(input_indices[j], data_unit) != SUCCESS) {
          GELOGW("Check int64 mul overflow failed. Indices is %d, data_unit is %ld.", input_indices[j], data_unit);
          return NOT_CHANGED;
        }
        allowance += data_unit;
      }
      indices_set.insert(input_indices[j]);
      if (!CheckInt64MulOverflow(input_indices[j], data_unit)) {
        GELOGW("Check int64 mul overflow failed. Indices is %d, data_unit is %ld.", input_indices[j], data_unit);
        return NOT_CHANGED;
      }
      dst_offset = input_indices[j] * data_unit;
      src_offset = j * data_unit;
      auto protected_size =
          allowance < static_cast<int64_t>(SECUREC_MEM_MAX_LEN) ? allowance : static_cast<int64_t>(SECUREC_MEM_MAX_LEN);
      auto ret = memcpy_s(buf.get() + dst_offset, protected_size, input_data + src_offset, data_unit);
      if (ret != EOK) {
        GELOGW("Memory copy failed.");
        return NOT_CHANGED;
      }
      allowance -= data_unit;
    }
  }
  return SUCCESS;
}

REGISTER_KERNEL(DYNAMICSTITCH, DynamicStitchKernel);
}  // namespace ge
