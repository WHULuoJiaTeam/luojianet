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

#ifndef GE_GRAPH_COMMON_BCAST_H_
#define GE_GRAPH_COMMON_BCAST_H_

#include <stdint.h>
#include <functional>
#include <vector>

#include "common/debug/log.h"
#include "common/types.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/ge_inner_error_codes.h"
#include "graph/attr_value.h"
#include "graph/ge_tensor.h"
#include "graph/utils/tensor_adapter.h"

namespace ge {
static const size_t kMinDimNum = 2;
class BCast {
 public:
  ///
  /// @ingroup domi_calibration
  /// @brief define kVecInt
  ///
  typedef std::vector<int64_t> kVecInt;

  ///
  /// @ingroup domi_calibration
  /// @brief constructor
  ///
  BCast() {}
  ///
  /// @ingroup domi_calibration
  /// @brief destructor
  ///
  ~BCast() {}

  ///
  /// @ingroup domi_calibration
  /// @brief Not optimize intermediate shapes
  /// @decrease dims, more efficient, set by user
  /// @param [in] x   first Tensor dim
  /// @param [in] y   second Tensor dim
  /// @return     SUCCESS broadcast message successfully generated
  /// @return     other   broadcast message failed to generate
  ///
  ge::Status GenerateBcastInfo(const kVecInt &x, const kVecInt &y);

  ///
  /// @ingroup domi_calibration
  /// @brief get x_reshape
  ///
  const kVecInt &GetXReshape() const { return x_reshape_; }

  ///
  /// @ingroup domi_calibration
  /// @brief get x_bcast
  ///
  const kVecInt &GetXBcast() const { return x_bcast_; }

  ///
  /// @ingroup domi_calibration
  /// @brief get y_reshape
  ///
  const kVecInt &GetYReshape() const { return y_reshape_; }
  ///
  /// @ingroup domi_calibration
  /// @brief get y_bcast
  ///
  const kVecInt &GetYBcast() const { return y_bcast_; }
  ///
  /// @ingroup domi_calibration
  /// @brief get result_shape
  ///
  const kVecInt &GetResultShape() const { return result_; }

  ///
  /// @ingroup domi_calibration
  /// @brief get result_shape
  ///
  const kVecInt &GetOutputShape() const { return output_; }
  const kVecInt &GetGradXReduceIdx() const { return grad_x_reduce_idx_; }
  const kVecInt &GetGradYReduceIdx() const { return grad_y_reduce_idx_; }

  ///
  /// @ingroup domi_calibration
  /// @brief convert TensorDescriptor to kVecInt
  /// @param [in] shape   Tensor descriptor
  /// @return     kVecInt     dim info
  ///
  static kVecInt TransShapeToDimVec(const GeTensorDesc &shape);

  void BCastIndexes(kVecInt &x_indexes, kVecInt &y_indexes);
  template <typename InT, typename OutT>
  Status BCastCompute(const std::vector<ConstGeTensorPtr> &input, std::vector<OutT> &v_output,
                      const std::function<OutT(InT const &, InT const &)> &func) {
    Status ret;
    if (func == nullptr) {
      REPORT_INNER_ERROR("E19999", "Check param func nullptr");
      GELOGE(domi::PARAM_INVALID, "Param func is null");
      return domi::PARAM_INVALID;
    }
    // Min input num is 2
    if (input.size() < kMinDimNum) {
      REPORT_INNER_ERROR("E19999", "Param input.size():%zu < %zu, check invalid",
                         input.size(), kMinDimNum);
      GELOGE(domi::PARAM_INVALID, "Input size is smaller than two.");
      return domi::PARAM_INVALID;
    }
    // Only broadcast shape
    ret =
      GenerateBcastInfo(TransShapeToDimVec(input[0]->GetTensorDesc()), TransShapeToDimVec(input[1]->GetTensorDesc()));
    if (ret != domi::SUCCESS) {
      GELOGE(ret, "Greater broadcasting failed.");
      return ret;
    }

    kVecInt x_indexes;
    kVecInt y_indexes;
    BCastIndexes(x_indexes, y_indexes);

    const void *x1_data = input[0]->GetData().data();
    const void *x2_data = input[1]->GetData().data();

    for (size_t i = 0; i < x_indexes.size(); i++) {
      int64_t x_index = x_indexes[i];
      int64_t y_index = y_indexes[i];
      auto value = func((*(reinterpret_cast<const InT *>(x1_data) + x_index)),
                        (*(reinterpret_cast<const InT *>(x2_data) + y_index)));
      v_output.push_back(value);
    }

    return domi::SUCCESS;
  }

  template <typename InT, typename OutT>
  Status BCastComputeCheck(const std::vector<ConstGeTensorPtr> &input, std::vector<OutT> &v_output,
                           const std::function<OutT(InT const &, InT const &, DataType &type, Status &)> &func) {
    if (func == nullptr) {
      REPORT_INNER_ERROR("E19999", "Check param func nullptr");
      GELOGE(PARAM_INVALID, "Param func is null");
      return PARAM_INVALID;
    }
    // Min input num is 2
    if (input.size() < kMinDimNum) {
      REPORT_INNER_ERROR("E19999", "Param input.size():%zu < %zu, check invalid",
                         input.size(), kMinDimNum);
      GELOGE(PARAM_INVALID, "Input size is smaller than two.");
      return PARAM_INVALID;
    }
    // Only broadcast shape
    Status ret =
      GenerateBcastInfo(TransShapeToDimVec(input[0]->GetTensorDesc()), TransShapeToDimVec(input[1]->GetTensorDesc()));
    if (ret != SUCCESS) {
      GELOGE(ret, "Greater broadcasting failed.");
      return ret;
    }

    DataType data_type = input[0]->GetTensorDesc().GetDataType();
    kVecInt x_indexes;
    kVecInt y_indexes;
    BCastIndexes(x_indexes, y_indexes);

    const void *x1_data = input[0]->GetData().data();
    const void *x2_data = input[1]->GetData().data();

    for (size_t i = 0; i < x_indexes.size(); i++) {
      int64_t x_index = x_indexes[i];
      int64_t y_index = y_indexes[i];
      auto value = func((*(reinterpret_cast<const InT *>(x1_data) + x_index)),
                        (*(reinterpret_cast<const InT *>(x2_data) + y_index)), data_type, ret);
      if (ret != SUCCESS) {
        REPORT_INNER_ERROR("E19999", "BCastComputeCheck func execute failed, datatype is %d.", data_type);
        GELOGE(ret, "BCastComputeCheck func execute failed, datatype is %d.", data_type);
        return ret;
      }
      v_output.push_back(value);
    }

    return SUCCESS;
  }

 private:
  ///
  /// @ingroup domi_calibration
  /// @brief reverse elements in kVecInt
  /// @param [in] shape   dim info
  /// @return null
  ///
  static void Reverse(kVecInt &shape);

  ///
  /// @ingroup domi_calibration
  /// @brief two Tensor with different shape, set broadcast info
  /// @param [in] x   first input Tensor dim info
  /// @param [in] y   second input Tensor dim info
  /// @return null
  ///
  ge::Status SetShapeDifferentInfo(const kVecInt &x, const kVecInt &y);
  ///
  /// @ingroup domi_calibration
  /// @brief extend Tensor dim
  /// @param [in] x   first input Tensor dim info
  /// @param [in] y   second input Tensor dim info
  /// @return null
  ///
  void ExtendTensorDim(kVecInt &x, kVecInt &y);
  ///
  /// @ingroup domi_calibration
  /// @brief reverse all intermediate shape params
  /// @param [in] void
  /// @return null
  ///
  void ReverseAllIntermediateShapes();

  kVecInt x_reshape_;
  kVecInt x_bcast_;
  kVecInt y_reshape_;
  kVecInt y_bcast_;
  kVecInt result_;
  kVecInt output_;
  kVecInt grad_x_reduce_idx_;
  kVecInt grad_y_reduce_idx_;
};
}  // namespace ge

#endif  // GE_GRAPH_COMMON_BCAST_H_
