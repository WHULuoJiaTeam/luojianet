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

#ifndef INC_GRAPH_UTILS_TENSOR_UTILS_H_
#define INC_GRAPH_UTILS_TENSOR_UTILS_H_

#include <vector>
#include "graph/def_types.h"
#include "graph/ge_error_codes.h"
#include "graph/ge_tensor.h"

namespace ge {
class TensorUtils {
 public:
  static GeTensor CreateShareTensor(const GeTensor &other);
  static GeTensor CreateShareTensor(const GeTensorDesc &tensor_desc,
                                    std::shared_ptr<AlignedPtr> aligned_ptr,
                                    const size_t size);
  static void ShareTensor(const GeTensor &from, GeTensor &to);
  static TensorData CreateShareTensorData(const TensorData &other);
  static void ShareTensorData(const TensorData &from, TensorData &to);
  static void ShareAlignedPtr(std::shared_ptr<AlignedPtr> ptr, const size_t size, TensorData &to);
  static void ShareAlignedPtr(std::shared_ptr<AlignedPtr> ptr, const size_t size, GeTensor &to);
  static void CopyTensor(const GeTensor &from, GeTensor &to);
  static ge::graphStatus GetSize(const GeTensorDesc &tensor_desc, int64_t &size);
  static void SetSize(GeTensorDesc &tensor_desc, const int64_t size);
  static uint32_t GetWeightSize(const ConstGeTensorPtr &tensor_ptr);
  static uint32_t GetWeightSize(const GeTensor &tensor);
  static uint32_t GetWeightSize(const GeTensorDesc &tensor_desc);
  static uint8_t *GetWeightAddr(const ConstGeTensorPtr &tensor_ptr, uint8_t * const base);
  static uint8_t *GetWeightAddr(const GeTensor &tensor, uint8_t * const base);
  static void SetWeightSize(GeTensorDesc &tensor_desc, const uint32_t size);
  static ge::graphStatus GetReuseInput(const GeTensorDesc &tensor_desc, bool &flag);
  static void SetReuseInput(GeTensorDesc &tensor_desc, const bool flag);
  static ge::graphStatus GetOutputTensor(const GeTensorDesc &tensor_desc, bool &flag);
  static void SetOutputTensor(GeTensorDesc &tensor_desc, const bool flag);
  static graphStatus GetDeviceType(const GeTensorDesc &tensor_desc, DeviceType &type);
  static void SetDeviceType(GeTensorDesc &tensor_desc, const DeviceType type);
  static ge::graphStatus GetInputTensor(const GeTensorDesc &tensor_desc, bool &flag);
  static void SetInputTensor(GeTensorDesc &tensor_desc, bool flag);
  static ge::graphStatus GetRealDimCnt(const GeTensorDesc &tensor_desc, uint32_t &cnt);
  static void SetRealDimCnt(GeTensorDesc &tensor_desc, const uint32_t cnt);
  static ge::graphStatus GetReuseInputIndex(const GeTensorDesc &tensor_desc, uint32_t &idx);
  static void SetReuseInputIndex(GeTensorDesc &tensor_desc, const uint32_t idx);
  static ge::graphStatus GetDataOffset(const GeTensorDesc &tensor_desc, int64_t &offset);
  static void SetDataOffset(GeTensorDesc &tensor_desc, const int64_t offset);
  static ge::graphStatus GetRC(const GeTensorDesc &tensor_desc, uint32_t &rc);
  static void SetRC(GeTensorDesc &tensor_desc, const uint32_t rc);
  static bool IsOriginShapeInited(const GeTensorDesc &tensor_desc);

  ///
  /// calculate tensor mem size.
  /// @param shape tensor shape
  /// @param format tensor format
  /// @param data_type tensor data type
  /// @param mem_size -1 means unknown shape,other means mem size
  /// @return GRAPH_SUCCESS:success, other:failed
  ///
  static ge::graphStatus CalcTensorMemSize(const GeShape &shape, const Format format,
                                           const DataType data_type, int64_t &mem_size);
  static ge::graphStatus CalcTensorMemSizeForNoTiling(const GeTensorDesc &tensor,
                                                      const Format format,
                                                      const DataType data_type,
                                                      int64_t &mem_size);
  static ge::graphStatus GetTensorMemorySizeInBytes(const GeTensorDesc &desc_temp, int64_t &size_temp);
  static ge::graphStatus GetTensorSizeInBytes(const GeTensorDesc &desc_temp, int64_t &size_temp);
  static ge::graphStatus CheckShapeByShapeRange(const GeShape &shape,
                                                const std::vector<std::pair<int64_t, int64_t>> &shape_range);
};
}  // namespace ge
#endif  // INC_GRAPH_UTILS_TENSOR_UTILS_H_
