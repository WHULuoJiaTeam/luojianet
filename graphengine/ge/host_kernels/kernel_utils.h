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

#ifndef GE_GRAPH_PASSES_FOLDING_KERNEL_KERNEL_UTILS_H_
#define GE_GRAPH_PASSES_FOLDING_KERNEL_KERNEL_UTILS_H_

#include <memory>
#include <vector>

#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"
#include "graph/compute_graph.h"

namespace ge {
class KernelUtils {
 public:
  KernelUtils() = delete;
  ~KernelUtils() = delete;
  static Status CheckDimensionNodeInfo(const NodePtr &node_ptr);
  static bool CheckFormatSupported(const NodePtr &node_ptr);
  static bool CheckSizeForTransOp(const ConstGeTensorPtr &const_weight_ptr, const OpDescPtr &op_desc_ptr);
  static bool IsUnknownShape(const GeShape &shape);

  /**
   * Generating a sequence of numbers
   * @param [in] data_num the num of generate
   * @param [in] value the value to write to buffer
   * @param [out] output the tensor for save sequence of numbers
   * @author
   */
  template<typename T>
  static Status GenData(const int64_t data_num, const T value, const GeTensorPtr &output) {
    if (data_num > 0) {
      if (!CheckInt64MulOverflow(data_num, static_cast<int64_t>(sizeof(T)))) {
        GELOGE(PARAM_INVALID, "Int64MulOverflow, data_num(%ld) type_len(%zu)", data_num, sizeof(T));
        return PARAM_INVALID;
      }

      std::unique_ptr<T[]> buf(new (std::nothrow) T[data_num]());
      if (buf == nullptr) {
        GELOGE(MEMALLOC_FAILED, "new sizeof(T) * data_num(%ld) memory failed", sizeof(T) * data_num);
        return MEMALLOC_FAILED;
      }

      for (int64_t i = 0; i < data_num; ++i) {
        buf[i] = value;
      }
      Status ret = output->SetData(reinterpret_cast<uint8_t *>(buf.get()), data_num * sizeof(T));
      if (ret != SUCCESS) {
        GELOGE(ret, " buf must not be null.");
        return ret;
      }
    }

    return SUCCESS;
  }

  /**
  * Calculate dimension
  * @param [in] dims save the tensor of the dimension
  * @param [in] vec_dim results of each dimension
  * @param [out] data_num total size of data
  * @author
  */
  template <typename T>
  static Status CalcDims(const ConstGeTensorPtr dims, std::vector<int64_t> &vec_dim, int64_t &data_num) {
    data_num = 1;
    int32_t size = dims->GetData().size() / sizeof(T);

    for (int32_t i = 0; i < size; i++) {
      T dim = *(reinterpret_cast<const T *>(dims->GetData().data()) + i);
      if (dim < 0) {
        GELOGE(PARAM_INVALID, "input dim(%d) is negative(%ld)", i, static_cast<int64_t>(dim));
        return PARAM_INVALID;
      }
      if (dim == 0) {
        GELOGI("input dim(%d) is zero", i);
        data_num = 0;
        vec_dim.clear();
        break;
      }
      if (!CheckInt64MulOverflow(data_num, dim)) {
        GELOGE(PARAM_INVALID, "Int64MulOverflow, data_num(%ld) dim(%ld)", data_num, static_cast<int64_t>(dim));
        return PARAM_INVALID;
      }

      data_num *= dim;
      vec_dim.push_back(dim);
    }

    return SUCCESS;
  }
};
}  // namespace ge

#endif  // GE_GRAPH_PASSES_FOLDING_KERNEL_KERNEL_UTILS_H_
