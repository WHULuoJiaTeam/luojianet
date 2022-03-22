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

#include "ge_runtime/output.h"
#include "framework/common/ge_inner_error_codes.h"
#include "framework/common/util.h"
#include "framework/common/debug/ge_log.h"

namespace ge {
namespace model_runner {
Output::Output(const OpInfoPtr &op_info, const std::shared_ptr<DavinciModel> &model)
    : model_(model), op_info_(op_info), input_num_(0) {}

Output::~Output() {}

bool Output::Init() {
  if (op_info_ == nullptr || model_ == nullptr) {
    GELOGE(INTERNAL_ERROR, "The op_desc_ or model_ is nullptr.");
    return false;
  }

  input_num_ = op_info_->input_tensors.size();
  v_input_size_.clear();
  v_input_data_addr_.clear();

  auto input_vector = op_info_->input_addrs;
  if (input_num_ != input_vector.size()) {
    GELOGE(INTERNAL_ERROR, "The input desc size: %zu !=  input addr size: %zu.", input_num_, input_vector.size());
    return false;
  }

  for (size_t i = 0; i < input_num_; i++) {
    uint32_t tensorSize = 0;
    const auto &input_info = op_info_->input_tensors.at(i);
    tensorSize = input_info.size;
    v_input_size_.push_back(tensorSize);
    v_input_data_addr_.push_back(reinterpret_cast<uint8_t *>(input_vector.at(i)));
  }

  GELOGI("Init output:%zu, %zu, %zu", input_num_, v_input_size_.size(), v_input_data_addr_.size());

  return true;
}

///
/// @ingroup domi_ome
/// @brief Copy Op Output to user space.
/// @brief when model running, Add one DataOp as input node, Add one Output Op as output node.
/// @return Status
///
bool Output::CopyRslt(OutputData *rslt, uint32_t data_begin, uint32_t &data_index, bool support_mem_share) {
  if (rslt == nullptr) {
    GELOGE(FAILED, "OutputData is null.");
    return false;
  }
  uint32_t data_count = 0;
  if (v_input_size_.empty() || v_input_data_addr_.empty()) {
    GELOGE(INTERNAL_ERROR, "v_output_size_ or v_output_data_addr_ is empty!");
    return false;
  }

  for (size_t i = 0; i < input_num_; i++) {
    DataBuffer data_buf = rslt->blobs[data_begin + data_count];
    bool ret = SetDataBuf(data_buf, data_begin, data_count, i, support_mem_share);
    if (!ret) {
      GELOGE(FAILED, "Copy data to host failed. index: %lu, addr: %p", i, v_input_data_addr_[i]);
      return ret;
    }
    data_index = data_begin + data_count;
  }

  return true;
}

bool Output::SetDataBuf(DataBuffer &data_buf, uint32_t data_begin, uint32_t &data_count, size_t i,
                        bool support_mem_share) {
  return true;
}

}  // namespace model_runner
}  // namespace ge
