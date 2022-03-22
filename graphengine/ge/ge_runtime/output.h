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

#ifndef GE_GE_RUNTIME_OUTPUT_H_
#define GE_GE_RUNTIME_OUTPUT_H_

#include <memory>
#include <vector>
#include "framework/ge_runtime/davinci_model.h"
#include "framework/common/ge_types.h"

namespace ge {
namespace model_runner {

class Output {
 public:
  Output(const OpInfoPtr &op_info, const std::shared_ptr<DavinciModel> &model);
  virtual ~Output();
  bool Init();

  bool CopyRslt(OutputData *rslt, uint32_t data_begin, uint32_t &data_index, bool support_mem_share);

  bool SetDataBuf(DataBuffer &data_buf, uint32_t data_begin, uint32_t &data_count, size_t i, bool support_mem_share);

  // Copy assignment operator and copy constructor are deleted
  Output &operator=(const Output &output) = delete;
  Output(const Output &output) = delete;

 protected:
  std::shared_ptr<DavinciModel> model_;
  OpInfoPtr op_info_;

  // Input descriptions
  size_t input_num_;
  vector<void *> v_input_data_addr_;  // Init as:buf_base + op_def_->input(i));
  vector<uint32_t> v_input_size_;
};
}  // namespace model_runner
}  // namespace ge
#endif  // GE_GE_RUNTIME_OUTPUT_H_
