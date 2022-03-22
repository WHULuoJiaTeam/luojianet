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
#ifndef LUOJIANET_MS_CCSRC_MINDDATA_DATASET_KERNELS_PLUGIN_OP_H_
#define LUOJIANET_MS_CCSRC_MINDDATA_DATASET_KERNELS_PLUGIN_OP_H_
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/plugin/include/shared_include.h"

#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/core/tensor_row.h"
#include "minddata/dataset/util/status.h"

namespace luojianet_ms {
namespace dataset {

// a generalized plugin for TensorOp
class PluginOp : public TensorOp {
 public:
  PluginOp(const std::string &lib_path, const std::string &func_name, const std::string &user_args);

  ~PluginOp() = default;

  Status Compute(const TensorRow &input, TensorRow *output) override;

  Status Init();  // load plugin module

  std::string Name() const override { return kPluginOp; }

  // helper function to convert between plugin Tensor and MindData Tensor
  static Status PluginToTensorRow(const std::vector<plugin::Tensor> &, TensorRow *);

  static Status TensorRowToPlugin(const TensorRow &, std::vector<plugin::Tensor> *);

 private:
  Status init_code_;
  plugin::TensorOp *plugin_op_;
  std::string lib_path_;
  std::string func_name_;
  std::string user_args_;
};

}  // namespace dataset
}  // namespace luojianet_ms

#endif  // LUOJIANET_MS_CCSRC_MINDDATA_DATASET_KERNELS_PLUGIN_OP_H_
