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
#ifndef ACL_TOOLS_COMPILE_PARSER_H
#define ACL_TOOLS_COMPILE_PARSER_H

#include <vector>
#include <string>

#include <nlohmann/json.hpp>

#include "external/ge/ge_api_error_codes.h"
#include "external/graph/types.h"
#include "graph/ge_attr_value.h"
#include "graph/op_desc.h"

namespace ge {
struct SingleOpTensorDesc {
public:
  bool GetValidFlag() const { return is_valid_; }
  void SetValidFlag(bool is_valid) { is_valid_ = is_valid; }
public:
  std::string name;
  std::vector<int64_t> dims;
  std::vector<int64_t> ori_dims;
  std::vector<std::vector<int64_t>> dim_ranges;
  ge::Format format = ge::FORMAT_RESERVED;
  ge::Format ori_format = ge::FORMAT_RESERVED;
  ge::DataType type = ge::DT_UNDEFINED;
  std::string dynamic_input_name;
private:
  bool is_valid_ = true;
};

struct SingleOpAttr {
  std::string name;
  std::string type;
  ge::GeAttrValue value;
};

struct SingleOpDesc {
  std::string op;
  std::vector<SingleOpTensorDesc> input_desc;
  std::vector<SingleOpTensorDesc> output_desc;
  std::vector<SingleOpAttr> attrs;
  int32_t compile_flag = 0;
};

struct SingleOpBuildParam {
  ge::OpDescPtr op_desc;
  std::vector<ge::GeTensor> inputs;
  std::vector<ge::GeTensor> outputs;
  std::string file_name;
  int32_t compile_flag = 0;
};

void from_json(const nlohmann::json &json, SingleOpTensorDesc &desc);

void from_json(const nlohmann::json &json, SingleOpAttr &desc);

void from_json(const nlohmann::json &json, SingleOpDesc &desc);

class SingleOpParser {
 public:
  static Status ParseSingleOpList(const std::string &file, std::vector<SingleOpBuildParam> &op_list);

 private:
  static Status ReadJsonFile(const std::string &file, nlohmann::json &json_obj);
  static bool Validate(const SingleOpDesc &op_desc);
  static std::unique_ptr<OpDesc> CreateOpDesc(const std::string &op_type);
  static Status ConvertToBuildParam(int index, const SingleOpDesc &single_op_desc, SingleOpBuildParam &build_param);
  static Status UpdateDynamicTensorName(std::vector<SingleOpTensorDesc> &desc);
  static Status VerifyOpInputOutputSizeByIr(const OpDesc &current_op_desc);
  static Status SetShapeRange(const std::string &op_name,
                              const SingleOpTensorDesc &tensor_desc,
                              GeTensorDesc &ge_tensor_desc);
};
}  // namespace ge

#endif  // ACL_TOOLS_COMPILE_PARSER_H
