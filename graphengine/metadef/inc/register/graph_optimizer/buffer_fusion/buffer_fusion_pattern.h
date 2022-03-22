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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_PATTERN_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_PATTERN_H_
#include <map>
#include <string>
#include <vector>
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"

namespace fe {
static const int TBE_FUSION_OP_NUM_MAX = 5;
static const int TBE_PATTERN_NUM_MAX = 5;
static const int TBE_PATTERN_NUM_NONE = 0;
static const int TBE_PATTERN_NUM_DEFAULT = 1;
static const int TBE_OUTPUT_BRANCH_DEFAULT = 0;
static const int TBE_OUTPUT_BRANCH_SINGLE = 1;
static const int TBE_OUTPUT_BRANCH_MULTI = 2;
static const int TBE_PATTERN_GROUPID_INVALID = -1;

enum SkipStatus { DISABLED = 0, AVAILABLE = 1, SKIPPED = 2 };

enum ShapeTypeRule { IGNORE_SHAPE_TYPE = 0, ONLY_SUPPORT_STATIC, ONLY_SUPPORT_DYNAMIC };

struct BufferFusionOpDesc {
  std::string desc_name;                       // description name
  std::vector<std::string> types;             // description type
  std::vector<BufferFusionOpDesc *> inputs;   // all input op
  std::vector<BufferFusionOpDesc *> outputs;  // all output op
  int64_t out_branch_type;                      // out desc type, 1:single, 2: multi
  int64_t repeate_min;                         // opdesc min repeat num
  int64_t repeate_max;                         // opdesc max repeat num
  int64_t repeate_curr;                        // opdesc current repeat num
  bool match_status;
  bool not_pattern;
  int64_t group_id;  // record desc groupid, need one desc matched at least in
                    // the same group
  ShapeTypeRule shape_type_rule;
  bool ignore_input_num;
  bool ignore_output_num;
  // used for two connected op, first opdesc has optional multiple nodes and
  // ignore_output_num is true, second opdesc is same pattern type and
  // out_branch_type is TBE_OUTPUT_BRANCH_MULTI
  std::map<int64_t, SkipStatus> multi_output_skip_status;
};
using BufferFusionMapping = std::map<const BufferFusionOpDesc *, std::vector<ge::NodePtr>>;
using BufferFusionMappings = std::vector<BufferFusionMapping>;

class BufferFusionPattern {
 public:
  explicit BufferFusionPattern(std::string name = "", int64_t op_max_count = TBE_FUSION_OP_NUM_MAX);

  virtual ~BufferFusionPattern();

  BufferFusionPattern &AddOpDesc(const std::string &desc_name, const std::vector<std::string> &patterns,
                                 int64_t repeat_min = TBE_PATTERN_NUM_DEFAULT,
                                 int64_t repeat_max = TBE_PATTERN_NUM_DEFAULT,
                                 int64_t group_id = TBE_PATTERN_GROUPID_INVALID,
                                 ShapeTypeRule shape_type_rule = ONLY_SUPPORT_STATIC,
                                 bool not_pattern = false);

  BufferFusionPattern &SetOutputs(const std::string &desc_name, const std::vector<std::string> &patterns,
                                  int64_t relation = TBE_OUTPUT_BRANCH_SINGLE, bool ignore_input_num = false,
                                  bool ignore_output_num = false);

  BufferFusionPattern &SetHead(const std::vector<std::string> &op_patterns);

  std::string GetName();
  int64_t GetOpMaxCount();
  std::vector<BufferFusionOpDesc *> GetOpDescs();
  bool GetOutputs(BufferFusionOpDesc *op_desc, std::vector<BufferFusionOpDesc *> &outputs, bool ignore_repeat = false);
  std::vector<BufferFusionOpDesc *> GetHead();
  int64_t GetErrorCnt();
  void InitRepeatCurr(const BufferFusionPattern &pattern);

 private:
  BufferFusionOpDesc *GetOpDesc(const std::string &desc_name);
  void UpdateSkipStatus(BufferFusionOpDesc *op_desc);
  std::string name_;
  int64_t op_max_count_;
  std::vector<BufferFusionOpDesc *> ops_;
  std::map<std::string, BufferFusionOpDesc *> op_map_;
  std::vector<BufferFusionOpDesc *> head_;
  int64_t error_count_;
};
}  // namespace fe
#endif  // INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_PATTERN_H_
