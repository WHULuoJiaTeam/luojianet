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

#ifndef H77F0BD09_6C00_4E45_8DED_38A676D6B20A
#define H77F0BD09_6C00_4E45_8DED_38A676D6B20A

#include <string>
#include "ge_graph_dsl/ge.h"
#include "graph/types.h"
#include "ge_graph_dsl/op_desc/op_type.h"

GE_NS_BEGIN

struct OpDescCfg {
  struct TensorCfg {
    TensorCfg(Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT, std::vector<int64_t> shape = {1, 1, 224, 224})
        : format_(format), data_type_(data_type), shape_(shape) {}
    Format format_;
    DataType data_type_;
    std::vector<int64_t> shape_;
  };

  OpDescCfg(const OpType &type, int in_cnt = 1, int out_cnt = 1, Format format = FORMAT_NCHW,
            DataType data_type = DT_FLOAT, std::vector<int64_t> shape = {1, 1, 224, 224})
      : type_(type), in_cnt_(in_cnt), out_cnt_(out_cnt), default_tensor_(format, data_type, shape) {}

 protected:
  OpType GetType() const { return type_; }
  OpType type_;
  int in_cnt_;
  int out_cnt_;
  TensorCfg default_tensor_;
};

GE_NS_END

#endif /* H77F0BD09_6C00_4E45_8DED_38A676D6B20A */
