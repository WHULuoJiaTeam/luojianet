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

#ifndef INC_EXTERNAL_GRAPH_OPERATOR_FACTORY_H_
#define INC_EXTERNAL_GRAPH_OPERATOR_FACTORY_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "./operator.h"
#include "./ge_error_codes.h"
#include "./ascend_string.h"
#include "./types.h"

namespace ge {
using OpCreator = std::function<Operator(const std::string &)>;
using OpCreatorV2 = std::function<Operator(const AscendString &)>;
using InferShapeFunc = std::function<graphStatus(Operator &)>;
using InferFormatFunc = std::function<graphStatus(Operator &)>;
using InferValueRangeFunc = std::function<graphStatus(Operator &)>;
using VerifyFunc = std::function<graphStatus(Operator &)>;

enum WHEN_CALL {
  INPUT_IS_DYNAMIC = 0,
  INPUT_HAS_VALUE_RANGE = 1
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OperatorFactory {
 public:
  ATTRIBUTED_DEPRECATED(static Operator CreateOperator(const char_t *, const char_t *))
  static Operator CreateOperator(const std::string &operator_name, const std::string &operator_type);

  static Operator CreateOperator(const char_t *operator_name, const char_t *operator_type);

  ATTRIBUTED_DEPRECATED(graphStatus GetOpsTypeList(std::vector<AscendString> &))
  static graphStatus GetOpsTypeList(std::vector<std::string> &all_ops);

  static graphStatus GetOpsTypeList(std::vector<AscendString> &all_ops);

  ATTRIBUTED_DEPRECATED(bool IsExistOp(const char_t *))
  static bool IsExistOp(const string &operator_type);

  static bool IsExistOp(const char_t *operator_type);
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY OperatorCreatorRegister {
 public:
  ATTRIBUTED_DEPRECATED(OperatorCreatorRegister(const char_t *, OpCreatorV2 const &))
  OperatorCreatorRegister(const std::string &operator_type, OpCreator const &op_creator);
  OperatorCreatorRegister(const char_t *operator_type, OpCreatorV2 const &op_creator);
  ~OperatorCreatorRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferShapeFuncRegister {
 public:
  ATTRIBUTED_DEPRECATED(InferShapeFuncRegister(const char_t *, const InferShapeFunc &))
  InferShapeFuncRegister(const std::string &operator_type, const InferShapeFunc &infer_shape_func);
  InferShapeFuncRegister(const char_t *operator_type, const InferShapeFunc &infer_shape_func);
  ~InferShapeFuncRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferFormatFuncRegister {
 public:
  ATTRIBUTED_DEPRECATED(InferFormatFuncRegister(const char_t *, const InferFormatFunc &))
  InferFormatFuncRegister(const std::string &operator_type, const InferFormatFunc &infer_format_func);
  InferFormatFuncRegister(const char_t *operator_type, const InferFormatFunc &infer_format_func);
  ~InferFormatFuncRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY InferValueRangeFuncRegister {
 public:
  InferValueRangeFuncRegister(const char_t *operator_type, WHEN_CALL when_call,
                              const InferValueRangeFunc &infer_value_range_func);
  InferValueRangeFuncRegister(const char_t *operator_type);
  ~InferValueRangeFuncRegister() = default;
};

class GE_FUNC_DEV_VISIBILITY GE_FUNC_HOST_VISIBILITY VerifyFuncRegister {
 public:
  ATTRIBUTED_DEPRECATED(VerifyFuncRegister(const char_t *, const VerifyFunc &))
  VerifyFuncRegister(const std::string &operator_type, const VerifyFunc &verify_func);
  VerifyFuncRegister(const char_t *operator_type, const VerifyFunc &verify_func);
  ~VerifyFuncRegister() = default;
};
}  // namespace ge

#endif  // INC_EXTERNAL_GRAPH_OPERATOR_FACTORY_H_
