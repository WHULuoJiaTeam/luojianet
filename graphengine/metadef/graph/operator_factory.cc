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

#include "graph/operator_factory_impl.h"
#include "debug/ge_log.h"

namespace ge {
Operator OperatorFactory::CreateOperator(const std::string &operator_name, const std::string &operator_type) {
  return OperatorFactoryImpl::CreateOperator(operator_name, operator_type);
}

Operator OperatorFactory::CreateOperator(const char *operator_name, const char *operator_type) {
  if ((operator_name == nullptr) || (operator_type == nullptr)) {
    REPORT_INNER_ERROR("E19999", "Create Operator input parameter is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Create Operator input parameter is nullptr.");
    return Operator();
  }
  std::string op_name = operator_name;
  std::string op_type = operator_type;
  return OperatorFactoryImpl::CreateOperator(op_name, op_type);
}

graphStatus OperatorFactory::GetOpsTypeList(std::vector<std::string> &all_ops) {
  return OperatorFactoryImpl::GetOpsTypeList(all_ops);
}

graphStatus OperatorFactory::GetOpsTypeList(std::vector<AscendString> &all_ops) {
  std::vector<std::string> all_op_types;
  if (OperatorFactoryImpl::GetOpsTypeList(all_op_types) != GRAPH_SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Get ops type list failed.");
    GELOGE(GRAPH_FAILED, "[Get][OpsTypeList] failed.");
    return GRAPH_FAILED;
  }
  for (auto &op_type : all_op_types) {
    all_ops.emplace_back(op_type.c_str());
  }
  return GRAPH_SUCCESS;
}

bool OperatorFactory::IsExistOp(const std::string &operator_type) {
  return OperatorFactoryImpl::IsExistOp(operator_type);
}

bool OperatorFactory::IsExistOp(const char *operator_type) {
  if (operator_type == nullptr) {
    REPORT_INNER_ERROR("E19999", "Operator type is nullptr, check invalid.");
    GELOGE(GRAPH_FAILED, "[Check][Param] Operator type is nullptr.");
    return false;
  }
  std::string op_type = operator_type;
  return OperatorFactoryImpl::IsExistOp(op_type);
}

OperatorCreatorRegister::OperatorCreatorRegister(const std::string &operator_type, OpCreator const &op_creator) {
  (void)OperatorFactoryImpl::RegisterOperatorCreator(operator_type, op_creator);
}

OperatorCreatorRegister::OperatorCreatorRegister(const char *operator_type, OpCreatorV2 const &op_creator) {
  std::string op_type;
  if (operator_type != nullptr) {
    op_type = operator_type;
  }
  (void)OperatorFactoryImpl::RegisterOperatorCreator(op_type, op_creator);
}

InferShapeFuncRegister::InferShapeFuncRegister(const std::string &operator_type,
                                               const InferShapeFunc &infer_shape_func) {
  (void)OperatorFactoryImpl::RegisterInferShapeFunc(operator_type, infer_shape_func);
}

InferShapeFuncRegister::InferShapeFuncRegister(const char *operator_type,
                                               const InferShapeFunc &infer_shape_func) {
  std::string op_type;
  if (operator_type != nullptr) {
    op_type = operator_type;
  }
  (void)OperatorFactoryImpl::RegisterInferShapeFunc(op_type, infer_shape_func);
}

InferFormatFuncRegister::InferFormatFuncRegister(const std::string &operator_type,
                                                 const InferFormatFunc &infer_format_func) {
  (void)OperatorFactoryImpl::RegisterInferFormatFunc(operator_type, infer_format_func);
}

InferFormatFuncRegister::InferFormatFuncRegister(const char *operator_type,
                                                 const InferFormatFunc &infer_format_func) {
  std::string op_type;
  if (operator_type != nullptr) {
    op_type = operator_type;
  }
  (void)OperatorFactoryImpl::RegisterInferFormatFunc(op_type, infer_format_func);
}

InferValueRangeFuncRegister::InferValueRangeFuncRegister(const char *operator_type,
                                                         WHEN_CALL when_call,
                                                         const InferValueRangeFunc &infer_value_range_func) {
  std::string op_type;
  if (operator_type != nullptr) {
    op_type = operator_type;
  }
  (void)OperatorFactoryImpl::RegisterInferValueRangeFunc(op_type, when_call, false, infer_value_range_func);
}

InferValueRangeFuncRegister::InferValueRangeFuncRegister(const char *operator_type) {
  std::string op_type;
  if (operator_type != nullptr) {
    op_type = operator_type;
  }
  (void)OperatorFactoryImpl::RegisterInferValueRangeFunc(op_type);
}

VerifyFuncRegister::VerifyFuncRegister(const std::string &operator_type, const VerifyFunc &verify_func) {
  (void)OperatorFactoryImpl::RegisterVerifyFunc(operator_type, verify_func);
}

VerifyFuncRegister::VerifyFuncRegister(const char *operator_type, const VerifyFunc &verify_func) {
  std::string op_type;
  if (operator_type != nullptr) {
    op_type = operator_type;
  }
  (void)OperatorFactoryImpl::RegisterVerifyFunc(op_type, verify_func);
}
}  // namespace ge
