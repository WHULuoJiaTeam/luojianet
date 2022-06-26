/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "tools/converter/parser/tflite/tflite_arithmetic_parser.h"
#include <vector>
#include <memory>
#include "ops/abs.h"
#include "ops/cos.h"
#include "ops/fusion/add_fusion.h"
#include "ops/fusion/mul_fusion.h"
#include "ops/fusion/div_fusion.h"
#include "ops/fusion/sub_fusion.h"
#include "ops/fusion/exp_fusion.h"
#include "ops/fusion/pow_fusion.h"
#include "ops/squared_difference.h"
#include "ops/square.h"
#include "ops/sqrt.h"
#include "ops/rsqrt.h"
#include "ops/sin.h"
#include "ops/log.h"
#include "ops/round.h"
#include "ops/neg.h"
#include "ops/maximum.h"
#include "ops/minimum.h"
#include "ops/floor.h"
#include "ops/floor_div.h"
#include "ops/floor_mod.h"
#include "ops/ceil.h"
#include "ops/equal.h"
#include "ops/greater.h"
#include "ops/greater_equal.h"
#include "ops/less.h"
#include "ops/less_equal.h"
#include "ops/not_equal.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TfliteAddParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                     const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                     const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::AddFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  const auto &tflite_attr = tflite_op->builtin_options.AsAddOptions();
  MS_CHECK_TRUE_MSG(tflite_attr != nullptr, nullptr, "get AddFusion attr failed");
  prim->set_activation_type(GetActivationFunctionType(tflite_attr->fused_activation_function));

  return prim->GetPrim();
}

PrimitiveCPtr TfliteMulParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                     const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                     const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::MulFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsMulOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get MulFusion attr failed";
    return nullptr;
  }
  prim->set_activation_type(GetActivationFunctionType(tflite_attr->fused_activation_function));

  return prim->GetPrim();
}

PrimitiveCPtr TfliteDivParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                     const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                     const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::DivFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsDivOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get DivFusion attr failed";
    return nullptr;
  }
  prim->set_activation_type(GetActivationFunctionType(tflite_attr->fused_activation_function));

  return prim->GetPrim();
}

PrimitiveCPtr TfliteSubParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                     const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                     const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::SubFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  const auto &tflite_attr = tflite_op->builtin_options.AsSubOptions();
  if (tflite_attr == nullptr) {
    MS_LOG(ERROR) << "get SubFusion attr failed";
    return nullptr;
  }
  prim->set_activation_type(GetActivationFunctionType(tflite_attr->fused_activation_function));

  return prim->GetPrim();
}

PrimitiveCPtr TfliteFloorDivParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::FloorDiv>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteFloorModParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::FloorMod>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TflitePowParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                     const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                     const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::PowFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  prim->set_scale(1.0);
  prim->set_shift(0.0);

  return prim->GetPrim();
}

PrimitiveCPtr TfliteSquaredDifferenceParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                   const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                                   const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::SquaredDifference>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteMaximumParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                         const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                         const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Maximum>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteMinimumParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                         const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                         const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Minimum>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteAbsParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                     const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                     const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Abs>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteCosParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                     const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                     const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Cos>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteFloorParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                       const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                       const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Floor>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteExpParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                     const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                     const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::ExpFusion>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  prim->set_base(-1.0);
  prim->set_scale(1.0);
  prim->set_shift(0.0);

  return prim->GetPrim();
}

PrimitiveCPtr TfliteCeilParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                      const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                      const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Ceil>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteLogParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                     const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                     const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Log>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteRoundParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                       const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                       const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Round>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteSqrtParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                      const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                      const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Sqrt>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteRsqrtParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                       const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                       const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Rsqrt>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteSquareParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Square>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteSinParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                     const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                     const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Sin>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteNegParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                     const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                     const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Neg>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteEqualParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                       const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                       const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Equal>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteNotEqualParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                          const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                          const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::NotEqual>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteGreaterParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                         const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                         const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Greater>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteGreaterEqualParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                              const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                              const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::GreaterEqual>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteLessParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                      const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                      const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::Less>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteLessEqualParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                           const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                           const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::LessEqual>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

TfliteNodeRegister g_tfliteAddParser(tflite::BuiltinOperator_ADD, new TfliteAddParser());
TfliteNodeRegister g_tfliteSubParser(tflite::BuiltinOperator_SUB, new TfliteSubParser());
TfliteNodeRegister g_TfliteMulParser(tflite::BuiltinOperator_MUL, new TfliteMulParser());
TfliteNodeRegister g_TfliteDivParser(tflite::BuiltinOperator_DIV, new TfliteDivParser());
TfliteNodeRegister g_tfliteFloorDivParser(tflite::BuiltinOperator_FLOOR_DIV, new TfliteFloorDivParser());
TfliteNodeRegister g_tfliteFloorModParser(tflite::BuiltinOperator_FLOOR_MOD, new TfliteFloorModParser());
TfliteNodeRegister g_TflitePowParser(tflite::BuiltinOperator_POW, new TflitePowParser());
TfliteNodeRegister g_tfliteSquaredDifferenceParser(tflite::BuiltinOperator_SQUARED_DIFFERENCE,
                                                   new TfliteSquaredDifferenceParser());
TfliteNodeRegister g_TfliteMaximumParser(tflite::BuiltinOperator_MAXIMUM, new TfliteMaximumParser());
TfliteNodeRegister g_TfliteMinimumParser(tflite::BuiltinOperator_MINIMUM, new TfliteMinimumParser());
TfliteNodeRegister g_TfliteAbsParser(tflite::BuiltinOperator_ABS, new TfliteAbsParser());
TfliteNodeRegister g_TfliteExpParser(tflite::BuiltinOperator_EXP, new TfliteExpParser());
TfliteNodeRegister g_TfliteSqrtParser(tflite::BuiltinOperator_SQRT, new TfliteSqrtParser());
TfliteNodeRegister g_tfliteRsqrtParser(tflite::BuiltinOperator_RSQRT, new TfliteRsqrtParser());
TfliteNodeRegister g_TfliteSquareParser(tflite::BuiltinOperator_SQUARE, new TfliteSquareParser());
TfliteNodeRegister g_TfliteSinParser(tflite::BuiltinOperator_SIN, new TfliteSinParser());
TfliteNodeRegister g_TfliteCosParser(tflite::BuiltinOperator_COS, new TfliteCosParser());
TfliteNodeRegister g_TfliteLogParser(tflite::BuiltinOperator_LOG, new TfliteLogParser());
TfliteNodeRegister g_tfliteRoundParser(tflite::BuiltinOperator_ROUND, new TfliteRoundParser());
TfliteNodeRegister g_TfliteCeilParser(tflite::BuiltinOperator_CEIL, new TfliteCeilParser());
TfliteNodeRegister g_tfliteFloorParser(tflite::BuiltinOperator_FLOOR, new TfliteFloorParser());
TfliteNodeRegister g_tfliteNegParser(tflite::BuiltinOperator_NEG, new TfliteNegParser());
TfliteNodeRegister g_tfliteEqualParser(tflite::BuiltinOperator_EQUAL, new TfliteEqualParser());
TfliteNodeRegister g_tfliteNotEqualParser(tflite::BuiltinOperator_NOT_EQUAL, new TfliteNotEqualParser());
TfliteNodeRegister g_tfliteGreaterEParser(tflite::BuiltinOperator_GREATER, new TfliteGreaterParser());
TfliteNodeRegister g_tfliteGreaterEqualParser(tflite::BuiltinOperator_GREATER_EQUAL, new TfliteGreaterEqualParser());
TfliteNodeRegister g_tfliteLessParser(tflite::BuiltinOperator_LESS, new TfliteLessParser());
TfliteNodeRegister g_tfliteLessEqualParser(tflite::BuiltinOperator_LESS_EQUAL, new TfliteLessEqualParser());
}  // namespace lite
}  // namespace mindspore
