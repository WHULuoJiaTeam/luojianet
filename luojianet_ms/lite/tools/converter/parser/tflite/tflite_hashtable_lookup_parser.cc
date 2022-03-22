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

#include "tools/converter/parser/tflite/tflite_hashtable_lookup_parser.h"
#include <vector>
#include <memory>
#include "ops/hashtable_lookup.h"
#include "nnacl/op_base.h"

namespace luojianet_ms {
namespace lite {
ops::PrimitiveC *TfliteHashtableLookupParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                                    const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                                    const std::unique_ptr<tflite::ModelT> &tflite_model) {
  auto prim = std::make_unique<ops::HashtableLookup>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim.release();
}

TfliteNodeRegister g_tfliteHashtableLookupParser(tflite::BuiltinOperator_HASHTABLE_LOOKUP,
                                                 new TfliteHashtableLookupParser());
}  // namespace lite
}  // namespace luojianet_ms
