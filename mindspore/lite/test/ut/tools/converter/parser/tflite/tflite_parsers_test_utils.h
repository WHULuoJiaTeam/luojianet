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

#ifndef MINDSPORE_LITE_TEST_UT_TOOLS_CONVERTER_PARSER_TFLITE_TFLITE_PARSERS_TEST_H_
#define MINDSPORE_LITE_TEST_UT_TOOLS_CONVERTER_PARSER_TFLITE_TFLITE_PARSERS_TEST_H_

#include <string>
#include "common/common_test.h"
#include "schema/inner/model_generated.h"

namespace mindspore {
class TestTfliteParser : public CommonTest {
 public:
  TestTfliteParser() = default;
  void TearDown() override;
  schema::MetaGraphT *LoadAndConvert(const std::string &model_path, const std::string &weight_path = "");
  schema::MetaGraphT *meta_graph = nullptr;
};

}  // namespace mindspore

#endif  // MINDSPORE_LITE_TEST_UT_TOOLS_CONVERTER_PARSER_TFLITE_TFLITE_PARSERS_TEST_H_
