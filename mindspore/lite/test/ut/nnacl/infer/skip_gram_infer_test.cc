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
#include "common/common_test.h"
#include "nnacl/infer/string/skip_gram_infer.h"

namespace mindspore {

class SkipGramInferTest : public mindspore::CommonTest {
 public:
  SkipGramInferTest() {}
};

TEST_F(SkipGramInferTest, SkipGramInferTest0) {
  size_t inputs_size = 1;
  std::vector<TensorC *> inputs(inputs_size, NULL);
  inputs[0] = new TensorC;
  inputs[0]->data_ = NULL;
  inputs[0]->data_type_ = kNumberTypeInt8;
  inputs[0]->format_ = kNHWC_C;
  std::vector<TensorC *> outputs(1, NULL);
  outputs[0] = new TensorC;
  OpParameter *parameter = new OpParameter;
  int ret = SkipGramInferShape((const TensorC **)inputs.data(), inputs.size(), outputs.data(), outputs.size(),
                               reinterpret_cast<OpParameter *>(parameter));
  ASSERT_EQ(ret, NNACL_INFER_INVALID);
  ASSERT_EQ(outputs[0]->format_, kNHWC_C);
  ASSERT_EQ(outputs[0]->data_type_, kNumberTypeInt8);
  delete parameter;
  for (size_t i = 0; i < inputs_size; i++) {
    delete inputs[i];
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    delete outputs[i];
  }
}

}  // namespace mindspore
