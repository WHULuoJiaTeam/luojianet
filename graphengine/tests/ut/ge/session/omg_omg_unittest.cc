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

#include <gtest/gtest.h>

#include "common/ge/ge_util.h"
#include "proto/ge_ir.pb.h"
#include "inc/framework/omg/omg.h"


using namespace std;

namespace ge {
class UtestOmg : public testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(UtestOmg, display_model_info_failed) {
  ge::proto::ModelDef model_def;
  PrintModelInfo(&model_def, 1);
}

TEST_F(UtestOmg, display_model_info_success) {
  ge::proto::ModelDef model_def;
  auto attrs = model_def.mutable_attr();
  ge::proto::AttrDef *attr_def_soc = &(*attrs)["soc_version"];
  attr_def_soc->set_s("Ascend310");
  ge::proto::AttrDef *attr_def = &(*attrs)["om_info_list"];
  attr_def->mutable_list()->add_i(1);
  attr_def->mutable_list()->add_i(2);
  attr_def->mutable_list()->add_i(3);
  attr_def->mutable_list()->add_i(4);
  PrintModelInfo(&model_def, 1);
}
}  // namespace ge
