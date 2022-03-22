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
#include <memory>

#include "common/ge_inner_error_codes.h"
#include "common/types.h"
#include "common/util.h"
#include "graph/utils/attr_utils.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/passes/addn_pass.h"

#define private public
#define protected public
#include "graph/manager/util/hcom_util.h"
#include "ge/ge_api.h"
#undef private
#undef protected

using namespace std;

namespace ge {
namespace {
GeTensorDescPtr CreateTensorDesc(std::initializer_list<int64_t> shape, Format format = FORMAT_NCHW,
                                 DataType data_type = DT_FLOAT) {
  GeShape ge_shape{vector<int64_t>(shape)};
  GeTensorDescPtr tensor_desc = std::make_shared<GeTensorDesc>();
  tensor_desc->SetShape(ge_shape);
  tensor_desc->SetFormat(format);
  tensor_desc->SetDataType(data_type);
  return tensor_desc;
}

class NodeBuilder {
 public:
  NodeBuilder(const std::string &name, const std::string &type) { op_desc_ = std::make_shared<OpDesc>(name, type); }

  NodeBuilder &AddInputDesc(std::initializer_list<int64_t> shape = {1, 1, 224, 224}, Format format = FORMAT_NCHW,
                            DataType data_type = DT_FLOAT) {
    op_desc_->AddInputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(std::initializer_list<int64_t> shape = {1, 1, 224, 224}, Format format = FORMAT_NCHW,
                             DataType data_type = DT_FLOAT) {
    op_desc_->AddOutputDesc(CreateTensorDesc(shape, format, data_type)->Clone());
    return *this;
  }

  NodeBuilder &AddOutputDesc(GeTensorDescPtr tensor_desc) {
    op_desc_->AddOutputDesc(tensor_desc->Clone());
    return *this;
  }

  NodePtr Build(const ComputeGraphPtr &graph) {
    NodePtr node = graph->AddNode(op_desc_);
    return node;
  }

 private:
  OpDescPtr op_desc_;
};
}  // namespace

class UtestHcomUtil : public testing::Test {
 protected:
  void SetUp() {
  }
  void TearDown() {
  }
};

TEST_F(UtestHcomUtil, test_GetHcomCount_succ) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = NodeBuilder("node", HCOMRECEIVE).AddInputDesc({1, 1, 224, 224}).AddOutputDesc({1, 1, 224, 224}).Build(graph);
  auto op_desc = node->GetOpDesc();

  HcomOmeUtil hcom_ome_util;
  int count = 0;
  auto ret = hcom_ome_util.GetHcomCount(op_desc, HCCL_DATA_TYPE_FP32, true, count);
  EXPECT_EQ(ret, 0);
}

TEST_F(UtestHcomUtil, test_GetHcomCount_succ_2) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  NodePtr node = NodeBuilder("node", HCOMSEND).AddInputDesc({1, 1, 224, 224}).Build(graph);
  auto op_desc = node->GetOpDesc();
  HcomOmeUtil hcom_util;
  int count = 0;
  auto ret = hcom_util.GetHcomCount(op_desc, HCCL_DATA_TYPE_FP32, true, count);
  EXPECT_EQ(ret, SUCCESS);
  EXPECT_EQ(count, 224 * 224);
}
}  // namespace ge