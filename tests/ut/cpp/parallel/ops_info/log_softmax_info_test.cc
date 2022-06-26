/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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

#include <string>
#include <list>
#include <vector>
#include "common/common_test.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/ops_info/activation_info.h"
#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/step_parallel.h"

namespace mindspore {
namespace parallel {

class LogSoftmaxInfo;
using LogSoftmaxInfoPtr = std::shared_ptr<LogSoftmaxInfo>;
LogSoftmaxInfoPtr log_softmax;

class TestLogSoftmaxInfo : public UT::Common {
 public:
  TestLogSoftmaxInfo() {}
  void SetUp();
  void TearDown() {}
};

void TestLogSoftmaxInfo::SetUp() {
  RankList dev_list;

  for (int32_t i = 0; i < 130; i++) {
    dev_list.push_back(i);
  }

  RankList stage_map;
  stage_map.push_back(128);
  stage_map.push_back(2);

  int32_t local_dev = 0;

  // create a new g_device_manager
  g_device_manager = std::make_shared<DeviceManager>();
  g_device_manager->Init(dev_list, local_dev, stage_map, "hccl");

  ValuePtr axis = MakeValue(static_cast<int64_t>(-2));
  mindspore::HashMap<std::string, ValuePtr> attr = {{"axis", axis}};

  Shapes inputs_shape = {{2, 4, 8, 16}};
  Shapes outputs_shape = {{2, 4, 8, 16}};

  log_softmax = std::make_shared<LogSoftmaxInfo>("log_softmax_info", inputs_shape, outputs_shape, attr);
}

TEST_F(TestLogSoftmaxInfo, InferDevMatrixShape1) {
  Strategys inputs = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  log_softmax->Init(strategy, nullptr);
  Shape dev_matrix_shape = log_softmax->dev_matrix_shape();

  Shape expect = {2, 4, 1, 16};
  ASSERT_EQ(dev_matrix_shape, expect);
}

TEST_F(TestLogSoftmaxInfo, InferSliceShape1) {
  Strategys str = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, str);

  log_softmax->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = log_softmax->inputs_tensor_info();
  std::vector<TensorInfo> outputs = log_softmax->outputs_tensor_info();

  Shape input_slice_shape_expect = {1, 1, 8, 1};
  Shape output_slice_shape_expect = {1, 1, 8, 1};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Shape input_slice_shape = input_tensor_info.slice_shape();
  Shape output_slice_shape = output_tensor_info.slice_shape();

  ASSERT_EQ(input_slice_shape, input_slice_shape_expect);
  ASSERT_EQ(output_slice_shape, output_slice_shape_expect);
}

TEST_F(TestLogSoftmaxInfo, GetTensorLayout1) {
  Strategys str = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, str);

  log_softmax->Init(strategy, nullptr);
  std::vector<TensorInfo> inputs = log_softmax->inputs_tensor_info();
  std::vector<TensorInfo> outputs = log_softmax->outputs_tensor_info();

  TensorMap input_expect = {3, 2, 1, 0};
  TensorMap output_expect = {3, 2, 1, 0};

  TensorInfo input_tensor_info = inputs.at(0);
  TensorInfo output_tensor_info = outputs.at(0);

  Map input_tensor_map = input_tensor_info.tensor_layout().origin_tensor_map();
  Map output_tensor_map = output_tensor_info.tensor_layout().origin_tensor_map();

  ASSERT_EQ(input_tensor_map.array(), input_expect);
  ASSERT_EQ(output_tensor_map.array(), output_expect);
}

TEST_F(TestLogSoftmaxInfo, GetForwardOp1) {
  Strategys inputs = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  log_softmax->Init(strategy, nullptr);
  OperatorVector forward_op = log_softmax->forward_op();
  size_t size = forward_op.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestLogSoftmaxInfo, GetMirrorOPs1) {
  Strategys inputs = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  log_softmax->Init(strategy, nullptr);
  MirrorOps mirror_ops = log_softmax->mirror_ops();

  size_t size = mirror_ops.size();

  ASSERT_EQ(size, 0);
}

TEST_F(TestLogSoftmaxInfo, CheckStrategy1) {
  // Success: {{2,4,1,16}}
  Strategys inputs = {{2, 2, 8, 16}, {2, 4, 16, 1}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = log_softmax->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestLogSoftmaxInfo, CheckStrategy2) {
  // Success: {{2,4,1,16}}
  Strategys inputs = {{2, 4, 8}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = log_softmax->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestLogSoftmaxInfo, CheckStrategy3) {
  // Success: {{2,4,1,16}}
  Strategys inputs = {{2, 4, 8, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  Status ret = log_softmax->Init(strategy, nullptr);
  ASSERT_EQ(ret, FAILED);
}

TEST_F(TestLogSoftmaxInfo, GetDeviceList1) {
  Strategys inputs = {{2, 4, 1, 16}};
  StrategyPtr strategy = NewStrategy(0, inputs);

  log_softmax->Init(strategy, nullptr);
  RankList dev_list = log_softmax->stage_device_list();
  ASSERT_EQ(dev_list.size(), 128);
}

}  // namespace parallel
}  // namespace mindspore
