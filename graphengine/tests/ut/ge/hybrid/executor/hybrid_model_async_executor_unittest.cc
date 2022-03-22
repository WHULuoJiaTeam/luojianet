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
#include <gmock/gmock.h>
#include <vector>

#define private public
#define protected public
#include "hybrid/executor/hybrid_model_async_executor.h"
#include "hybrid/executor/hybrid_model_executor.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_attr_define.h"


using namespace std;
using namespace testing;


namespace ge {
using namespace hybrid;

class UtestHybridModelAsyncExecutor : public testing::Test {
 protected:
  void SetUp() {}

  void TearDown() { }
};

TEST_F(UtestHybridModelAsyncExecutor, CopyOutputs_success) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelAsyncExecutor executor(&hybrid_model);
  
  TensorValue input_tensor;
  HybridModelExecutor::ExecuteArgs args;
  args.inputs.emplace_back(input_tensor);
  auto desc = MakeShared<GeTensorDesc>();
  GeShape geshape({2,2,2,2});
  desc->SetShape(geshape);

  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto output_tensor = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  args.outputs.emplace_back(output_tensor);
  args.output_desc.emplace_back(desc);
  
  OutputData output_data;
  std::vector<ge::Tensor> outputs;
  auto ret = executor.CopyOutputs(args, &output_data, outputs);
  ASSERT_EQ(ret,SUCCESS);
}

TEST_F(UtestHybridModelAsyncExecutor, BuildDeviceTensor) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelAsyncExecutor executor(&hybrid_model);
  
  auto allocator = NpuMemoryAllocator::GetAllocator();
  auto tensor_buffer = TensorBuffer::Create(allocator, 100);
  auto tensor = TensorValue(shared_ptr<TensorBuffer>(tensor_buffer.release()));
  GeTensorDesc ge_tensor_desc;
  int64_t output_size = 100;
  std::vector<ge::Tensor> outputs;
  auto ret = executor.BuildDeviceTensor(tensor, ge_tensor_desc, output_size, outputs);
  auto size = tensor.GetSize();
  ASSERT_EQ(size, 100);
}

TEST_F(UtestHybridModelAsyncExecutor, Test_execute) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  HybridModel hybrid_model(ge_root_model);
  hybrid_model.root_graph_item_.reset(new GraphItem);

  HybridModelExecutor executor(&hybrid_model, 0, nullptr);
  ASSERT_EQ(executor.Init(), SUCCESS);
  auto &context = executor.context_;
  HybridModelExecutor::ExecuteArgs args;
  std::pair<rtEvent_t, std::pair<rtCallback_t, void *>> eof_entry;
  eof_entry.first = nullptr;
  context.callback_manager->callback_queue_.Push(eof_entry);
  ASSERT_EQ(executor.Execute(args), SUCCESS);
}

TEST_F(UtestHybridModelAsyncExecutor, test_PrepareInputs) {
  ComputeGraphPtr graph = std::make_shared<ComputeGraph>("test");
  GeRootModelPtr ge_root_model = make_shared<GeRootModel>(graph);
  ge_root_model->SetModelName("test_name");
  GeModelPtr ge_sub_model = make_shared<GeModel>();
  HybridModel hybrid_model(ge_root_model);
  HybridModelAsyncExecutor executor(&hybrid_model);
  GeTensorDescPtr tensor_desc = make_shared<GeTensorDesc>(GeShape({-1, 16, 16, 3}));
  tensor_desc->SetShapeRange({{1, 256}, {16, 16}, {16, 16}, {3, 3}});
  executor.input_tensor_desc_.insert({0, tensor_desc});
  executor.device_id_ = 0;
  executor.input_sizes_.insert({0, -1});
  executor.is_input_dynamic_.push_back(true);

  unique_ptr<uint8_t[]> data_buf(new (std::nothrow)uint8_t[3072]);
  InputData input_data;
  input_data.blobs.push_back(DataBuffer(data_buf.get(), 3072, false));
  input_data.shapes.push_back({1, 16, 16, 3});
  HybridModelExecutor::ExecuteArgs args;

  auto ret = executor.PrepareInputs(input_data, args);
  ASSERT_EQ(ret, SUCCESS);
  ASSERT_EQ(args.input_desc[0]->GetShape().ToString(), GeShape({1, 16, 16, 3}).ToString());
  int64_t tensor_size = 0;
  TensorUtils::GetSize(*(args.input_desc[0]), tensor_size);
  ASSERT_EQ(tensor_size, 3104);
}
} // namespace ge