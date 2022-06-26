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
#include <cmath>
#include <memory>
#include "schema/inner/model_generated.h"
#include "common/common_test.h"
#include "include/api/context.h"
#include "include/api/model.h"
#include "include/lite_session.h"
#include "include/context.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/lite_session.h"
#include "src/runtime/inner_allocator.h"
#include "include/registry/register_kernel_interface.h"
#include "include/registry/register_kernel.h"

using mindspore::kernel::Kernel;
using mindspore::kernel::KernelInterface;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_AddFusion;

namespace mindspore {
class TestCustomAdd : public Kernel {
 public:
  TestCustomAdd(const std::vector<mindspore::MSTensor> &inputs, const std::vector<mindspore::MSTensor> &outputs,
                const schema::Primitive *primitive, const mindspore::Context *ctx)
      : Kernel(inputs, outputs, primitive, ctx) {}
  int Prepare() override { return 0; }

  int Execute() override;

  int ReSize() override { return 0; }

 private:
  int PreProcess() {
    for (auto &output : outputs_) {
      // malloc data for output tensor
      auto data = output.MutableData();
      if (data == nullptr) {
        MS_LOG(ERROR) << "Get data failed";
        return RET_ERROR;
      }
    }
    return RET_OK;
  }
};

int TestCustomAdd::Execute() {
  if (inputs_.size() != 2) {
    return RET_PARAM_INVALID;
  }
  PreProcess();
  auto *in0 = static_cast<const float *>(inputs_[0].Data().get());
  auto *in1 = static_cast<const float *>(inputs_[1].Data().get());
  float *out = static_cast<float *>(outputs_[0].MutableData());
  auto num = outputs_[0].ElementNum();
  for (int i = 0; i < num; ++i) {
    out[i] = in0[i] + in1[i];
  }
  return RET_OK;
}

class TestCustomAddInfer : public KernelInterface {
 public:
  TestCustomAddInfer() = default;
  ~TestCustomAddInfer() = default;
  Status Infer(std::vector<mindspore::MSTensor> *inputs, std::vector<mindspore::MSTensor> *outputs,
               const schema::Primitive *primitive) override {
    (*outputs)[0].SetFormat((*inputs)[0].format());
    (*outputs)[0].SetDataType((*inputs)[0].DataType());
    (*outputs)[0].SetShape((*inputs)[0].Shape());
    return kSuccess;
  }
};

namespace {
std::shared_ptr<Kernel> TestCustomAddCreator(const std::vector<MSTensor> &inputs, const std::vector<MSTensor> &outputs,
                                             const schema::Primitive *primitive, const mindspore::Context *ctx) {
  return std::make_shared<TestCustomAdd>(inputs, outputs, primitive, ctx);
}

std::shared_ptr<KernelInterface> CustomAddInferCreator() { return std::make_shared<TestCustomAddInfer>(); }
const auto kFloat32 = DataType::kNumberTypeFloat32;
}  // namespace

REGISTER_KERNEL(CPU, BuiltInTest, kFloat32, PrimitiveType_AddFusion, TestCustomAddCreator)
REGISTER_KERNEL_INTERFACE(BuiltInTest, PrimitiveType_AddFusion, CustomAddInferCreator)

class TestRegistry : public mindspore::CommonTest {
 public:
  TestRegistry() = default;
};

TEST_F(TestRegistry, TestAdd) {
  auto meta_graph = std::make_shared<schema::MetaGraphT>();
  meta_graph->name = "graph";

  auto node = std::make_unique<schema::CNodeT>();
  node->inputIndex = {0, 1};
  node->outputIndex = {2};
  node->primitive = std::make_unique<schema::PrimitiveT>();
  node->primitive->value.type = schema::PrimitiveType_AddFusion;
  auto primitive = new schema::AddFusionT;
  node->primitive->value.value = primitive;
  node->name = "Add";
  meta_graph->nodes.emplace_back(std::move(node));
  meta_graph->inputIndex = {0, 1};
  meta_graph->outputIndex = {2};

  auto input0 = std::make_unique<schema::TensorT>();
  input0->nodeType = lite::NodeType_Parameter;
  input0->format = schema::Format_NHWC;
  input0->dataType = TypeId::kNumberTypeFloat32;
  input0->dims = {1, 28, 28, 3};
  input0->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(input0));

  auto weight = std::make_unique<schema::TensorT>();
  weight->nodeType = lite::NodeType_ValueNode;
  weight->format = schema::Format_NHWC;
  weight->dataType = TypeId::kNumberTypeFloat32;
  weight->dims = {1, 28, 28, 3};

  weight->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(weight));

  auto output = std::make_unique<schema::TensorT>();
  output->nodeType = lite::NodeType_Parameter;
  output->format = schema::Format_NHWC;
  output->dataType = TypeId::kNumberTypeFloat32;
  output->offset = -1;
  meta_graph->allTensors.emplace_back(std::move(output));

  flatbuffers::FlatBufferBuilder builder(1024);
  auto offset = schema::MetaGraph::Pack(builder, meta_graph.get());
  builder.Finish(offset);
  schema::FinishMetaGraphBuffer(builder, offset);
  size_t size = builder.GetSize();
  const char *content = reinterpret_cast<char *>(builder.GetBufferPointer());

  // create a context
  auto context = std::make_shared<mindspore::Context>();
  context->SetThreadNum(1);
  context->SetEnableParallel(false);
  context->SetThreadAffinity(1);
  auto &device_list = context->MutableDeviceInfo();
  std::shared_ptr<CPUDeviceInfo> device_info = std::make_shared<CPUDeviceInfo>();
  device_info->SetEnableFP16(false);
  device_list.push_back(device_info);

  // build a model
  auto model = std::make_shared<mindspore::Model>();
  auto ret = model->Build(content, size, kMindIR_Lite, context);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  auto inputs = model->GetInputs();
  ASSERT_EQ(inputs.size(), 2);
  auto inTensor = inputs.front();
  auto impl = inTensor.impl();
  ASSERT_NE(nullptr, impl);
  float *in0_data = static_cast<float *>(inTensor.MutableData());
  in0_data[0] = 10.0f;
  auto inTensor1 = inputs.back();
  impl = inTensor1.impl();
  ASSERT_NE(nullptr, impl);
  float *in1_data = static_cast<float *>(inTensor1.MutableData());
  in1_data[0] = 20.0f;
  std::vector<mindspore::MSTensor> outputs;
  ret = model->Predict(inputs, &outputs);
  ASSERT_EQ(kSuccess, ret.StatusCode());
  ASSERT_EQ(outputs.size(), 1);
  impl = outputs.front().impl();
  ASSERT_NE(nullptr, impl);
  ASSERT_EQ(28 * 28 * 3, outputs.front().ElementNum());
  ASSERT_EQ(DataType::kNumberTypeFloat32, outputs.front().DataType());
  auto *outData = reinterpret_cast<const float *>(outputs.front().Data().get());
  ASSERT_NE(nullptr, outData);
  ASSERT_EQ(30.0f, outData[0]);
  MS_LOG(INFO) << "Register add op test pass.";
}

}  // namespace mindspore
