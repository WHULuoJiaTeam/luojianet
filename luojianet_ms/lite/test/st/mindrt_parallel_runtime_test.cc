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

#include "ut/src/runtime/kernel/opencl/common.h"
#include "include/errorcode.h"
#include "src/mindrt_executor.h"
#include "src/lite_session.h"
#include "src/lite_kernel.h"

class MindrtRuntimeTest : public luojianet_ms::CommonTest {
 public:
  MindrtRuntimeTest() = default;
};

int CheckRuntime(luojianet_ms::session::LiteSession *session) {
  luojianet_ms::lite::LiteSession *lite_session = reinterpret_cast<luojianet_ms::lite::LiteSession *>(session);
  auto kernels = lite_session->get_kernels();

  int cpu_kernel_count = 0;
  int gpu_kernel_count = 0;
  for (auto kernel : kernels) {
    if (kernel->subgraph_type() == luojianet_ms::kernel::kGpuFp32SubGraph) {
      gpu_kernel_count++;
    }
    if (kernel->subgraph_type() == luojianet_ms::kernel::kCpuFP32SubGraph) {
      cpu_kernel_count++;
    }
  }

  if (kernels.size() != 6) {
    return -1;
  }
  if (cpu_kernel_count != 4) {
    return -2;
  }
  if (gpu_kernel_count != 2) {
    return -3;
  }

  return 0;
}

TEST_F(MindrtRuntimeTest, Runtime) {
  size_t size = 0;
  char *graph_buf = luojianet_ms::lite::ReadFile("./test_data/mindrt_parallel/parallel.ms", &size);
  ASSERT_NE(graph_buf, nullptr);

  auto model = std::shared_ptr<luojianet_ms::lite::Model>(luojianet_ms::lite::Model::Import(graph_buf, size));
  delete[](graph_buf);
  ASSERT_NE(model, nullptr);

  auto context = std::make_shared<luojianet_ms::lite::Context>();
  ASSERT_NE(context, nullptr);
  context->enable_parallel_ = true;
  luojianet_ms::lite::DeviceContext gpu_device_ctx{luojianet_ms::lite::DT_GPU, {false}};
  gpu_device_ctx.device_info_.gpu_device_info_.enable_float16_ = false;
  context->device_list_.push_back(gpu_device_ctx);

  luojianet_ms::session::LiteSession *session = luojianet_ms::session::LiteSession::CreateSession(context.get());
  ASSERT_NE(session, nullptr);

  int benchmark_ret = session->CompileGraph(model.get());
  ASSERT_EQ(benchmark_ret, luojianet_ms::lite::RET_OK);

  ASSERT_EQ(CheckRuntime(session), 0);

  auto inputs = session->GetInputs();
  for (auto in : inputs) {
    in->MutableData();
  }
  benchmark_ret = session->RunGraph(nullptr, nullptr);
  ASSERT_EQ(benchmark_ret, luojianet_ms::lite::RET_OK);

  delete session;
}

int CheckRuntime2(luojianet_ms::session::LiteSession *session) {
  luojianet_ms::lite::LiteSession *lite_session = reinterpret_cast<luojianet_ms::lite::LiteSession *>(session);
  auto kernels = lite_session->get_kernels();

  for (auto kernel : kernels) {
    if (kernel->subgraph_type() != luojianet_ms::kernel::kCpuFP16SubGraph) {
      return -1;
    }
  }

  if (kernels.size() != 6) {
    return -2;
  }

  return 0;
}

TEST_F(MindrtRuntimeTest, RuntimeFp16) {
  size_t size = 0;
  char *graph_buf = luojianet_ms::lite::ReadFile("./test_data/mindrt_parallel/parallel.ms", &size);
  ASSERT_NE(graph_buf, nullptr);

  auto model = std::shared_ptr<luojianet_ms::lite::Model>(luojianet_ms::lite::Model::Import(graph_buf, size));
  delete[](graph_buf);
  ASSERT_NE(model, nullptr);

  auto context = std::make_shared<luojianet_ms::lite::Context>();
  ASSERT_NE(context, nullptr);
  context->enable_parallel_ = true;
  auto &cpu_device_ctx = context->device_list_[0];
  cpu_device_ctx.device_info_.cpu_device_info_.enable_float16_ = true;

  luojianet_ms::session::LiteSession *session = luojianet_ms::session::LiteSession::CreateSession(context.get());
  ASSERT_NE(session, nullptr);

  int benchmark_ret = session->CompileGraph(model.get());
  ASSERT_EQ(benchmark_ret, luojianet_ms::lite::RET_OK);

  ASSERT_EQ(CheckRuntime2(session), 0);

  auto inputs = session->GetInputs();
  for (auto in : inputs) {
    in->MutableData();
  }
  benchmark_ret = session->RunGraph(nullptr, nullptr);
  ASSERT_EQ(benchmark_ret, luojianet_ms::lite::RET_OK);

  delete session;
}
