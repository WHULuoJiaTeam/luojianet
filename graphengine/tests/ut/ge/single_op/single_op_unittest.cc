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
#include <vector>

#include "runtime/rt.h"

#define protected public
#define private public
#include "single_op/single_op.h"
#include "single_op/single_op_manager.h"
#include "single_op/task/build_task_utils.h"
#undef private
#undef protected

using namespace std;
using namespace ge;

class UtestSingleOp : public testing::Test {
 protected:
  void SetUp() {}
  void TearDown() {}
};

TEST_F(UtestSingleOp, test_dynamic_singleop_execute_async) {
  uintptr_t resource_id = 0;
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  DynamicSingleOp dynamic_single_op(resource_id, &stream_mu, stream);

  vector<int64_t> dims_vec_0 = {2};
  vector<GeTensorDesc> input_desc;
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  // input data from device
  AttrUtils::SetInt(tensor_desc_0, ATTR_NAME_PLACEMENT, 0);
  input_desc.emplace_back(tensor_desc_0);

  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer;
  data_buffer.data = new char[4];
  data_buffer.length = 4;
  input_buffers.emplace_back(data_buffer);

  vector<GeTensorDesc> output_desc;
  vector<DataBuffer> output_buffers;

  // UpdateRunInfo failed
  EXPECT_EQ(dynamic_single_op.ExecuteAsync(input_desc, input_buffers, output_desc, output_buffers), ACL_ERROR_GE_PARAM_INVALID);
}

TEST_F(UtestSingleOp, test_dynamic_singleop_execute_async1) {
  uintptr_t resource_id = 0;
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  DynamicSingleOp dynamic_single_op(resource_id, &stream_mu, stream);
  dynamic_single_op.num_inputs_ = 1;

  vector<int64_t> dims_vec_0 = {2};
  vector<GeTensorDesc> input_desc;
  GeTensorDesc tensor_desc_0(GeShape(dims_vec_0), FORMAT_NCHW, DT_INT32);
  // input data from host
  AttrUtils::SetInt(tensor_desc_0, ATTR_NAME_PLACEMENT, 1);
  input_desc.emplace_back(tensor_desc_0);

  int64_t input_size = 0;
  EXPECT_EQ(TensorUtils::GetTensorMemorySizeInBytes(tensor_desc_0, input_size), SUCCESS);
  EXPECT_EQ(input_size, 64);
  EXPECT_NE(SingleOpManager::GetInstance().GetResource(resource_id, stream), nullptr);

  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer;
  data_buffer.data = new char[4];
  data_buffer.length = 4;
  input_buffers.emplace_back(data_buffer);

  vector<GeTensorDesc> output_desc;
  vector<DataBuffer> output_buffers;

  auto *tbe_task = new (std::nothrow) TbeOpTask();
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  tbe_task->node_ = node;

  dynamic_single_op.op_task_.reset((OpTask *)(tbe_task));

  OpDescPtr desc_ptr = MakeShared<OpDesc>("name1", "type1");
  EXPECT_EQ(desc_ptr->AddInputDesc("x", GeTensorDesc(GeShape({2}), FORMAT_NCHW)), GRAPH_SUCCESS);
  dynamic_single_op.op_task_->op_desc_ = desc_ptr;
  // UpdateRunInfo failed
  EXPECT_EQ(dynamic_single_op.ExecuteAsync(input_desc, input_buffers, output_desc, output_buffers), ACL_ERROR_GE_PARAM_INVALID);
}


TEST_F(UtestSingleOp, test_singleop_execute_async1) {
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  SingleOp single_op(res, &stream_mu, stream);

  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer;
  data_buffer.data = new char[4];
  data_buffer.length = 4;
  data_buffer.placement = 1;
  input_buffers.emplace_back(data_buffer);
  vector<DataBuffer> output_buffers;

  single_op.input_sizes_.emplace_back(4);
  SingleOpModelParam model_params;
  single_op.running_param_.reset(new (std::nothrow)SingleOpModelParam(model_params));
  single_op.args_.resize(1);

  auto *tbe_task = new (std::nothrow) TbeOpTask();
  ge::OpDescPtr op_desc = std::make_shared<OpDesc>("Mul", MATMUL);
  EXPECT_EQ(op_desc->AddInputDesc("x", GeTensorDesc(GeShape({2}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_EQ(op_desc->AddOutputDesc("x", GeTensorDesc(GeShape({2}), FORMAT_NCHW)), GRAPH_SUCCESS);
  EXPECT_NE(BuildTaskUtils::GetTaskInfo(op_desc), "");
  ge::ComputeGraphPtr graph = std::make_shared<ge::ComputeGraph>("default");
  ge::NodePtr node = graph->AddNode(op_desc);
  tbe_task->node_ = node;
  tbe_task->op_desc_ = op_desc;
  single_op.tasks_.push_back(tbe_task);
  EXPECT_EQ(single_op.hybrid_model_executor_, nullptr);
  EXPECT_EQ(single_op.running_param_->mem_base, nullptr);
  EXPECT_EQ(single_op.ExecuteAsync(input_buffers, output_buffers), SUCCESS);
}

TEST_F(UtestSingleOp, test_singleop_execute_async2) {
  StreamResource *res = new (std::nothrow) StreamResource(1);
  std::mutex stream_mu;
  rtStream_t stream = nullptr;
  rtStreamCreate(&stream, 0);
  SingleOp single_op(res, &stream_mu, stream);

  vector<DataBuffer> input_buffers;
  ge::DataBuffer data_buffer;
  data_buffer.data = new char[4];
  data_buffer.length = 4;
  data_buffer.placement = 1;
  input_buffers.emplace_back(data_buffer);
  vector<DataBuffer> output_buffers;

  single_op.input_sizes_.emplace_back(4);
  SingleOpModelParam model_params;
  single_op.running_param_.reset(new (std::nothrow)SingleOpModelParam(model_params));
  single_op.args_.resize(1);

  GeTensorDesc tensor_desc(GeShape({1}), FORMAT_NHWC, DT_UINT64);
  single_op.inputs_desc_.emplace_back(tensor_desc);
  std::shared_ptr<ge::GeRootModel> root_model = ge::MakeShared<ge::GeRootModel>();
  single_op.hybrid_model_.reset(new (std::nothrow)hybrid::HybridModel(root_model));
  single_op.hybrid_model_executor_.reset(new (std::nothrow)hybrid::HybridModelExecutor(single_op.hybrid_model_.get(), 0, stream));
  EXPECT_EQ(single_op.running_param_->mem_base, nullptr);
  EXPECT_EQ(single_op.tasks_.size(), 0);

  GeTensorDesc tensor;
  int64_t storage_format_val = static_cast<Format>(FORMAT_NCHW);
  AttrUtils::SetInt(tensor, "storage_format", storage_format_val);
  std::vector<int64_t> storage_shape{1, 1, 1, 1};
  AttrUtils::SetListInt(tensor, "storage_shape", storage_shape);
  single_op.inputs_desc_.emplace_back(tensor);
  EXPECT_EQ(single_op.ExecuteAsync(input_buffers, output_buffers), PARAM_INVALID);
}

TEST_F(UtestSingleOp, test_set_host_mem) {
  std::mutex stream_mu_;
  DynamicSingleOp single_op(0, &stream_mu_, nullptr);
  
  vector<DataBuffer> input_buffers;
  DataBuffer data_buffer;
  input_buffers.emplace_back(data_buffer);

  vector<GeTensorDesc> input_descs;
  GeTensorDesc tensor_desc1;
  input_descs.emplace_back(tensor_desc1);

  vector<GeTensorDescPtr> op_input_descs;
  auto tensor_desc2 = std::make_shared<GeTensorDesc>();
  op_input_descs.emplace_back(tensor_desc2);
  single_op.tensor_with_hostmem_[0] = op_input_descs;
  EXPECT_EQ(single_op.SetHostTensorValue(input_descs, input_buffers), SUCCESS);
}
