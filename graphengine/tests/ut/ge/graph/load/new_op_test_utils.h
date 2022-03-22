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

#ifndef OME_REBUILD_OME_OP_TEST_UTILS_H
#define OME_REBUILD_OME_OP_TEST_UTILS_H

#include <gtest/gtest.h>
#include <memory>
#include <utility>

#include "common/fmk_types.h"
#include "common/helper/model_helper.h"
#include "common/op/attr_value_util.h"
#include "common/properties_manager.h"
#include "common/types.h"
#include "executor/ge_executor.h"
#include "graph/buffer.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/ge_attr_value.h"
#include "graph/model_serialize.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "proto/ge_ir.pb.h"

#define protected public
#define private public
#include "graph/compute_graph.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/load/model_manager/davinci_model.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/utils/attr_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#undef protected
#undef private

using namespace ge;

class GlobalModelData {
 public:
  GlobalModelData() {}

  ~GlobalModelData() {
    if (data_.model_data != nullptr) {
      delete[](uint8_t *) data_.model_data;
      data_.model_data = nullptr;
    }
  }

  ge::ModelData data_;
};

static GlobalModelData g_model_data;

class OmeTestOpUtils {
 public:
  static void InitModel(std::shared_ptr<ge::DavinciModel> davinciModel) { InitModel(*davinciModel); }
  static ge::NodePtr GenNodeFromOpDesc(ge::OpDescPtr op_desc) {
    if (!op_desc) {
      return nullptr;
    }

    auto g = std::make_shared<ge::ComputeGraph>("g");
    return g->AddNode(std::move(op_desc));
  }

  static void AddInputOutputToTaskModel(std::shared_ptr<ge::Model> model,
                                        std::shared_ptr<domi::ModelTaskDef> model_task_def) {
    uint32_t stream_num111 = model_task_def->stream_num();
    uint32_t weights_num = model_task_def->weight_size();
    uint32_t mem_num = model_task_def->memory_size();

    int64_t memory_size = 0;
    int64_t weight_size = 0;
    (void)ge::AttrUtils::GetInt(model.get(), ATTR_MODEL_MEMORY_SIZE, memory_size);
    (void)ge::AttrUtils::GetInt(model.get(), ATTR_MODEL_WEIGHT_SIZE, weight_size);
    // Save memory_size/weight_size/stream_num/event_num to proto
    model_task_def->set_memory_size(memory_size);
    model_task_def->set_weight_size(weight_size);
    int64_t stream_num = 0;
    (void)ge::AttrUtils::GetInt(model.get(), ATTR_MODEL_STREAM_NUM, stream_num);
    model_task_def->set_stream_num(stream_num);

    ge::ComputeGraphPtr graph = ge::GraphUtils::GetComputeGraph(model->GetGraph());
    vector<ConstOpDescPtr> op_desc_ptrs;
    for (const auto &node_ptr : graph->GetAllNodes()) {
      if (node_ptr->GetType() == DATA_TYPE || node_ptr->GetType() == ANN_DATA_TYPE) {
        op_desc_ptrs.push_back(node_ptr->GetOpDesc());
        continue;
      }

      for (auto tensor_desc : node_ptr->GetOpDesc()->GetAllOutputsDescPtr()) {
        bool is_output = false;
        ge::TensorUtils::GetOutputTensor(*tensor_desc, is_output);
        if (is_output) {
          // output Op and add to array
          op_desc_ptrs.push_back(node_ptr->GetOpDesc());
          break;
        }
      }
    }

    // save multi OpDescPtr to attr
    ge::ModelSerialize model_serialize;
    for (auto op_desc_ptr : op_desc_ptrs) {
      ge::Buffer buffer = model_serialize.SerializeOpDesc(op_desc_ptr);
      model_task_def->add_op(string(reinterpret_cast<const char *>(buffer.GetData()), buffer.GetSize()));
    }

    int64_t run_mode = -1;
    for (auto node_ptr : graph->GetAllNodes()) {
      // TE CUSTOM op need to init
      if (ge::AttrUtils::GetInt(node_ptr->GetOpDesc(), ATTR_NAME_IMPLY_TYPE, run_mode) &&
          run_mode != (uint32_t)domi::ImplyType::BUILDIN && run_mode != (uint32_t)domi::ImplyType::INVALID) {
        (*(model_task_def->mutable_attr()))["contain_custom"] = "1";
        break;
      }
    }
  }

  static Status TransModelToGeModel(const ModelPtr &model, GeModelPtr &ge_model) {
    if (model == nullptr) {
      GELOGE(FAILED, "Model is null");
      return FAILED;
    }
    ge_model = ge::MakeShared<ge::GeModel>();
    GE_CHECK_NOTNULL(ge_model);
    ge_model->SetGraph(model->GetGraph());
    ge_model->SetName(model->GetName());
    ge_model->SetVersion(model->GetVersion());
    ge_model->SetPlatformVersion(model->GetPlatformVersion());
    ge_model->SetAttr(model->MutableAttrMap());

    auto compute_graph = ge::GraphUtils::GetComputeGraph(model->GetGraph());
    ge::Buffer weight;
    (void)ge::AttrUtils::GetZeroCopyBytes(compute_graph, ge::ATTR_NAME_WEIGHTS_DATA, weight);
    ge_model->SetWeight(weight);
    if (model->HasAttr(MODEL_ATTR_TASKS)) {
       ge::Buffer task_buffer;
       GE_CHK_BOOL_RET_STATUS(ge::AttrUtils::GetZeroCopyBytes(model, MODEL_ATTR_TASKS, task_buffer), FAILED,
		                          "Get bytes failed.");
       std::shared_ptr<ModelTaskDef> task = ge::MakeShared<ModelTaskDef>();
       GE_CHECK_NOTNULL(task);
       GE_IF_BOOL_EXEC(task_buffer.GetData() == nullptr, GELOGE(FAILED, "Get data fail"); return FAILED);
       GE_IF_BOOL_EXEC(task_buffer.GetSize() == 0, GELOGE(FAILED, "Get size fail"); return FAILED);
       GE_CHK_BOOL_EXEC(ReadProtoFromArray(task_buffer.GetData(), static_cast<int>(task_buffer.GetSize()), task.get()),
		        return INTERNAL_ERROR, "ReadProtoFromArray failed.");
       ge_model->SetModelTaskDef(task);
    }

    TBEKernelStore kernel_store;
    if (compute_graph != nullptr && compute_graph->GetDirectNodesSize() != 0) {
      for (const ge::NodePtr &n : compute_graph->GetDirectNode()) {
        auto node_op_desc = n->GetOpDesc();
	GE_IF_BOOL_EXEC(node_op_desc == nullptr, continue);
	TBEKernelPtr tbe_kernel = node_op_desc->TryGetExtAttr(ge::OP_EXTATTR_NAME_TBE_KERNEL, TBEKernelPtr());
	GE_IF_BOOL_EXEC(tbe_kernel == nullptr, continue);
	kernel_store.AddTBEKernel(tbe_kernel);
	GELOGI("Add tbe kernel bin %s", tbe_kernel->GetName().c_str());
      }
    }
    if (!kernel_store.Build()) {
      GELOGE(FAILED, "TBE Kernels store build failed!");
      return FAILED;
    }
    ge_model->SetTBEKernelStore(kernel_store);
    return SUCCESS;
  }

  static void LoadStandardModelDataLocal(ge::ModelData &data) {
    static const std::string STANDARD_MODEL_DATA_PATH =
        "llt/framework/domi/ut/ome/test/data/standard_partition_model.txt";
    ge::proto::ModelDef model_def;
    ReadProtoFromText(STANDARD_MODEL_DATA_PATH.c_str(), &model_def);

    data.model_len = model_def.ByteSizeLong();
    data.model_data = new uint8_t[data.model_len];
    model_def.SerializePartialToArray(data.model_data, data.model_len);
  }
  static void InitModel(ge::DavinciModel &davinciModel) {
    ge::ModelData data;
    LoadStandardModelDataLocal(data);
    std::shared_ptr<ge::Model> model_ = std::make_shared<ge::Model>();
    ge::Model::Load((uint8_t *)data.model_data, data.model_len, *model_);

    GeModelPtr ge_model;
    TransModelToGeModel(model_, ge_model);
    davinciModel.Assign(ge_model);

    if (data.model_data != nullptr) {
      delete[](uint8_t *) data.model_data;
    }
  }

  static void InitEmptyModel(ge::DavinciModel &davinciModel) {
    auto model = std::make_shared<ge::Model>();
    ge::AttrUtils::SetInt(model, ATTR_MODEL_MEMORY_SIZE, 81000000);
    ge::AttrUtils::SetInt(model, ATTR_MODEL_WEIGHT_SIZE, 4100000);
    ge::AttrUtils::SetInt(model, ATTR_MODEL_STREAM_NUM, 1);
    ge::AttrUtils::SetInt(model, ATTR_MODEL_EVENT_NUM, 1);
    ge::AttrUtils::SetInt(model, MODEL_ATTR_TASK_GEN_BASE_ADDR, 0x123);
    ge::AttrUtils::SetInt(model, MODEL_ATTR_TASK_GEN_WEIGHT_ADDR, 0x456);
    ge::AttrUtils::SetInt(model, ATTR_MODEL_BATCH_NUM, 1);

    //        ge::AttrUtils::SetStr(model, ATTR_MODEL_TARGET_TYPE, "MINI"); // domi::MINI

    auto compute_graph = std::make_shared<ge::ComputeGraph>("graph");
    ge::GeAttrValue::BYTES buffer(4100000, 0);
    ge::AttrUtils::SetBytes(compute_graph, "weights_data", buffer);
    auto graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    model->SetGraph(graph);

    GeModelPtr ge_model;
    TransModelToGeModel(model, ge_model);

    davinciModel.Assign(ge_model);
  }

  static void InitModelWithoutMem(ge::DavinciModel &davinciModel) { InitModel(davinciModel); }

  static Status ModelLoadStub(const uint8_t *data, size_t len, ge::Model &model) {
    auto compute_graph = std::make_shared<ge::ComputeGraph>("graph");
    auto graph = ge::GraphUtils::CreateGraphFromComputeGraph(compute_graph);
    model.SetGraph(graph);
    return SUCCESS;
  }
  static void InitDefaultTensorDesc(ge::GeTensorDesc &tensor_desc) {}
  static void AddInputDesc(ge::OpDescPtr op_desc, vector<int64_t> shape, ge::Format format, ge::DataType dataType,
                           int64_t dataSize = 0) {
    ge::GeTensorDesc tensor_desc(ge::GeShape(shape), format, dataType);
    InitDefaultTensorDesc(tensor_desc);
    ge::TensorUtils::SetSize(tensor_desc, dataSize);
    op_desc->AddInputDesc(tensor_desc);
  }
  static void AddOutputDesc(ge::OpDescPtr op_desc, vector<int64_t> shape, ge::Format format, ge::DataType dataType,
                            int64_t dataSize = 0) {
    ge::GeTensorDesc tensor_desc(ge::GeShape(shape), format, dataType);
    InitDefaultTensorDesc(tensor_desc);
    ge::TensorUtils::SetSize(tensor_desc, dataSize);
    op_desc->AddOutputDesc(tensor_desc);
  }
  static void AddWeight(ge::NodePtr node_ptr, uint8_t *data, size_t dataLen, vector<int64_t> shape = {},
                        ge::Format format = ge::FORMAT_NCHW, ge::DataType dataType = ge::DT_FLOAT) {
    ge::GeTensorDesc tensor_desc(ge::GeShape(shape), format, dataType);

    vector<ge::GeTensorPtr> weigths = ge::OpDescUtils::MutableWeights(node_ptr);
    weigths.push_back(std::make_shared<ge::GeTensor>(tensor_desc, data, dataLen));
    ge::OpDescUtils::SetWeights(node_ptr, weigths);
  }
  static ge::OpDescPtr CreateOpDesc() {
    auto op_desc = std::make_shared<ge::OpDesc>();
    return op_desc;
  }
};

class OmeTestOpDescBuilder {
 public:
  OmeTestOpDescBuilder(ge::OpDescPtr orgOpDesc = nullptr) : orgOpDesc_(orgOpDesc) {
    if (orgOpDesc_) {
      streamId_ = orgOpDesc_->GetStreamId();
    }
  }

  OmeTestOpDescBuilder &SetStreamId(int64_t streamId) {
    streamId_ = streamId;
    return *this;
  }
  OmeTestOpDescBuilder &SetWorkspace(vector<int64_t> workspace) {
    workspace_ = workspace;
    return *this;
  }
  OmeTestOpDescBuilder &SetWorkspaceBytes(vector<int64_t> workspaceBytes) {
    workspaceBytes_ = workspaceBytes;
    return *this;
  }
  OmeTestOpDescBuilder &SetType(const string &type) {
    type_ = type;
    return *this;
  }
  OmeTestOpDescBuilder &SetName(const string &name) {
    name_ = name;
    return *this;
  }
  OmeTestOpDescBuilder &SetInputs(vector<int64_t> inputs) {
    inputsDataOffeset_ = inputs;
    return *this;
  }
  OmeTestOpDescBuilder &AddInput(int64_t input) {
    inputsDataOffeset_.push_back(input);
    return *this;
  }
  OmeTestOpDescBuilder &SetOutputs(vector<int64_t> outputs) {
    outputsDataOffeset_ = outputs;
    return *this;
  }
  OmeTestOpDescBuilder &AddOutput(int64_t output) {
    outputsDataOffeset_.push_back(output);
    return *this;
  }

  OmeTestOpDescBuilder &SetEventId(int64_t eventId) {
    eventId_ = eventId;
    return *this;
  }

  OmeTestOpDescBuilder &Setscopeid(int64_t scopeid) {
    scopeid_ = scopeid;
    return *this;
  }

  ge::GeTensorDesc &AddInputDesc(vector<int64_t> shape, ge::Format format, ge::DataType dataType,
                                 int64_t dataSize = 0) {
    ge::GeTensorDesc tensor_desc(ge::GeShape(shape), format, dataType);
    OmeTestOpUtils::InitDefaultTensorDesc(tensor_desc);
    ge::TensorUtils::SetSize(tensor_desc, dataSize);
    inputTensorDescs.push_back(tensor_desc);
    return inputTensorDescs.back();
  }
  ge::GeTensorDesc &AddInputDesc(vector<int64_t> shape, ge::Format format, ge::DataType dataType, int64_t realdimcnt,
                                 int64_t dataSize) {
    ge::GeTensorDesc tensor_desc(ge::GeShape(shape), format, dataType);
    OmeTestOpUtils::InitDefaultTensorDesc(tensor_desc);
    ge::TensorUtils::SetSize(tensor_desc, dataSize);
    ge::TensorUtils::SetRealDimCnt(tensor_desc, realdimcnt);
    inputTensorDescs.push_back(tensor_desc);
    return inputTensorDescs.back();
  }

  ge::GeTensorDesc &AddOutputDesc(vector<int64_t> shape, ge::Format format, ge::DataType dataType,
                                  int64_t dataSize = 0) {
    ge::GeTensorDesc tensor_desc(ge::GeShape(shape), format, dataType);
    OmeTestOpUtils::InitDefaultTensorDesc(tensor_desc);
    ge::TensorUtils::SetSize(tensor_desc, dataSize);
    outputTensorDescs.push_back(tensor_desc);
    return outputTensorDescs.back();
  }

  ge::GeTensorDesc &AddOutputDesc(vector<int64_t> shape, ge::Format format, ge::DataType dataType, int64_t realdimcnt,
                                  int64_t dataSize) {
    ge::GeTensorDesc tensor_desc(ge::GeShape(shape), format, dataType);
    OmeTestOpUtils::InitDefaultTensorDesc(tensor_desc);
    ge::TensorUtils::SetSize(tensor_desc, dataSize);
    ge::TensorUtils::SetRealDimCnt(tensor_desc, realdimcnt);
    outputTensorDescs.push_back(tensor_desc);
    return outputTensorDescs.back();
  }

  ge::GeTensorPtr AddWeight(uint8_t *data, size_t dataLen, vector<int64_t> shape = {},
                            ge::Format format = ge::FORMAT_NCHW, ge::DataType dataType = ge::DT_FLOAT) {
    ge::GeTensorDesc tensor_desc(ge::GeShape(shape), format, dataType);

    weights_.emplace_back(std::make_shared<ge::GeTensor>(tensor_desc, data, dataLen));
    return weights_.back();
  }
  ge::NodePtr Finish() {
    ge::OpDescPtr op_desc;
    if (orgOpDesc_) {
      op_desc = orgOpDesc_;
    } else {
      op_desc = OmeTestOpUtils::CreateOpDesc();  // std::make_shared<ge::OpDesc>(name_, type_);
    }
    if (!type_.empty()) {
      op_desc->SetType(type_);
    }
    if (!name_.empty()) {
      op_desc->SetName(name_);
    }

    op_desc->SetStreamId(streamId_);
    ge::AttrUtils::SetInt(op_desc, "id", 1);

    if (eventId_ != -1) {
      ge::AttrUtils::SetInt(op_desc, SEND_ATTR_EVENT_ID, eventId_);
    }

    if (scopeid_ != -1) {
      ge::AttrUtils::SetInt(op_desc, "fusion_scope", scopeid_);
    }
    // ge::AttrUtils::SetInt(op_desc, ATTR_NAME_STREAM_ID, streamId_);
    // if(!inputsDataOffeset_.empty())
    {
      vector<int64_t> inputs;
      inputs = op_desc->GetInputOffset();
      inputs.insert(inputs.end(), inputsDataOffeset_.begin(), inputsDataOffeset_.end());

      op_desc->SetInputOffset(inputs);
    }
    // if(!outputsDataOffeset_.empty())
    {
      vector<int64_t> outputs;
      outputs = op_desc->GetOutputOffset();
      outputs.insert(outputs.end(), outputsDataOffeset_.begin(), outputsDataOffeset_.end());

      op_desc->SetOutputOffset(outputs);
    }
    // if(!workspace_.empty())
    {
      vector<int64_t> workspace = op_desc->GetWorkspace();
      workspace.insert(workspace.end(), workspace_.begin(), workspace_.end());

      op_desc->SetWorkspace(workspace);
    }
    // if(!workspaceBytes_.empty())
    {
      vector<int64_t> workspaceBytes;
      workspaceBytes = op_desc->GetWorkspaceBytes();
      workspaceBytes.insert(workspaceBytes.end(), workspaceBytes_.begin(), workspaceBytes_.end());

      op_desc->SetWorkspaceBytes(workspaceBytes);
    }
    for (auto &tensor_desc : inputTensorDescs) {
      op_desc->AddInputDesc(tensor_desc);
    }
    for (auto &tensor_desc : outputTensorDescs) {
      op_desc->AddOutputDesc(tensor_desc);
    }

    static std::shared_ptr<ge::ComputeGraph> graph;
    // clear graph
    graph = std::make_shared<ge::ComputeGraph>("g");

    ge::NodePtr node_op = graph->AddNode(op_desc);
    // for(int i=0; i < inputTensorDescs.size(); i++)
    for (int i = 0; i < op_desc->GetInputsSize(); i++) {
      ge::OpDescPtr src_op_desc = std::make_shared<ge::OpDesc>();

      ge::GeTensorDesc src_out_desc;
      src_op_desc->AddOutputDesc(src_out_desc);

      ge::NodePtr src_node = graph->AddNode(src_op_desc);
      if (nullptr == src_node) {
        GELOGE(ge::FAILED, "Finish: nullptr == src_node");
      }
      Status res = ge::GraphUtils::AddEdge(src_node->GetOutDataAnchor(0), node_op->GetInDataAnchor(i));
      if (SUCCESS != res) {
        GELOGE(ge::FAILED, "Finish: GraphUtils::AddEdge failed");
      }
    }

    {
      vector<ge::GeTensorPtr> weights;
      weights = ge::OpDescUtils::MutableWeights(node_op);
      weights.insert(weights.end(), weights_.begin(), weights_.end());

      ge::OpDescUtils::SetWeights(node_op, weights);
    }

    *this = OmeTestOpDescBuilder(op_desc);  // clear up

    return node_op;
  }

 private:
  ge::OpDescPtr orgOpDesc_;
  int64_t streamId_ = 0;
  string type_;
  string name_;
  vector<int64_t> inputsDataOffeset_;   // input
  vector<int64_t> outputsDataOffeset_;  // output
  vector<ge::GeTensorDesc> inputTensorDescs;
  vector<ge::GeTensorDesc> outputTensorDescs;
  vector<int64_t> workspace_;
  vector<int64_t> workspaceBytes_;
  vector<ge::GeTensorPtr> weights_;
  int64_t eventId_ = -1;
  int64_t scopeid_ = -1;
};

#endif  // OME_REBUILD_OME_OP_TEST_UTILS_H
