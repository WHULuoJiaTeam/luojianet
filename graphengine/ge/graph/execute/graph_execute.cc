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

#include "graph/execute/graph_execute.h"

#include <memory>
#include <string>

#include "graph/load/model_manager/model_manager.h"
#include "graph/load/model_manager/davinci_model.h"
#include "common/profiling/profiling_manager.h"

namespace ge {
using Uint32Pair = pair<uint32_t, uint32_t>;
const uint32_t kInvalidModelId = UINT32_MAX;
GraphExecutor::GraphExecutor()
    : init_flag_(false),
      train_graph_flag_(false),
      sync_run_mutex_(nullptr),
      condition_(nullptr),
      graph_run_listener_(nullptr),
      last_graph_id_(UINT32_MAX),
      malloc_flag_(false) {}

GraphExecutor::~GraphExecutor() {
  outputs_desc_.clear();
  if (malloc_flag_) {
    for (auto &buffer_addr : buffer_addr_) {
      rtError_t rt_ret;
      rt_ret = rtFreeHost(buffer_addr);
      if (rt_ret != RT_ERROR_NONE) {
        REPORT_CALL_ERROR("E19999", "Call rtFreeHost failed, ret:0x%X", rt_ret);
        GELOGE(RT_FAILED, "[Call][RtFreeHost] subgraph free buffer failed, ret: 0x%X", rt_ret);
      }
    }
  }
  malloc_flag_ = false;
  buffer_addr_.clear();
}

Status GraphExecutor::SetCondition(std::mutex *mutex, std::condition_variable *cond,
                                   std::shared_ptr<GraphModelListener> listener) {
  if (mutex == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param mutex nullptr");
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[Check][Param] input param mutex is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }
  if (cond == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param cond nullptr");
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[Check][Param] input param cond is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }
  if (listener == nullptr) {
    REPORT_INNER_ERROR("E19999", "Check param listener nullptr");
    GELOGE(GE_GRAPH_PARAM_NULLPTR, "[Check][Param] input param listener is nullptr.");
    return GE_GRAPH_PARAM_NULLPTR;
  }

  sync_run_mutex_ = mutex;
  condition_ = cond;

  graph_run_listener_ = listener;

  init_flag_ = true;

  return SUCCESS;
}

Status GraphExecutor::SetDynamicSize(uint32_t model_id, const std::vector<uint64_t> &batch_num, int32_t dynamic_type) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->SetDynamicSize(model_id, batch_num, dynamic_type);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Set][DynamicSize] failed, model_id:%u", model_id);
    return ret;
  }
  return SUCCESS;
}

void GraphExecutor::SetTrainFlag(bool is_train_graph) { train_graph_flag_ = is_train_graph; }

Status GraphExecutor::FreeInOutBuffer() {
  if (malloc_flag_) {
    for (auto iter = buffer_addr_.begin(); iter != buffer_addr_.end(); ++iter) {
      rtError_t rt_ret;
      rt_ret = rtFreeHost(*iter);
      if (rt_ret != RT_ERROR_NONE) {
        REPORT_CALL_ERROR("E19999", "Call rtFreeHost failed, ret:0x%X", rt_ret);
        GELOGE(RT_FAILED, "[Call][RtFreeHost] subgraph free buffer failed, ret: 0x%X", rt_ret);
        (void)buffer_addr_.erase(buffer_addr_.begin(), iter);
        return GE_GRAPH_FREE_FAILED;
      }
    }
    buffer_addr_.clear();

    malloc_flag_ = false;
    return SUCCESS;
  } else {
    GELOGD("[GraphManager] not malloc buffer.");
    return SUCCESS;
  }
}

Status GraphExecutor::MallocInOutBuffer(const std::vector<uint64_t> &buffer_size, std::vector<void *> &data_addr) {
  if (malloc_flag_) {
    auto all_size_same = true;
    if (buffer_size.size() == buffer_size_.size()) {
      for (size_t i = 0; i < buffer_size.size(); i++) {
        if (buffer_size[i] != buffer_size_[i]) {
          all_size_same = false;
          break;
        }
      }
    } else {
      all_size_same = false;
    }
    if (all_size_same) {
      data_addr = buffer_addr_;
      return SUCCESS;
    }
    buffer_size_.clear();
    auto rt_ret = FreeInOutBuffer();
    if (rt_ret != SUCCESS) {
      GELOGE(RT_FAILED, "[Free][Buffer] failed, ret: 0x%X", rt_ret);
      return RT_FAILED;
    }
  }

  rtError_t rt_ret;
  for (size_t i = 0; i < buffer_size.size(); ++i) {
    void *tmp_buf = nullptr;
    rt_ret = rtMallocHost(&tmp_buf, buffer_size[i]);
    if (rt_ret != RT_ERROR_NONE) {
      REPORT_CALL_ERROR("E19999", "Call rtMallocHost failed, size:%lu, ret:0x%X", buffer_size[i], rt_ret);
      GELOGE(RT_FAILED, "[Malloc][Buffer] failed, size:%lu, ret:0x%X", buffer_size[i], rt_ret);
      return GE_GRAPH_MALLOC_FAILED;
    }
    malloc_flag_ = true;
    data_addr.push_back(tmp_buf);
    buffer_addr_.push_back(tmp_buf);
  }
  buffer_size_ = buffer_size;
  return SUCCESS;
}

Status GraphExecutor::PrepareInputData(const std::vector<GeTensor> &input_tensor, InputData &graph_input_data,
                                       OutputData &graph_output_data, std::vector<InputOutputDescInfo> &output_desc) {
  // Preprocessing input data
  graph_input_data.index = 0;
  graph_input_data.timeout = 0;
  graph_input_data.timestamp = 0;
  std::size_t inputSize = input_tensor.size();
  std::size_t output_size = output_desc.size();
  std::vector<uint64_t> bufferSizeVec;
  std::vector<void *> addrVec;

  for (std::size_t i = 0; i < inputSize; ++i) {
    const GeTensor *InTensor = &input_tensor[i];
    GE_CHECK_NOTNULL(InTensor);
    bufferSizeVec.push_back(InTensor->GetData().size());
  }

  for (const auto &desc : output_desc) {
    bufferSizeVec.push_back(desc.size);
  }

  Status ret = MallocInOutBuffer(bufferSizeVec, addrVec);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_MALLOC_FAILED, "[Malloc][Mem] failed");
    return GE_GRAPH_MALLOC_FAILED;
  }

  for (std::size_t i = 0; i < input_tensor.size() && i < addrVec.size(); ++i) {
    const GeTensor *in_tensor = &input_tensor[i];
    GE_CHECK_NOTNULL(in_tensor);
    if ((addrVec[i] != nullptr) && (in_tensor->GetData().data() != nullptr)) {
      rtError_t rt_ret = rtMemcpy(addrVec[i], bufferSizeVec[i], in_tensor->GetData().data(),
                                  in_tensor->GetData().size(), RT_MEMCPY_HOST_TO_HOST);
      if (rt_ret != RT_ERROR_NONE) {
        REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, dst_size:%lu, src_size:%zu, ret:0x%X",
                          bufferSizeVec[i], in_tensor->GetData().size(), rt_ret);
        GELOGE(RT_FAILED, "[Call][RtMemcpy] failed, dst_size:%lu, src_size:%zu, ret:0x%X",
               bufferSizeVec[i], in_tensor->GetData().size(), rt_ret);
        return RT_FAILED;
      }
    }

    DataBuffer in_data_buf;
    in_data_buf.data = reinterpret_cast<uint8_t *>(addrVec[i]);
    in_data_buf.length = in_tensor->GetData().size();
    in_data_buf.isDataSupportMemShare = false;
    graph_input_data.blobs.push_back(in_data_buf);
  }

  graph_output_data.index = 0;

  for (std::size_t j = 0; j < output_size; j++) {
    auto desc = output_desc[j];
    uint64_t buffer_size = desc.size;

    DataBuffer out_data_buf;
    out_data_buf.data = reinterpret_cast<uint8_t *>(addrVec[inputSize + j]);
    out_data_buf.length = buffer_size;
    out_data_buf.isDataSupportMemShare = false;
    graph_output_data.blobs.push_back(out_data_buf);
  }

  return SUCCESS;
}

Status GraphExecutor::SyncExecuteModel(uint32_t model_id, const std::vector<GeTensor> &input_tensor,
                                       std::vector<GeTensor> &output_tensor) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  if (model_manager->IsDynamicShape(model_id)) {
    GELOGI("[ExecuteGraph] GetInputOutputDescInfo via dynamic shape model executor, modelId=%u", model_id);
    return model_manager->SyncExecuteModel(model_id, input_tensor, output_tensor);
  }

  // Prepare input and output
  std::vector<InputOutputDescInfo> inputs_desc;
  std::vector<InputOutputDescInfo> output_desc;

  GELOGI("[ExecuteGraph] GetInputOutputDescInfo via new ome begin.");
  Status ret = GetInputOutputDescInfo(model_id, inputs_desc, output_desc);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_GET_IN_OUT_FAILED, "[Get][InputOutputDescInfo] failed, modelId=%u.", model_id);
    return GE_GRAPH_GET_IN_OUT_FAILED;
  }
  outputs_desc_.assign(output_desc.begin(), output_desc.end());

  InputData input_data;
  OutputData output_data;
  input_data.model_id = model_id;
  ret = PrepareInputData(input_tensor, input_data, output_data, output_desc);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_PREPARE_FAILED, "[Prepare][InputData] failed, modelId=%u.", model_id);
    return GE_GRAPH_PREPARE_FAILED;
  }

  if (graph_run_listener_->ResetResult() != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Call graph_run_listener_.ResetResult fail, model_id:%u", model_id);
    GELOGE(GE_GRAPH_EXECUTE_FAILED, "[Reset][Result] failed, model_id:%u", model_id);
    return GE_GRAPH_EXECUTE_FAILED;
  }

  // Run mode async
  GELOGI("[ExecuteGraph] DataInput via new ome begin.");
  ret = DataInput(input_data, output_data);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_DATA_INPUT_FAILED, "[Call][DataInput] push data failed, modelId=%u.", model_id);
    return GE_GRAPH_DATA_INPUT_FAILED;
  }
  GELOGI("[GraphExecutor] input data push to wrapper finish, waiting for result...");

  // Pending until async execute graph complete
  {
    std::unique_lock<std::mutex> ulock(*sync_run_mutex_);
    if (!graph_run_listener_->IsFinished()) {
      (*condition_).wait(ulock);
    }

    // Run graph return
    uint32_t result_code = graph_run_listener_->GetResultCode();
    if (result_code != SUCCESS && result_code != END_OF_SEQUENCE) {
      REPORT_CALL_ERROR("E19999", "Graph_run_listener_ run fail, result:%u, model_id:%u", result_code, model_id);
      GELOGE(GE_GRAPH_EXECUTE_FAILED, "[Execute][Model] failed, ret=%u, modelId=%u.", result_code, model_id);
      return GE_GRAPH_EXECUTE_FAILED;
    }
  }
  for (size_t i = 0; i < output_data.blobs.size(); ++i) {
    DataBuffer outputDataTmp = output_data.blobs[i];
    CHECK_FALSE_EXEC(outputDataTmp.length != 0,
                     REPORT_INNER_ERROR("E19999", "Param output_data.length is 0 in model:%u, check invalid",
                                        model_id);
                     GELOGE(GE_GRAPH_EXECUTE_FAILED, "[Check][Param] Failed to allocate memory, "
                            "length is 0, model id:%u", model_id);
                     return GE_GRAPH_EXECUTE_FAILED);
    std::unique_ptr<uint8_t> outBufTmp(new (std::nothrow) uint8_t[outputDataTmp.length]);
    if (outBufTmp == nullptr) {
      REPORT_CALL_ERROR("E19999", "New output buffer fail, length:%lu, model:%u", outputDataTmp.length, model_id);
      GELOGE(FAILED, "[Allocate][Memory] failed, length:%lu, model:%u", outputDataTmp.length, model_id);
      return FAILED;
    }
    GE_PRINT_DYNAMIC_MEMORY(new, "the output memory of data on training.", sizeof(uint8_t) * outputDataTmp.length)
    rtError_t ret_value = rtMemcpy(outBufTmp.get(), outputDataTmp.length, outputDataTmp.data, outputDataTmp.length,
                                   RT_MEMCPY_HOST_TO_HOST);
    CHECK_FALSE_EXEC(ret_value == RT_ERROR_NONE,
                     REPORT_CALL_ERROR("E19999", "Call rtMemcpy failed, dst_size:%lu, src_size:%zu, ret:0x%X",
                                       outputDataTmp.length, outputDataTmp.length, ret_value);
                     GELOGE(GE_GRAPH_EXECUTE_FAILED, "[Call][RtMemcpy] failed, dst_size:%lu, src_size:%zu, ret:0x%X",
                            outputDataTmp.length, outputDataTmp.length, ret_value);
                     return GE_GRAPH_EXECUTE_FAILED);
    GeTensor outTensor;
    std::vector<int64_t> shapeDims;
    for (const auto &dim : output_desc[i].shape_info.dims) {
      shapeDims.push_back(dim);
    }

    GeShape outShape(shapeDims);
    outTensor.MutableTensorDesc().SetShape(outShape);
    outTensor.MutableTensorDesc().SetDataType((DataType)output_desc[i].data_type);
    (void)outTensor.SetData(outBufTmp.get(), outputDataTmp.length);
    output_tensor.push_back(outTensor);
  }

  GELOGI("[GraphExecutor] execute model success, modelId=%u.", model_id);

  return SUCCESS;
}

void GraphExecutor::InitModelIdInfo(std::vector<uint32_t> &out_model_id_info,
                                    std::vector<SubGraphInfoPtr> &sub_graph_vec, uint32_t output_size) {
  for (uint32_t i = 0; i < output_size; i++) {
    for (size_t j = 0; j < sub_graph_vec.size(); j++) {
      if (sub_graph_vec[j]->GetOutputFlag().size() == output_size && sub_graph_vec[j]->GetOutputFlag().at(i)) {
        out_model_id_info.push_back(sub_graph_vec[j]->GetModelIdInfo().model_id);
      }
    }
  }
}

Status GraphExecutor::FreeExecuteMemory() {
  auto ret = FreeInOutBuffer();
  if (ret != SUCCESS) {
    GELOGE(ret, "[Free][InOutBuffer] Error!");
    return ret;
  }

  return SUCCESS;
}

Status GraphExecutor::ExecuteGraph(GraphId graph_id, const GeRootModelPtr &ge_root_model,
                                   const std::vector<GeTensor> &input_tensor, std::vector<GeTensor> &output_tensor) {
  if (graph_id != last_graph_id_) {
    auto ret = FreeExecuteMemory();
    if (ret != SUCCESS) {
      return ret;
    }
  }
  last_graph_id_ = graph_id;

  if (!init_flag_) {
    REPORT_INNER_ERROR("E19999", "No SetCondition called before, graph:%u, check invalid",
                       graph_id);
    GELOGE(GE_GRAPH_EXECUTE_NOT_INIT, "[Check][Param] AI Core Engine without calling SetCondition! graph id:%u",
           graph_id);
    return GE_GRAPH_EXECUTE_NOT_INIT;
  }
  GE_CHECK_NOTNULL_EXEC(ge_root_model, return FAILED);
  Status ret = SyncExecuteModel(ge_root_model->GetModelId(), input_tensor, output_tensor);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_SYNC_MODEL_FAILED, "[SyncExecute][Model] Error! graph id:%u", graph_id);
    return GE_GRAPH_SYNC_MODEL_FAILED;
  }
  ret = ModelSubscribe(graph_id);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][ModelSubscribe] failed, graph_id:%u", graph_id);
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::ExecuteGraphAsync(GraphId graph_id, const GeRootModelPtr &ge_root_model,
                                        const std::vector<ge::Tensor> &input_tensor,
                                        const RunAsyncCallback& callback) {
  GELOGI("[GraphExecutor] Start to async execute graph, graph_id=%u", graph_id);
  if (graph_id != last_graph_id_) {
    auto ret = FreeExecuteMemory();
    if (ret != SUCCESS) {
      return ret;
    }
  }
  last_graph_id_ = graph_id;
  GE_CHECK_NOTNULL_EXEC(ge_root_model, return FAILED);
  Status ret = AsyncExecuteModel(ge_root_model, input_tensor, callback);
  if (ret != SUCCESS) {
    GELOGE(GE_GRAPH_SYNC_MODEL_FAILED, "[AsyncExecute][Model] Error! graph id:%u", graph_id);
    return GE_GRAPH_SYNC_MODEL_FAILED;
  }

  GELOGI("[GraphExecutor] Async execute graph success, graph_id=%u", graph_id);
  return SUCCESS;
}

Status GraphExecutor::GetExecuteData(const std::vector<GeTensor> &input_tensor, std::vector<DataBuffer> &blobs,
                                     std::vector<GeTensorDesc> &tensor_desc) {
  for (const auto &tensor : input_tensor) {
    DataBuffer in_data_buf;
    // check placement
    in_data_buf.data = const_cast<uint8_t *>(tensor.GetData().data());
    in_data_buf.length = tensor.GetData().size();
    in_data_buf.isDataSupportMemShare = false;
    blobs.emplace_back(in_data_buf);
    tensor_desc.emplace_back(tensor.GetTensorDesc());
  }
  return SUCCESS;
}

Status GraphExecutor::ExecuteGraphWithStream(GraphId graph_id,
                                             rtStream_t stream,
                                             const GeRootModelPtr &ge_root_model,
                                             const std::vector<GeTensor> &input_tensor,
                                             std::vector<GeTensor> &output_tensor) {
  GELOGI("[GraphExecutor] Start to execute graph with stream, graph id = %u, stream = %p.", graph_id, stream);
  if (!init_flag_) {
    REPORT_INNER_ERROR("E19999", "No SetCondition called before, graph id = %u, stream = %p, check invalid.",
                       graph_id, stream);
    GELOGE(GE_GRAPH_EXECUTE_NOT_INIT, "[Check][Param] AI Core Engine without calling SetCondition! graph id = %u",
           graph_id);
    return GE_GRAPH_EXECUTE_NOT_INIT;
  }

  if (graph_id != last_graph_id_) {
    auto ret = FreeExecuteMemory();
    if (ret != SUCCESS) {
      return ret;
    }
  }
  last_graph_id_ = graph_id;

  GE_CHECK_NOTNULL_EXEC(ge_root_model, return FAILED);
  auto model_id = ge_root_model->GetModelId();
  InputData input_data;
  input_data.index = 0;
  input_data.model_id = model_id;
  std::vector<GeTensorDesc> input_desc;
  auto ret = GetExecuteData(input_tensor, input_data.blobs, input_desc);
  if (ret != SUCCESS) {
    return ret;
  }
  OutputData output_data;
  output_data.index = 0;
  output_data.model_id = model_id;
  std::vector<GeTensorDesc> output_desc;
  ret = GetExecuteData(output_tensor, output_data.blobs, output_desc);
  if (ret != SUCCESS) {
    return ret;
  }

  auto async_mode = true;
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  ret = model_manager->ExecuteModel(model_id, stream, async_mode, input_data, input_desc, output_data, output_desc);
  if (ret != SUCCESS) {
    return ret;
  }

  GELOGI("[GraphExecutor] Async execute graph with stream success graph id = %u, stream = %p.", graph_id, stream);
  return SUCCESS;
}

bool CompareByLoad(const Uint32Pair &lhs, const Uint32Pair &rhs) {
  return lhs.second < rhs.second;
}

uint32_t GraphExecutor::GetExecuteModelId(const GeRootModelPtr &ge_root_model) {
  std::vector<uint32_t> model_ids = ge_root_model->GetAllModelId();
  if (model_ids.empty()) {
    return kInvalidModelId;
  }
  if (model_ids.size() == 1) {
    return ge_root_model->GetModelId();
  }
  std::vector<Uint32Pair> model_id_to_loads;
  auto model_manager = ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  for (auto model_id : model_ids) {
    auto davinci_model = model_manager->GetModel(model_id);
    auto hybrid_model = model_manager->GetHybridModel(model_id);
    if (hybrid_model == nullptr) {
      GE_CHECK_NOTNULL(davinci_model);
    }
    uint32_t input_load = hybrid_model != nullptr ? hybrid_model->GetDataInputerSize() :
                                                    davinci_model->GetDataInputerSize();
    uint32_t running_load = hybrid_model != nullptr ? static_cast<uint32_t>(hybrid_model->GetRunningFlag()) :
                                                      static_cast<uint32_t>(davinci_model->GetRunningFlag());
    uint32_t load = input_load + running_load;
    if (load == 0) {
      return model_id;
    }
    model_id_to_loads.emplace_back(model_id, load);
  }
  sort(model_id_to_loads.begin(), model_id_to_loads.end(), CompareByLoad);
  if (model_id_to_loads.empty()) {
    return kInvalidModelId;
  }
  return model_id_to_loads.begin()->first;
}

Status GraphExecutor::SetCallback(uint32_t model_id, const GeRootModelPtr &ge_root_model,
                                  const RunAsyncCallback &callback) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  if (model_manager->IsNeedHybridLoad(*ge_root_model)) {
    auto model = model_manager->GetHybridModel(model_id);
    GE_CHECK_NOTNULL(model);
    if (model->SetRunAsyncListenerCallback(callback) != SUCCESS) {
      GELOGE(FAILED, "[Set][RunAsyncListenerCallback] failed, model_id %u", model_id);
      return FAILED;
    }
  } else {
    auto model = model_manager->GetModel(model_id);
    GE_CHECK_NOTNULL(model);
    if (model->SetRunAsyncListenerCallback(callback) != SUCCESS) {
      GELOGE(FAILED, "[Set][RunAsyncListenerCallback] failed, model_id %u", model_id);
      return FAILED;
    }
  }
  return SUCCESS;
}

Status GraphExecutor::AsyncExecuteModel(const GeRootModelPtr &ge_root_model, const std::vector<ge::Tensor> &inputs,
                                        const RunAsyncCallback &callback) {
  uint32_t model_id = GetExecuteModelId(ge_root_model);
  if (model_id == kInvalidModelId) {
    GELOGE(INTERNAL_ERROR, "No valid model id.");
    return INTERNAL_ERROR;
  }
  try {
    auto model_manager = ge::ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    GELOGI("RunAsync begin.model_id %u", model_id);
    if (SetCallback(model_id, ge_root_model, callback) != SUCCESS) {
      GELOGE(FAILED, "[Set][CallBack] for model fail, model_id %u", model_id);
      return FAILED;
    }

    Status ret = model_manager->DataInputTensor(model_id, inputs);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Call][DataInputTensor] RunAsync: DataInput fail, model_id %u", model_id);
      return ret;
    }

    GELOGI("RunAsync success.");
  } catch (std::bad_alloc &) {
    REPORT_INNER_ERROR("E19999", "Bad memory allocation exception occur failed, model_id %u", model_id);
    GELOGE(MEMALLOC_FAILED, "[Run][Async] failed, bad memory allocation occur, model_id %u", model_id);
    return MEMALLOC_FAILED;
  } catch (...) {
    REPORT_INNER_ERROR("E19999", "Some exceptions occur failed, model_id %u", model_id);
    GELOGE(FAILED, "[Run][Async] failed, some exceptions occur, model_id %u", model_id);
    return FAILED;
  }

  return SUCCESS;
}

Status GraphExecutor::DataInput(const InputData &input_data, OutputData &output_data) {
  try {
    auto model_manager = ge::ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    Status ret = model_manager->DataInput(input_data, output_data);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Call][DataInput] failed.");
      return ret;
    }
  } catch (std::bad_alloc &) {
    REPORT_INNER_ERROR("E19999", "Bad memory allocation exception occur failed");
    GELOGE(MEMALLOC_FAILED, "[Call][DataInput] failed, bad memory allocation occur !");
    return MEMALLOC_FAILED;
  } catch (...) {
    REPORT_INNER_ERROR("E19999", "Some exceptions occur failed");
    GELOGE(FAILED, "[Call][DataInput] failed, some exceptions occur !");
    return FAILED;
  }

  return SUCCESS;
}

Status GraphExecutor::GetInputOutputDescInfo(const uint32_t model_id, vector<InputOutputDescInfo> &input_desc,
                                             vector<InputOutputDescInfo> &output_desc) {
  try {
    auto model_manager = ge::ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    Status ret = model_manager->GetInputOutputDescInfo(model_id, input_desc, output_desc);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Get][InputOutputDescInfo] failed, model_id:%u.", model_id);
      return ret;
    }
  } catch (std::bad_alloc &) {
    REPORT_INNER_ERROR("E19999", "Bad memory allocation exception occur failed, model_id:%u.", model_id);
    GELOGE(MEMALLOC_FAILED, "[Get][InputOutputDescInfo] failed, bad memory allocation occur, model_id:%u.", model_id);
    return MEMALLOC_FAILED;
  } catch (...) {
    REPORT_INNER_ERROR("E19999", "Some exceptions occur failed, model_id:%u.", model_id);
    GELOGE(FAILED, "[Get][InputOutputDescInfo] failed, some exceptions occur, model_id:%u.", model_id);
    return FAILED;
  }

  return SUCCESS;
}

Status GraphExecutor::GetInputOutputDescInfo(const uint32_t model_id, vector<InputOutputDescInfo> &input_desc,
                                             vector<InputOutputDescInfo> &output_desc,
                                             std::vector<uint32_t> &input_formats, std::vector<uint32_t> &out_formats,
                                             bool new_model_desc) {
  try {
    auto model_manager = ge::ModelManager::GetInstance();
    GE_CHECK_NOTNULL(model_manager);
    Status ret = model_manager->GetInputOutputDescInfo(model_id, input_desc, output_desc, input_formats, out_formats,
                                                       new_model_desc);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Get][InputOutputDescInfo] failed, model_id:%u.", model_id);
      return ret;
    }
  } catch (std::bad_alloc &) {
    REPORT_INNER_ERROR("E19999", "Bad memory allocation exception occur failed, model_id:%u.", model_id);
    GELOGE(MEMALLOC_FAILED, "[Get][InputOutputDescInfo] failed, bad memory allocation occur, model_id:%u.", model_id);
    return MEMALLOC_FAILED;
  } catch (...) {
    REPORT_INNER_ERROR("E19999", "Some exceptions occur failed, model_id:%u.", model_id);
    GELOGE(FAILED, "[Get][InputOutputDescInfo] failed, some exceptions occur, model_id:%u.", model_id);
    return FAILED;
  }

  return SUCCESS;
}
///
/// @ingroup ge
/// @brief Get dynamic batch_info
/// @param [in] model_id
/// @param [out] batch_info
/// @param [out] dynamic_type
/// @return execute result
///
Status GraphExecutor::GetDynamicBatchInfo(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info,
                                          int32_t &dynamic_type) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetDynamicBatchInfo(model_id, batch_info, dynamic_type);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][DynamicBatchInfo] failed, model_id:%u.", model_id);
    return ret;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get combined dynamic dims info
/// @param [in] model_id
/// @param [out] batch_info
/// @return execute result
///
Status GraphExecutor::GetCombinedDynamicDims(uint32_t model_id, std::vector<std::vector<int64_t>> &batch_info) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetCombinedDynamicDims(model_id, batch_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Call][GetCombinedDynamicDims] failed, model_id:%u.", model_id);
    return ret;
  }
  return SUCCESS;
}

///
/// @ingroup ge
/// @brief Get user designate shape order
/// @param [in] model_id
/// @param [out] user_input_shape_order
/// @return execute result
///
ge::Status GraphExecutor::GetUserDesignateShapeOrder(uint32_t model_id,
                                                     std::vector<std::string> &user_input_shape_order) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetUserDesignateShapeOrder(model_id, user_input_shape_order);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][UserDesignateShapeOrder] failed, model_id:%u.", model_id);
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetCurShape(const uint32_t model_id, std::vector<int64_t> &batch_info, int32_t &dynamic_type) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetCurShape(model_id, batch_info, dynamic_type);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][CurShape] failed, model_id:%u", model_id);
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetOpAttr(uint32_t model_id, const std::string &op_name, const std::string &attr_name,
                                std::string &attr_value) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetOpAttr(model_id, op_name, attr_name, attr_value);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][OpAttr]Get op:%s attr:%s failed.", op_name.c_str(), attr_name.c_str());
    REPORT_CALL_ERROR("E19999", "Get op:%s attr:%s failed.", op_name.c_str(), attr_name.c_str());
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetModelAttr(uint32_t model_id, std::vector<string> &dynamic_output_shape_info) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetModelAttr(model_id, dynamic_output_shape_info);
  if (ret != SUCCESS) {
    GELOGE(FAILED, "[Get][ModelAttr] failed, model_id:%u", model_id);
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetAippInfo(uint32_t model_id, uint32_t index, AippConfigInfo &aipp_info) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetAippInfo(model_id, index, aipp_info);
  if (ret != SUCCESS) {
    GELOGW("GetAIPPInfo is not success.");
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetAippType(uint32_t model_id, uint32_t index, InputAippType &type, size_t &aipp_index) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetAippType(model_id, index, type, aipp_index);
  if (ret != SUCCESS) {
    GELOGW("Get aipp type is not success.");
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetOrigInputInfo(uint32_t model_id, uint32_t index, OriginInputInfo &orig_input_info) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetOrigInputInfo(model_id, index, orig_input_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][OrigInputInfo] failed, model_id:%u, index:%u.", model_id, index);
    return ret;
  }

  return SUCCESS;
}

Status GraphExecutor::GetAllAippInputOutputDims(uint32_t model_id, uint32_t index,
                                                std::vector<InputOutputDims> &input_dims,
                                                std::vector<InputOutputDims> &output_dims) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetAllAippInputOutputDims(model_id, index, input_dims, output_dims);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][AllAippInputOutputDims] failed, model_id:%u, index:%u.", model_id, index);
    return ret;
  }

  return SUCCESS;
}

Status GraphExecutor::GetOpDescInfo(uint32_t device_id, uint32_t stream_id, uint32_t task_id,
                                    OpDescInfo &op_desc_info) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  Status ret = model_manager->GetOpDescInfo(device_id, stream_id, task_id, op_desc_info);
  if (ret != SUCCESS) {
    GELOGE(ret, "[Get][OpDescInfo] failed, device_id:%u, stream_id:%u, task_id:%u.",
           device_id, stream_id, task_id);
    return ret;
  }
  return SUCCESS;
}

Status GraphExecutor::GetModelByID(uint32_t model_id, std::shared_ptr<DavinciModel> &davinci_model) {
  auto model_manager = ge::ModelManager::GetInstance();
  GE_CHECK_NOTNULL(model_manager);
  davinci_model = model_manager->GetModel(static_cast<uint32_t>(model_id));
  if (davinci_model == nullptr) {
    REPORT_INNER_ERROR("E19999", "GetModel from model_manager fail, model_id:%u", model_id);
    GELOGE(ge::FAILED, "[Get][Model] failed, Model id:%d is invaild or model is not loaded.", model_id);
    return ge::FAILED;
  }
  return ge::SUCCESS;
}

Status GraphExecutor::ModelSubscribe(uint32_t graph_id) {
  auto &profiling_manager = ProfilingManager::Instance();
  const auto &subcribe_info = profiling_manager.GetSubscribeInfo();
  if (subcribe_info.is_subscribe) {
    std::shared_ptr<DavinciModel> davinci_model = nullptr;
    uint32_t model_id = 0;
    Status ret = profiling_manager.GetModelIdFromGraph(graph_id, model_id);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Call][GetModelIdFromGraph] failed, graph_id:%u", graph_id);
      return ret;
    }
    ret = GetModelByID(model_id, davinci_model);
    if (ret != SUCCESS) {
      GELOGE(ret, "[Call][GetModelByID] failed, model_id:%u", model_id);
      return ret;
    }
    ret = profiling_manager.ProfModelSubscribe(subcribe_info.prof_switch, davinci_model.get());
    if (ret != SUCCESS) {
      GELOGE(ret, "[Call][ProfModelSubscribe] failed");
      return ret;
    }
  }
  return SUCCESS;
}
}  // namespace ge
