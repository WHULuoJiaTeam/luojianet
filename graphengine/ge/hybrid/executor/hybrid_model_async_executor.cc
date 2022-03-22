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

#include "hybrid/executor/hybrid_model_async_executor.h"
#include "graph/load/model_manager/model_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/type_utils.h"
#include "graph/ge_context.h"
#include "external/graph/types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/manager/graph_caching_allocator.h"
#include "graph/manager/graph_mem_allocator.h"
#include "graph/manager/rdma_pool_allocator.h"
#include "graph/manager/host_mem_allocator.h"
#include "graph/manager/graph_mem_manager.h"

namespace ge {
namespace hybrid {
namespace {
const int kDataOutputIndex = 0;
const size_t kMinimumPiplineStages = 2;
const int kDefaultLoopCount = 10;
const size_t kAlignment = 64;
}
HybridModelAsyncExecutor::HybridModelAsyncExecutor(HybridModel *model)
    : model_(model), run_flag_(false), data_dumper_(nullptr) {
}

HybridModelAsyncExecutor::~HybridModelAsyncExecutor() {
  if (stream_ != nullptr) {
    GE_CHK_RT(rtStreamDestroy(stream_));
  }
}

void HybridModelAsyncExecutor::SetDeviceId(uint32_t device_id) {
  device_id_ = device_id;
}

void HybridModelAsyncExecutor::SetModelId(uint32_t model_id) {
  model_id_ = model_id;
}

Status HybridModelAsyncExecutor::EnqueueData(const shared_ptr<InputDataWrapper> &data) {
  if (data_inputer_->Push(data) != SUCCESS) {
    REPORT_CALL_ERROR("E19999", "Data queue is full, please call again later, model_id %u.", model_id_);
    GELOGE(domi::DATA_QUEUE_ISFULL,
        "[Push][Data] Data queue is full, please call again later, model_id %u ", model_id_);
    return domi::DATA_QUEUE_ISFULL;
  }
  GELOGD("EnqueueData successfully. model_id = %u, data_index = %u", data->GetInput().model_id, data->GetInput().index);
  return SUCCESS;
}

Status HybridModelAsyncExecutor::Start(const std::shared_ptr<ModelListener> &listener) {
  GELOGD("HybridModelExecutor::Start IN, has listener = %d", listener != nullptr);
  std::lock_guard<std::mutex> lk(mu_);
  if (run_flag_) {
    REPORT_INNER_ERROR("E19999", "Model already started, model_id:%u.", model_id_);
    GELOGE(INTERNAL_ERROR, "[Check][RunState] Model already started, model_id:%u.", model_id_);
    return INTERNAL_ERROR;
  }
  run_flag_ = true;
  listener_ = listener;
  future_ = std::async(std::launch::async, [&]() -> Status {
    GetThreadLocalContext() = *executor_->GetContext()->ge_context;
    GetContext().SetSessionId(executor_->GetContext()->session_id);
    GetContext().SetContextId(executor_->GetContext()->context_id);
    return RunInternal();
  });

  GE_CHK_BOOL_RET_STATUS(future_.valid(), INTERNAL_ERROR,
                         "[Check][RunState] Failed to start, model_id:%u.", model_id_);
  GELOGD("HybridModelExecutor::Start successfully");
  return SUCCESS;
}

Status HybridModelAsyncExecutor::Stop() {
  std::lock_guard<std::mutex> lk(mu_);
  run_flag_ = false;
  data_inputer_->Stop();

  Status ret = SUCCESS;
  if (future_.valid()) {
    ret = future_.get();
  }

  if (is_op_debug_reg_) {
    op_debug_register_.UnregisterDebugForStream(stream_);
  }

  if (stream_ != nullptr) {
    GE_CHK_RT(rtStreamDestroy(stream_));
    stream_ = nullptr;
  }

  return ret;
}

Status HybridModelAsyncExecutor::Init() {
  data_inputer_ = std::unique_ptr<DataInputer>(new(std::nothrow) DataInputer());
  GE_CHECK_NOTNULL(data_inputer_);
  GE_CHK_RT_RET(rtStreamCreate(&stream_, RT_STREAM_PRIORITY_DEFAULT));

  executor_ = std::unique_ptr<HybridModelExecutor>(new(std::nothrow) HybridModelExecutor(model_, device_id_, stream_));
  GE_CHECK_NOTNULL(executor_);
  GE_CHK_STATUS_RET(executor_->Init(),
                    "[Init][HybridModelExecutor] failed, model_id:%u.", model_id_);
  GE_CHK_STATUS_RET(DumpOpDebug(), "[Dump][OpDebug] failed, model_id:%u.", model_id_);

  GELOGI("HybridModel stage nums:%zu", model_->GetRootGraphItem()->NumGroups());
  if (model_->GetRootGraphItem()->NumGroups() >= kMinimumPiplineStages) {
    pipe_executor_ =
        std::unique_ptr<HybridModelPipelineExecutor>(new(std::nothrow) HybridModelPipelineExecutor(model_, device_id_));
    GE_CHECK_NOTNULL(pipe_executor_);
    GE_CHK_STATUS_RET(pipe_executor_->Init(),
                      "[Init][HybridModelPipelineExecutor] failed, model_id:%u.", model_id_);
  }

  GE_CHK_STATUS_RET(InitInputDesc(), "[Init][InputDesc] failed, model_id:%u.", model_id_);

  return SUCCESS;
}

Status HybridModelAsyncExecutor::PreRun(InputData &current_data, HybridModelExecutor::ExecuteArgs &args) {
  GE_CHK_STATUS_RET(SyncVarData(), "[Invoke][SyncVarData] failed, model_id:%u.", model_id_);
  RECORD_MODEL_EXECUTION_EVENT(executor_->GetContext(), "[SyncVarData] End");
  GE_CHK_STATUS_RET(PrepareInputs(current_data, args),
                    "[Invoke][PrepareInputs] failed to copy input data to model, model_id:%u.", model_id_);
  RECORD_MODEL_EXECUTION_EVENT(executor_->GetContext(), "[CopyInputData] End");
  return SUCCESS;
}

Status HybridModelAsyncExecutor::RunInternal() {
  auto device_id = static_cast<int32_t>(device_id_);
  GELOGD("Hybrid model start. model_id = %u, device_id = %u", model_id_, device_id_);
  GE_CHK_RT_RET(rtSetDevice(device_id));
  // DeviceReset before thread run finished!
  GE_MAKE_GUARD(not_used_var, [&] { GE_CHK_RT(rtDeviceReset(device_id)); });

  while (run_flag_) {
    // Model has not indeedly started running before received data
    SetRunningFlag(false);
    std::shared_ptr<InputDataWrapper> data_wrapper;
    Status ret = data_inputer_->Pop(data_wrapper);
    // Model indeedly start running
    SetRunningFlag(true);
    GE_IF_BOOL_EXEC(data_wrapper == nullptr || ret != SUCCESS, GELOGI("data_wrapper is null!, ret = %u", ret);
                    continue);

    GELOGI("Getting the input data, model_id:%u", model_id_);
    GE_IF_BOOL_EXEC(!run_flag_, break);
    InputData current_data = data_wrapper->GetInput();
    GELOGI("Model thread Run begin, model id:%u, data index:%u.", model_id_, current_data.index);

    RECORD_MODEL_EXECUTION_EVENT(executor_->GetContext(), "[RunInternal] [iteration = %d] Start", iterator_count_);
    HybridModelExecutor::ExecuteArgs args;
    ret = PreRun(current_data, args);
    GE_CHK_BOOL_TRUE_EXEC_WITH_LOG(
        ret != SUCCESS, (void) HandleResult(ret, current_data.index, args, data_wrapper->GetOutput());
        continue, "[Invoke][PreRun] failed, model_id:%u.", model_id_);  // [No need to check value]

    if (pipe_executor_ != nullptr) {
      GELOGI("HybridModel will execute in pipeline mode");
      auto iter_per_run = std::getenv("ITER_NUM");
      if (iter_per_run) {
        args.num_loops = static_cast<int>(strtol(iter_per_run, nullptr, kDefaultLoopCount));
      }
      ret = pipe_executor_->Execute(args);
    } else {
      GELOGI("HybridModel will execute in singleline mode");
      ge::GetContext().SetSessionId(executor_->GetContext()->session_id);
      ge::GetContext().SetContextId(executor_->GetContext()->context_id);
      ret = executor_->Execute(args);
    }
    ret = HandleResult(ret, current_data.index, args, data_wrapper->GetOutput());
    if (ret != SUCCESS) {
      continue;
    }

    RECORD_MODEL_EXECUTION_EVENT(executor_->GetContext(), "[RunInternal] [iteration = %d] End", iterator_count_);
    iterator_count_++;
    SetRunningFlag(false);
    GELOGI("run iterator count is %lu,  model_id:%u", iterator_count_, model_id_);
  }

  GELOGI("Model run end, model id:%u", model_id_);
  return SUCCESS;
}

Status HybridModelAsyncExecutor::HandleResult(Status exec_ret,
                                              uint32_t data_id,
                                              HybridModelExecutor::ExecuteArgs &args,
                                              OutputData *output_data) {
  GELOGD("Start to handle result. model id = %u, data index = %u, execution ret = %u", model_id_, data_id, exec_ret);
  std::vector<ge::Tensor> output_tensor_info_list;
  if (args.is_eos) {
    GELOGI("End of sequence, model id = %u", model_id_);
    GE_CHK_STATUS_RET_NOLOG(OnComputeDone(data_id, END_OF_SEQUENCE, output_tensor_info_list));
    return SUCCESS;
  }

  if (exec_ret != SUCCESS) {
    GELOGE(exec_ret, "[Check][Param:Status] failed to execute graph. model_id = %u", model_id_);
    REPORT_INNER_ERROR("E19999", "failed to execute graph. model_id = %u", model_id_);
    return OnComputeDone(data_id, INTERNAL_ERROR, output_tensor_info_list);
  }

  GE_CHECK_NOTNULL(output_data);
  auto ret = CopyOutputs(args, output_data, output_tensor_info_list);
  if (ret != SUCCESS) {
    OnComputeDone(data_id, INTERNAL_ERROR, output_tensor_info_list);
    return INTERNAL_ERROR;
  }

  GELOGD("Executed graph successfully, model id = %u, data_index = %u", model_id_, data_id);
  return OnComputeDone(data_id, SUCCESS, output_tensor_info_list);
}

Status HybridModelAsyncExecutor::SyncVarData() {
  GELOGI("Sync var data, model id:%u", model_id_);

  TensorValue *global_step_var = model_->GetVariable(NODE_NAME_GLOBAL_STEP);
  if (global_step_var != nullptr) {
    std::vector<uint64_t> v_step;
    v_step.push_back(iterator_count_);
    GE_CHK_RT_RET(rtMemcpy(global_step_var->MutableData(),
                           global_step_var->GetSize(),
                           v_step.data(),
                           v_step.size() * sizeof(uint64_t),
                           RT_MEMCPY_HOST_TO_DEVICE));
  } else {
    GELOGD("No GLOBAL_STEP variable was found.");
  }

  return SUCCESS;
}

Status HybridModelAsyncExecutor::PrepareInputs(const InputData &current_data, HybridModelExecutor::ExecuteArgs &args) {
  if (current_data.blobs.size() < input_tensor_desc_.size()) {
    GELOGE(PARAM_INVALID,
           "[Check][Size]Blob size mismatches, expect at least %zu, but got %zu, model_id = %u",
           input_tensor_desc_.size(), current_data.blobs.size(), model_id_);
    REPORT_INNER_ERROR("E19999", "Blob size mismatches, expect at least %zu, but got %zu, model_id = %u.",
                       input_tensor_desc_.size(), current_data.blobs.size(), model_id_);
    return PARAM_INVALID;
  }

  auto allocator = NpuMemoryAllocator::GetAllocator(device_id_);
  GE_CHECK_NOTNULL(allocator);
  args.input_desc.resize(input_tensor_desc_.size());
  const std::vector<DataBuffer> &blobs = current_data.blobs;
  for (size_t input_index = 0; input_index < input_tensor_desc_.size(); ++input_index) {
    auto tensor_size = input_sizes_[input_index];
    if (is_input_dynamic_[input_index]) {
      if (input_index >= current_data.shapes.size()) {
        GELOGE(PARAM_INVALID,
               "[Check][Range]Shape index out of range, index = %zu, shape size = %zu model_id = %u.",
               input_index, current_data.shapes.size(), model_id_);
        REPORT_INNER_ERROR("E19999", "Shape index out of range, index = %zu, shape size = %zu, model_id = %u.",
                           input_index, current_data.shapes.size(), model_id_);
        return PARAM_INVALID;
      }
      auto &tensor_desc = input_tensor_desc_[input_index];
      GeShape shape(current_data.shapes[input_index]);
      std::vector<std::pair<int64_t, int64_t>> range;
      auto range_ret = tensor_desc->GetShapeRange(range);
      GE_CHK_BOOL_RET_STATUS(range_ret == GRAPH_SUCCESS, INTERNAL_ERROR,
                             "[Invoke][GetShapeRange] failed, ret=%u, model_id = %u.", range_ret, model_id_);
      for (size_t k = 0; k < range.size(); ++k) {
        if (k >= shape.GetDimNum()) {
          break;
        }
        // range[k].second can be -1
        if (shape.GetDim(k) < range[k].first || (range[k].second >= 0 && shape.GetDim(k) > range[k].second)) {
          GELOGE(PARAM_INVALID, "[Check][Range]Dim out of range, shape idx = %zu, dim idx = %zu,"
                 "dim = %ld, range = [%ld, %ld], model_id = %u.",
                 input_index, k, shape.GetDim(k), range[k].first, range[k].second, model_id_);
          REPORT_INNER_ERROR("E19999", "Dim out of range, shape idx = %zu, dim idx = %zu, dim = %ld,"
                             "range = [%ld, %ld], model_id = %u.",
                             input_index, k, shape.GetDim(k), range[k].first, range[k].second, model_id_);
          return PARAM_INVALID;
        }
      }
      tensor_desc->SetShape(shape);
      GELOGD("Update shape[%s] of input[%zu] to [%s]",
             shape.ToString().c_str(), input_index, tensor_desc->MutableShape().ToString().c_str());
      GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetTensorMemorySizeInBytes(*tensor_desc, tensor_size),
                              "[Invoke][GetTensorMemorySizeInBytes]Failed to calc tensor size,"
                              "index = %zu, shape = [%s], model_id = %u.",
                              input_index, tensor_desc->GetShape().ToString().c_str(), model_id_);
      GELOGD("Input tensor[%zu] size = %ld", input_index, tensor_size);
      TensorUtils::SetSize(*tensor_desc, tensor_size);
      args.input_desc[input_index] = tensor_desc;
    }

    GE_CHECK_GE(tensor_size, 0);
    AllocationAttr attr;
    if (GetContext().GetHostExecFlag()) {
      attr.SetMemType(HOST_DDR);
    }
    auto tensor_buffer = TensorBuffer::Create(allocator, tensor_size, &attr);
    GE_CHECK_NOTNULL(tensor_buffer);
    args.inputs.emplace_back(std::shared_ptr<TensorBuffer>(tensor_buffer.release()));

    GELOGD("To copy input data for input[%zu]", input_index);
    const DataBuffer &data_buf = blobs[input_index];
    auto mem_size = static_cast<uint64_t>(tensor_size);

    if (mem_size < data_buf.length) {
      REPORT_INNER_ERROR("E19999",
                         "input data size(%lu) does not match model required size(%lu), ret failed, model_id = %u.",
                         data_buf.length, mem_size, model_id_);
      GELOGE(PARAM_INVALID,
             "[Check][Size]input data size(%lu) does not match model required size(%lu), ret failed, model_id = %u.",
             data_buf.length, mem_size, model_id_);
      return PARAM_INVALID;
    }
    if (data_buf.length > 0) {
      GELOGI("[IMAS]CopyPlainData memcpy graph_%u type[F] output[%zu] memaddr[%p] mem_size[%zu] datasize[%lu]",
             model_->root_runtime_param_.graph_id,
             input_index,
             args.inputs[input_index].GetData(),
             mem_size,
             data_buf.length);
      GE_CHK_RT_RET(rtMemcpy(args.inputs[input_index].MutableData(),
                             mem_size,
                             data_buf.data,
                             data_buf.length,
                             RT_MEMCPY_HOST_TO_DEVICE));
    }
  }
  return SUCCESS;
}

Status HybridModelAsyncExecutor::InitInputDesc() {
  int input_index = 0;
  for (const auto &input_node : model_->GetRootGraphItem()->GetInputNodes()) {
    GELOGD("Init input[%u], node = %s, is_dynamic = %d",
           input_index,
           input_node->NodeName().c_str(),
           input_node->is_dynamic);
    auto output_desc = input_node->MutableOutputDesc(kDataOutputIndex);
    GE_CHECK_NOTNULL(output_desc);
    int64_t tensor_size = -1;
    if (!input_node->is_dynamic) {
      GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetSize(*output_desc, tensor_size),
                              "Failed to get size from %s",
                              input_node->NodeName().c_str());

      if (tensor_size == 0) {
        GELOGW("[%s] Tensor size == 0", input_node->NodeName().c_str());
        GE_CHK_GRAPH_STATUS_RET(TensorUtils::GetTensorMemorySizeInBytes(*output_desc, tensor_size),
                                "Failed to calc tensor size");
        GELOGD("[%s] Tensor size updated to %ld", input_node->NodeName().c_str(), tensor_size);
      }
    }

    input_sizes_.emplace(input_index, tensor_size);
    input_tensor_desc_.emplace(input_index, output_desc);
    is_input_dynamic_.push_back(input_node->is_dynamic);
    input_index += 1;
  }

  return SUCCESS;
}

Status HybridModelAsyncExecutor::OnComputeDone(uint32_t data_index, uint32_t result_code,
                                               std::vector<ge::Tensor> &outputs) {
  GELOGD("OnComputeDone. model id = %u, data index = %u, execution ret = %u", model_id_, data_index, result_code);
  if (listener_ != nullptr) {
    GE_CHK_STATUS(listener_->OnComputeDone(model_id_, data_index, result_code, outputs),
                  "[Invoke][OnComputeDone] failed, model_id = %u.", model_id_);
  }

  return result_code;
}

Status HybridModelAsyncExecutor::CopyOutputs(HybridModelExecutor::ExecuteArgs &args, OutputData *output_data,
                                             std::vector<ge::Tensor> &outputs) {
  // copy output data from op to designated position
  std::vector<ConstGeTensorDescPtr> &output_tensor_desc_list = args.output_desc;
  std::vector<TensorValue> &output_tensors = args.outputs;
  if (output_tensor_desc_list.size() != output_tensors.size()) {
    GELOGE(INTERNAL_ERROR,
           "[Check][Size]Output sizes mismatch. From op_desc = %zu, and from output tensors = %zu, model_id = %u.",
           output_tensor_desc_list.size(), output_tensors.size(), model_id_);
    REPORT_INNER_ERROR("E19999",
                       "Output sizes mismatch. From op_desc = %zu, and from output tensors = %zu, model_id = %u.",
                       output_tensor_desc_list.size(), output_tensors.size(), model_id_);
    return INTERNAL_ERROR;
  }

  GELOGD("Number of outputs = %zu", output_tensor_desc_list.size());
  string execute_mode;
  auto result = ge::GetContext().GetOption(OPTION_EXEC_DYNAMIC_EXECUTE_MODE, execute_mode);
  if (result != SUCCESS) {
    GELOGW("Can not get dynamic execute mode attr");
  }
  GELOGD("The dynamic execute is %s", execute_mode.c_str());
  for (size_t i = 0; i < output_tensors.size(); ++i) {
    GELOGD("Start to process output[%zu]", i);
    auto &output_tensor = output_tensors[i];
    auto &tensor_desc = output_tensor_desc_list.at(i);
    GE_CHECK_NOTNULL(tensor_desc);
    int64_t output_size = -1;
    GE_CHK_GRAPH_STATUS_RET(TensorUtils::CalcTensorMemSize(tensor_desc->GetShape(),
                                                           tensor_desc->GetFormat(),
                                                           tensor_desc->GetDataType(),
                                                           output_size),
                            "[Calc][TensorMemSize]Failed for output[%zu]. shape = [%s], type = %s, format = %s",
                            i,
                            tensor_desc->GetShape().ToString().c_str(),
                            TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType()).c_str(),
                            TypeUtils::FormatToSerialString(tensor_desc->GetFormat()).c_str());

    GELOGD("Got tensor size for output[%zu] successfully. shape = [%s], type = %s, format = %s, size = %ld",
           i,
           tensor_desc->GetShape().ToString().c_str(),
           TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType()).c_str(),
           TypeUtils::FormatToSerialString(tensor_desc->GetFormat()).c_str(),
           output_size);

    GE_CHECK_GE(output_size, 0);
    GE_CHECK_LE(output_size, UINT32_MAX);
    if (output_tensor.GetSize() < static_cast<size_t>(output_size)) {
      GELOGE(INTERNAL_ERROR,
             "[Check][Size]output[%zu] tensor size(%zu) is not enough for output shape [%s], model_id = %u.",
             i, output_tensor.GetSize(), tensor_desc->GetShape().ToString().c_str(), model_id_);
      REPORT_INNER_ERROR("E19999", "output[%zu] tensor size(%zu) is not enough for output shape [%s] model_id = %u",
                         i, output_tensor.GetSize(), tensor_desc->GetShape().ToString().c_str(), model_id_);
      return INTERNAL_ERROR;
    }

    GeShape ge_shape(tensor_desc->GetShape().GetDims());
    GeTensorDesc ge_tensor_desc;
    ge_tensor_desc.SetShape(ge_shape);
    if (output_size > 0) {
      if (execute_mode != kLazyRecompile) {
        auto aligned_ptr = MakeShared<AlignedPtr>(output_size, kAlignment);
        GE_CHECK_NOTNULL(aligned_ptr);
        auto data_buf = aligned_ptr->MutableGet();
        GE_CHECK_NOTNULL(data_buf);
        GE_CHK_RT_RET(rtMemcpy(data_buf, output_size, output_tensor.GetData(), output_size, RT_MEMCPY_DEVICE_TO_HOST));
        GeTensor ge_tensor(ge_tensor_desc);
        ge_tensor.SetData(aligned_ptr, output_size);
        output_data->blobs.emplace_back(data_buf, static_cast<uint32_t>(output_size), false);
        auto tensor = TensorAdapter::AsTensor(ge_tensor);
        outputs.emplace_back(std::move(tensor));
      } else {
        GE_CHK_STATUS_RET(BuildDeviceTensor(output_tensor, ge_tensor_desc, output_size, outputs),
                          "Build device tensor failed");
        output_data->blobs.emplace_back(output_tensor.Release(), static_cast<uint32_t>(output_size), false,
                                        static_cast<uint32_t>(kPlacementDevice));
      }
    } else {
      GELOGW("Output [%zu] is empty. shape = [%s]", i, tensor_desc->GetShape().ToString().c_str());
      GeTensor ge_tensor(ge_tensor_desc);
      ge_tensor.SetData(nullptr, 0U);
      output_data->blobs.emplace_back(nullptr, 0U, false);
      auto tensor = TensorAdapter::AsTensor(ge_tensor);
      outputs.emplace_back(std::move(tensor));
    }
    GELOGD("Output[%zu] added, type = %s, shape = [%s], size = %ld", i,
           TypeUtils::DataTypeToSerialString(tensor_desc->GetDataType()).c_str(),
           tensor_desc->GetShape().ToString().c_str(), output_size);
  }

  return SUCCESS;
}

Status HybridModelAsyncExecutor::BuildDeviceTensor(TensorValue &output_tensor, GeTensorDesc &ge_tensor_desc,
                                                   int64_t output_size, std::vector<ge::Tensor> &outputs) {
  GELOGD("Start to build device tensor");
  MemStorageType mem_type = HBM;
  GE_CHK_STATUS_RET(output_tensor.GetMemType(mem_type), "[Build][DeviceTensor] Get mem type failed");
  GELOGD("Mem type is %d", static_cast<uint32_t>(mem_type));
  auto deleter = [=](uint8_t *device_data) {
    if (device_data != nullptr) {
      GELOGD("Free device addr is %p", device_data);
      if (mem_type == RDMA_HBM) {
        MemManager::Instance().RdmaPoolInstance(RT_MEMORY_HBM).Free(device_data, device_id_);
      } else if (mem_type == HOST_DDR) {
        MemManager::Instance().HostMemInstance(RT_MEMORY_HBM).Free(device_data);
      } else {
        MemManager::Instance().CachingInstance(RT_MEMORY_HBM).Free(device_data, device_id_);
      }
    }
  };
  ge_tensor_desc.SetPlacement(kPlacementDevice);
  GeTensor ge_tensor(ge_tensor_desc);
  auto tensor = TensorAdapter::AsTensor(ge_tensor);
  tensor.SetData(reinterpret_cast<uint8_t *>(output_tensor.Release()), static_cast<size_t>(output_size), deleter);
  outputs.emplace_back(std::move(tensor));
  return SUCCESS;
}

Status HybridModelAsyncExecutor::Execute(const std::vector<DataBuffer> &inputs,
                                         const std::vector<GeTensorDesc> &input_desc,
                                         std::vector<DataBuffer> &outputs,
                                         std::vector<GeTensorDesc> &output_desc) {
  GELOGI("Start to execute model.");

  HybridModelExecutor::ExecuteArgs args;
  args.inputs.resize(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    TensorValue tensor_value(inputs[i].data, inputs[i].length);
    args.inputs[i] = tensor_value;
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    args.outputs.emplace_back(TensorValue(outputs[i].data, outputs[i].length));
  }
  // usr must designate input tensorDesc when input shape is dynamic in inference
  for (size_t i = 0; i < input_desc.size(); ++i) {
    ConstGeTensorDescPtr tensor_desc_ptr = MakeShared<GeTensorDesc>(input_desc[i]);
    args.input_desc.emplace_back(tensor_desc_ptr);
  }

  GE_CHK_STATUS_RET(executor_->Execute(args), "[Invoke][Execute] Failed, model_id = %u.", model_id_);
  for (const auto &output_tensor_desc : args.output_desc) {
    output_desc.emplace_back(*output_tensor_desc);
  }

  return SUCCESS;
}

Status HybridModelAsyncExecutor::Execute(const vector<GeTensor> &inputs, vector<GeTensor> &outputs) {
  GELOGD("Start to execute model.");
  // prepare inputs
  InputData input_data;
  for (auto &tensor : inputs) {
    DataBuffer buffer;
    buffer.data = const_cast<uint8_t *>(tensor.GetData().GetData());
    buffer.length = tensor.GetData().size();
    input_data.blobs.emplace_back(buffer);
    input_data.shapes.emplace_back(tensor.GetTensorDesc().GetShape().GetDims());
  }

  HybridModelExecutor::ExecuteArgs args;
  GE_CHK_STATUS_RET(PrepareInputs(input_data, args),
      "[Invoke][PrepareInputs]Failed to copy input data to model, model_id = %u", model_id_);
  GELOGD("Done copying input data successfully.");
  GE_CHK_STATUS_RET(executor_->Execute(args), "[Invoke][Execute] Failed, model_id = %u.", model_id_);

  std::vector<ge::Tensor> output_tensor_info_list;
  OutputData output_data;
  GE_CHK_STATUS_RET(CopyOutputs(args, &output_data, output_tensor_info_list),
      "[Invoke][CopyOutputs]Failed to copy outputs, model_id = %u.", model_id_);
  GELOGD("Done copying output data successfully. output count = %zu", output_tensor_info_list.size());

  int out_index = 0;
  outputs.resize(output_tensor_info_list.size());
  for (auto &out_tensor_info : output_tensor_info_list) {
    auto &ge_tensor = outputs[out_index];
    if (out_tensor_info.GetSize() > 0) {
      GE_CHK_GRAPH_STATUS_RET(ge_tensor.SetData(out_tensor_info.GetData(), out_tensor_info.GetSize()),
                              "Failed to set output[%d].", out_index);
    }

    ge_tensor.MutableTensorDesc() = *args.output_desc[out_index];
    GELOGD("Set output[%d], tensor size = %ld, shape = [%s]",
           out_index,
           out_tensor_info.GetSize(),
           ge_tensor.MutableTensorDesc().MutableShape().ToString().c_str());
    ++out_index;
  }

  return SUCCESS;
}
Status HybridModelAsyncExecutor::DumpOpDebug() {
  const DumpProperties &dump_properties = executor_->GetContext()->dump_properties;
  if (dump_properties.IsOpDebugOpen()) {
    GELOGD("Opdebug is open in hybrid engine");
    uint32_t op_debug_mode = dump_properties.GetOpDebugMode();
    GE_CHK_RT_RET(op_debug_register_.RegisterDebugForStream(stream_, op_debug_mode, data_dumper_));
    is_op_debug_reg_ = true;
    data_dumper_.SetDumpProperties(dump_properties);
    data_dumper_.SetModelName(model_->GetModelName());
    data_dumper_.SetModelId(model_->GetModelId());
    data_dumper_.SetDeviceId(model_->GetDeviceId());
    void *global_step = nullptr;

    if (dump_properties.IsInferOpDebug()) {
      GELOGD("Init global step when infer with op debug.");
      global_step = executor_->GetContext()->global_step;
    } else {
      TensorValue *varible_global_step = model_->GetVariable(NODE_NAME_GLOBAL_STEP);
      if (varible_global_step != nullptr) {
        global_step = const_cast<void *>(varible_global_step->GetData());
      }
    }

    void *loop_per_iter = nullptr;
    TensorValue *varible_loop_per_iter = model_->GetVariable(NODE_NAME_FLOWCTRL_LOOP_PER_ITER);
    if (varible_loop_per_iter != nullptr) {
      loop_per_iter = const_cast<void *>(varible_loop_per_iter->GetData());
    }

    void *loop_cond = nullptr;
    TensorValue *varible_loop_cond = model_->GetVariable(NODE_NAME_FLOWCTRL_LOOP_COND);
    if (varible_loop_cond != nullptr) {
      loop_cond = const_cast<void *>(varible_loop_cond->GetData());
    }
    data_dumper_.SetLoopAddr(global_step, loop_per_iter, loop_cond);
    GE_CHK_STATUS_RET(data_dumper_.LoadDumpInfo(),
                      "[Invoke][LoadDumpInfo] failed in hybrid engine, model_id = %u.", model_id_);
    GELOGD("Dump op debug SUCCESS in hybrid engine");
  }
  return SUCCESS;
}

}  // namespace hybrid
}  // namespace ge
