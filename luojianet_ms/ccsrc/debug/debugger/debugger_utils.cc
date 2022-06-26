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

#include "debug/debugger/debugger_utils.h"
#include <iostream>
#include <vector>
#include <memory>
#include <string>
#include "include/common/debug/anf_dump_utils.h"
#include "debug/debugger/debugger.h"
#include "plugin/device/gpu/hal/device/gpu_device_address.h"
#include "debug/data_dump/dump_json_parser.h"
#ifdef ENABLE_D
#include "debug/dump_data_builder.h"
#endif
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/kernel.h"
#include "debug/data_dump/e2e_dump.h"
#include "include/common/utils/config_manager.h"
#include "backend/common/session/session_basic.h"

constexpr int kFailure = 1;

using luojianet_ms::kernel::AddressPtr;
using luojianet_ms::kernel::KernelLaunchInfo;
using AddressPtrList = std::vector<luojianet_ms::kernel::AddressPtr>;
using KernelGraph = luojianet_ms::session::KernelGraph;
using AnfAlgo = luojianet_ms::session::AnfRuntimeAlgorithm;

namespace luojianet_ms {
/*
 * Feature group: Online debugger.
 * Target device group: GPU.
 * Runtime category: MindRT.
 * Description: Returns a vector containing real output number.
 */
std::vector<size_t> CheckRealOutput(const std::string &node_name, const size_t &output_size) {
  std::vector<size_t> real_outputs;
  // P.BatchNorm is used for training and inference
  // can add the filter list for more operators here....
  if (node_name == "BatchNorm") {
    MS_LOG(INFO) << "loading node named " << node_name;
    (void)real_outputs.insert(real_outputs.end(), {0, 3, 4});
  } else {
    // by default, TensorLoader will load all outputs
    for (size_t j = 0; j < output_size; ++j) {
      real_outputs.push_back(j);
    }
  }
  return real_outputs;
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU.
 * Runtime category: MindRT.
 * Description: Get kernel inputs from launch_info and load the inputs from device to host.
 */
void LoadInputs(const CNodePtr &cnode, const KernelLaunchInfo *launch_info, uint32_t exec_order, uint32_t root_graph_id,
                const DeviceContext *device_context) {
  // get inputs
  auto kernel_inputs = launch_info->inputs_;
  auto input_size = common::AnfAlgo::GetInputTensorNum(cnode);
  for (size_t j = 0; j < input_size; ++j) {
    auto input_kernel = cnode->input(j + 1);
    std::string input_kernel_name = GetKernelNodeName(input_kernel);
    auto addr = kernel_inputs[j];
    auto device_type = AnfAlgo::GetOutputDeviceDataType(input_kernel, PARAMETER_OUTPUT_INDEX);
    auto host_type = common::AnfAlgo::GetOutputInferDataType(input_kernel, PARAMETER_OUTPUT_INDEX);
    // For example, this happens with the Depend op
    if (host_type == kMetaTypeNone) {
      continue;
    }

    auto host_format = kOpFormat_DEFAULT;
    auto device_format =
      E2eDump::IsDeviceTargetGPU() ? kOpFormat_DEFAULT : AnfAlgo::GetOutputFormat(input_kernel, PARAMETER_OUTPUT_INDEX);
    auto device_addr =
      device_context->CreateDeviceAddress(addr->addr, addr->size, device_format, device_type, ShapeVector());
    string input_tensor_name = input_kernel_name + ':' + "0";
    ShapeVector int_shapes = trans::GetRuntimePaddingShape(input_kernel, PARAMETER_OUTPUT_INDEX);
    auto ret = device_addr->LoadMemToHost(input_tensor_name, UintToInt(exec_order), host_format, int_shapes, host_type,
                                          0, true, root_graph_id, false);
    if (!ret) {
      MS_LOG(ERROR) << "LoadMemToHost:"
                    << ", tensor_name:" << input_tensor_name << ", host_format:" << host_format << ".!";
    }
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU.
 * Runtime category: MindRT.
 * Description: Get kernel outputs from launch_info and load the inputs from device to host.
 */
void LoadOutputs(const CNodePtr &cnode, const KernelLaunchInfo *launch_info, uint32_t exec_order,
                 uint32_t root_graph_id, const DeviceContext *device_context) {
  // get outputs
  auto kernel_outputs = launch_info->outputs_;
  auto output_size = common::AnfAlgo::GetOutputTensorNum(cnode);
  auto node_name = common::AnfAlgo::GetCNodeName(cnode);
  std::string kernel_name = GetKernelNodeName(cnode);
  std::vector<size_t> real_outputs = CheckRealOutput(node_name, output_size);

  for (size_t j : real_outputs) {
    auto addr = kernel_outputs[j];
    auto device_type = AnfAlgo::GetOutputDeviceDataType(cnode, j);
    auto host_type = common::AnfAlgo::GetOutputInferDataType(cnode, j);
    // For example, this happens with the Depend op
    if (host_type == kMetaTypeNone) {
      continue;
    }

    auto host_format = kOpFormat_DEFAULT;
    auto device_format = E2eDump::IsDeviceTargetGPU() ? kOpFormat_DEFAULT : AnfAlgo::GetOutputFormat(cnode, j);
    auto device_addr =
      device_context->CreateDeviceAddress(addr->addr, addr->size, device_format, device_type, ShapeVector());
    string tensor_name = kernel_name + ':' + std::to_string(j);
    ShapeVector int_shapes = trans::GetRuntimePaddingShape(cnode, j);
    auto ret = device_addr->LoadMemToHost(tensor_name, UintToInt(exec_order), host_format, int_shapes, host_type, j,
                                          false, root_graph_id, false);
    if (!ret) {
      MS_LOG(ERROR) << "LoadMemToHost:"
                    << ", tensor_name:" << tensor_name << ", host_format:" << host_format << ".!";
    }
  }
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Returns true if the node needs to be read for Dump or online debugger. This function is used by GPU
 * and Ascend kernel-by-kernel mindRT.
 */
bool CheckReadData(const CNodePtr &cnode) {
  auto debugger = Debugger::GetInstance();
  if (!debugger) {
    return false;
  }
  bool read_data = false;
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  bool dump_enabled = dump_json_parser.DumpEnabledForIter();
  MS_LOG(DEBUG) << "dump_enabled: " << dump_enabled;
  std::string kernel_name = GetKernelNodeName(cnode);
  if (dump_enabled) {
    if (dump_json_parser.NeedDump(kernel_name)) {
      read_data = true;
    }
  }
  if (debugger->debugger_enabled()) {
    read_data = debugger->ReadNodeDataRequired(cnode);
  }
  return read_data;
}

bool IsDeviceTargetGPU() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  return context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice;
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Load inputs and outputs of the given node if needed and dump them if dump is enabled, then it performs
 * PostExecuteNode function on the given node for GPU.
 */
void ReadDataAndDump(const CNodePtr &cnode, const KernelLaunchInfo *launch_info, uint32_t exec_order,
                     const DeviceContext *device_context) {
  auto debugger = Debugger::GetInstance();
  if (!debugger) {
    return;
  }
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  bool dump_enabled = dump_json_parser.DumpEnabledForIter();
  MS_LOG(DEBUG) << "dump_enabled: " << dump_enabled;
  auto kernel_graph = std::dynamic_pointer_cast<KernelGraph>(cnode->func_graph());
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto root_graph_id = kernel_graph->root_graph_id();
  if (debugger->debugger_enabled() || dump_json_parser.InputNeedDump()) {
    LoadInputs(cnode, launch_info, exec_order, root_graph_id, device_context);
  }
  if (debugger->debugger_enabled() || dump_json_parser.OutputNeedDump()) {
    LoadOutputs(cnode, launch_info, exec_order, root_graph_id, device_context);
  }
  // Dump kernel
  if (dump_enabled) {
    MS_EXCEPTION_IF_NULL(kernel_graph);
    auto graph_id = kernel_graph->graph_id();
    // for GPU, nodes are dumped in graph_id directory.
    if (IsDeviceTargetGPU()) {
      debugger->DumpSingleNode(cnode, graph_id);
    } else {
      // for Ascend, node are dumped in root_graph_id directory.
      debugger->DumpSingleNode(cnode, root_graph_id, launch_info);
    }
    // Clear Dumped data when online debugger is not enabled
    if (!debugger->debugger_enabled()) {
      debugger->ClearCurrentData();
    }
  }
  if (IsDeviceTargetGPU()) {
    // check if the node is last kernel
    bool last_kernel = !common::AnfAlgo::IsInplaceNode(cnode, "skip");
    debugger->PostExecuteNode(cnode, last_kernel);
  }
}

/*
 * Feature group: Dump, Online Debugger.
 * Target device group: Ascend, GPU.
 * Runtime category: MindRT.
 * Description: Returns the error_info when sink_mode is true and we are in online debugger mode or dump mode for
 * GPU, if everything is normal the error_info string will be empty.
 */
std::string CheckDatasetSinkMode(const KernelGraphPtr &graph_ptr) {
  std::string error_info = "";
  bool sink_mode = ConfigManager::GetInstance().dataset_mode() || graph_ptr->IsDatasetGraph();
  auto debugger = Debugger::GetInstance();
  if (debugger->CheckDebuggerDumpEnabled() && sink_mode && IsDeviceTargetGPU()) {
    error_info = "e2e_dump is not supported on GPU with dataset_sink_mode=True. Please set dataset_sink_mode=False";
  }
  if (debugger->CheckDebuggerEnabled() && sink_mode) {
    error_info = "Debugger is not supported with dataset_sink_mode=True. Please set dataset_sink_mode=False";
  }
  return error_info;
}

/*
 * Feature group: Online Debugger.
 * Target device group: Ascend.
 * Runtime category: MindRT.
 * Description: Loads graph's outputs and parameters for Ascend super kernel mode.
 */
void LoadDataForDebugger(const KernelGraphPtr &graph_ptr) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) != kAscendDevice) {
    return;
  }
#ifdef ENABLE_DEBUGGER
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  if (!debugger->CheckDebuggerEnabled()) {
    return;
  }
  MS_LOG(INFO) << "Start load step";
  debugger->SetGraphPtr(graph_ptr);
  // load output
  debugger->LoadGraphOutputs();
  // load parameters
  debugger->LoadParametersAndConst();

#endif
}

void Dump(const KernelGraphPtr &graph, uint32_t rank_id) {
  MS_LOG(DEBUG) << "Start!";
  MS_EXCEPTION_IF_NULL(graph);
  E2eDump::DumpData(graph.get(), rank_id);
  MS_LOG(DEBUG) << "Finish!";
}

uint32_t GetRankID() {
  uint32_t rank_id = 0;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto env_rank_id = common::GetEnv("RANK_ID");
  if (ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL) && !env_rank_id.empty()) {
    // get actual rank id if it's distribution training case.
    rank_id = GetRankId();
  }
  return rank_id;
}

void SuperKernelE2eDump(const KernelGraphPtr &graph) {
#ifndef ENABLE_SECURITY
  Dump(graph, GetRankID());
#endif
}

#ifdef ENABLE_D
/*
 * Feature group: Dump.
 * Target device group: Ascend.
 * Runtime category: Old runtime, MindRT.
 * Description: It is a function to be registered to Adx server for a + m dump feature with the following steps:
 * 1) Merge chunks into one memory segment after receiving all the data for one node.
 * 2) Parse dump data object.
 * 3) Convert data from device to host format.
 * 4) Dump to disk based on configuration.
 */
int32_t DumpDataCallBack(const DumpChunk *dump_chunk, int32_t size) {
  MS_LOG(DEBUG) << "ADX DumpDataCallBack is called";
  MS_LOG(DEBUG) << "The dump_chunk size is: " << size;
  string file_name = dump_chunk->fileName;
  uint32_t isLastChunk = dump_chunk->isLastChunk;

  // parse chunk header
  auto debugger = Debugger::GetInstance();
  MS_EXCEPTION_IF_NULL(debugger);
  auto dump_data_build = debugger->LoadDumpDataBuilder(file_name);
  if (dump_data_build == nullptr) {
    MS_LOG(ERROR) << "Failed to load dump data builder for node " << file_name;
    return 0;
  }
  if (!dump_data_build->CopyDumpChunk(dump_chunk)) {
    return 1;
  }

  if (isLastChunk == 1) {
    // construct dump data object
    debugger::dump::DumpData dump_data;
    std::vector<char> data_buf;
    if (!dump_data_build->ConstructDumpData(&dump_data, &data_buf)) {
      MS_LOG(ERROR) << "Failed to parse data for node " << file_name;
      return 0;
    }

    // convert and save to files
    auto separator = file_name.rfind("/");
    auto path_name = file_name.substr(0, separator);
    auto file_base_name = file_name.substr(separator + 1);
    if (file_base_name.rfind("Opdebug.Node_OpDebug.") == 0) {
      // save overflow data
      E2eDump::DumpOpDebugToFile(file_name, dump_data, data_buf.data());
    } else {
      // save tensor data
      // generate fully qualified file name
      // before: op_type.op_name.task_id.stream_id.timestamp
      // after: op_type.op_name_no_scope.task_id.stream_id.timestamp
      size_t first_dot = file_base_name.find(".");
      size_t second_dot = file_base_name.size();
      const int kNumDots = 3;
      int nth_dot_from_back = 0;
      while (nth_dot_from_back != kNumDots && second_dot != std::string::npos) {
        second_dot = file_base_name.rfind(".", second_dot - 1);
        nth_dot_from_back++;
      }
      if (first_dot == std::string::npos || second_dot == std::string::npos) {
        MS_LOG(ERROR) << "Failed to generate fully qualified file name for " << file_name;
        return 0;
      }
      auto op_type = file_base_name.substr(0, first_dot);
      auto task_stream_timestamp = file_base_name.substr(second_dot);
      std::string op_name = dump_data.op_name();
      auto op_name_no_scope = GetOpNameWithoutScope(op_name, "/");
      E2eDump::DumpTensorToFile(path_name + "/" + op_type + "." + op_name_no_scope + task_stream_timestamp, dump_data,
                                data_buf.data());
    }

    debugger->ClearDumpDataBuilder(file_name);
  }

  return 0;
}
#endif
}  // namespace luojianet_ms
