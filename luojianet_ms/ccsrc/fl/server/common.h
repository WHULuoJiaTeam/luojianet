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

#ifndef LUOJIANET_MS_CCSRC_FL_SERVER_COMMON_H_
#define LUOJIANET_MS_CCSRC_FL_SERVER_COMMON_H_

#include <map>
#include <string>
#include <numeric>
#include <climits>
#include <memory>
#include <functional>
#include "proto/ps.pb.h"
#include "proto/fl.pb.h"
#include "ir/anf.h"
#include "include/common/utils/utils.h"
#include "ir/dtype/type_id.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "schema/fl_job_generated.h"
#include "schema/cipher_generated.h"
#include "ps/ps_context.h"
#include "ps/core/communicator/http_message_handler.h"
#include "ps/core/communicator/tcp_server.h"
#include "ps/core/communicator/message_handler.h"

namespace luojianet_ms {
namespace fl {
namespace server {
// Definitions for the server framework.
enum ServerMode { PARAMETER_SERVER = 0, FL_SERVER };
enum CommType { HTTP = 0, TCP };
enum AggregationType { FedAvg = 0, FedAdam, FedAdagarg, FedMeta, qffl, DenseGradAccum, SparseGradAccum };

struct RoundConfig {
  // The name of the round. Please refer to round kernel *.cc files.
  std::string name;
  // Whether this round has the time window limit.
  bool check_timeout = false;
  // The length of the time window. Only used when check_timeout is set to true.
  size_t time_window = 3000;
  // Whether this round has to check the request count has reach the threshold.
  bool check_count = false;
  // This round's request threshold count. Only used when threshold_count is set to true.
  size_t threshold_count = 0;
  // Whether this round uses the server number as threshold. This is vital for some rounds in elastic scaling scenario.
  bool server_num_as_threshold = false;
};

struct CipherConfig {
  float share_secrets_ratio = 1.0;
  uint64_t cipher_time_window = 300000;
  size_t exchange_keys_threshold = 0;
  size_t get_keys_threshold = 0;
  size_t share_secrets_threshold = 0;
  size_t get_secrets_threshold = 0;
  size_t client_list_threshold = 0;
  size_t push_list_sign_threshold = 0;
  size_t get_list_sign_threshold = 0;
  size_t minimum_clients_for_reconstruct = 0;
};

// Every instance is one training loop that runs fl_iteration_num iterations of federated learning.
// During every instance, server's training process could be controlled by scheduler, which will change the state of
// this instance.
enum class InstanceState {
  // If this instance is in kRunning state, server could communicate with client/worker and the traning process moves
  // on.
  kRunning = 0,
  // The server is not available for client/worker if in kDisable state.
  kDisable,
  // The server is not available for client/worker if in kDisable state. And this state means one instance has finished.
  // In other words, fl_iteration_num iterations are completed.
  kFinish
};

enum class IterationResult {
  // The iteration is failed.
  kFail,
  // The iteration is successful aggregation.
  kSuccess
};

using luojianet_ms::kernel::Address;
using luojianet_ms::kernel::AddressPtr;
using luojianet_ms::kernel::NativeCpuKernelMod;
using FBBuilder = flatbuffers::FlatBufferBuilder;
using TimeOutCb = std::function<void(bool, const std::string &)>;
using StopTimerCb = std::function<void(void)>;
using FinishIterCb = std::function<void(bool, const std::string &)>;
using FinalizeCb = std::function<void(void)>;
using MessageCallback = std::function<void(const std::shared_ptr<ps::core::MessageHandler> &)>;

// Information about whether server kernel will reuse kernel node memory from the front end.
// Key refers to the server kernel's parameter name, like "weights", "grad", "learning_rate".
// Value refers to the kernel node's parameter index.
using ReuseKernelNodeInfo = std::map<std::string, size_t>;

// UploadData refers to the data which is uploaded by workers.
// Key refers to the data name. For example: "weights", "grad", "learning_rate", etc. This will be set by the worker.
// Value refers to the data of the key.

// We use Address instead of AddressPtr because:
// 1. Address doesn't need to call make_shared<T> so it has better performance.
// 2. The data uploaded by worker is normally parsed from FlatterBuffers or ProtoBuffer. For example: learning rate, new
// weights, etc. Address is enough to store these data.

// Pay attention that Address only stores the void* pointer of the data, so the data must not be released before the
// related logic is done.
using UploadData = std::map<std::string, Address>;

constexpr auto kWeight = "weight";
constexpr auto kNewWeight = "new_weight";
constexpr auto kAccumulation = "accum";
constexpr auto kLearningRate = "lr";
constexpr auto kGradient = "grad";
constexpr auto kNewGradient = "new_grad";
constexpr auto kMomentum = "momentum";
constexpr auto kIndices = "indices";
constexpr auto kAdamM = "m";
constexpr auto kAdamV = "v";
constexpr auto kAdamBeta1Power = "beta1_power";
constexpr auto kAdamBeta2Power = "beta2_power";
constexpr auto kAdamBeta1 = "beta1";
constexpr auto kAdamBeta2 = "beta2";
constexpr auto kAdamEps = "eps";
constexpr auto kFtrlLinear = "linear";
constexpr auto kDataSize = "data_size";
constexpr auto kNewDataSize = "new_data_size";
constexpr auto kStat = "stat";
constexpr auto kStartFLJobTotalClientNum = "startFLJobTotalClientNum";
constexpr auto kStartFLJobAcceptClientNum = "startFLJobAcceptClientNum";
constexpr auto kStartFLJobRejectClientNum = "startFLJobRejectClientNum";
constexpr auto kUpdateModelTotalClientNum = "updateModelTotalClientNum";
constexpr auto kUpdateModelAcceptClientNum = "updateModelAcceptClientNum";
constexpr auto kUpdateModelRejectClientNum = "updateModelRejectClientNum";
constexpr auto kGetModelTotalClientNum = "getModelTotalClientNum";
constexpr auto kGetModelAcceptClientNum = "getModelAcceptClientNum";
constexpr auto kGetModelRejectClientNum = "getModelRejectClientNum";
constexpr auto kMinVal = "min_val";
constexpr auto kMaxVal = "max_val";
constexpr auto kQuant = "QUANT";
constexpr auto kDiffSparseQuant = "DIFF_SPARSE_QUANT";
constexpr auto kNoCompress = "NO_COMPRESS";

// OptimParamNameToIndex represents every inputs/workspace/outputs parameter's offset when an optimizer kernel is
// launched.
using OptimParamNameToIndex = std::map<std::string, std::map<std::string, size_t>>;
const OptimParamNameToIndex kMomentumNameToIdx = {
  {"inputs", {{kWeight, 0}, {kAccumulation, 1}, {kLearningRate, 2}, {kGradient, 3}, {kMomentum, 4}}}, {"outputs", {}}};
const OptimParamNameToIndex kAdamNameToIdx = {{"inputs",
                                               {{kWeight, 0},
                                                {kAdamM, 1},
                                                {kAdamV, 2},
                                                {kAdamBeta1Power, 3},
                                                {kAdamBeta2Power, 4},
                                                {kLearningRate, 5},
                                                {kAdamBeta1, 6},
                                                {kAdamBeta2, 7},
                                                {kAdamEps, 8},
                                                {kGradient, 9}}},
                                              {"outputs", {}}};
const OptimParamNameToIndex kSparseAdamNameToIdx = {{"inputs",
                                                     {{kWeight, 0},
                                                      {kAdamM, 1},
                                                      {kAdamV, 2},
                                                      {kAdamBeta1Power, 3},
                                                      {kAdamBeta2Power, 4},
                                                      {kLearningRate, 5},
                                                      {kAdamBeta1, 6},
                                                      {kAdamBeta1, 7},
                                                      {kAdamEps, 8},
                                                      {kGradient, 9},
                                                      {kIndices, 10}}},
                                                    {"outputs", {}}};
const OptimParamNameToIndex kSparseFtrlNameToIdx = {
  {"inputs", {{kWeight, 0}, {kAccumulation, 1}, {kFtrlLinear, 2}, {kGradient, 3}, {kIndices, 4}}}, {"outputs", {}}};
const OptimParamNameToIndex kAdamWeightDecayNameToIdx = {{"inputs",
                                                          {{"weight", 0},
                                                           {"m", 1},
                                                           {"v", 2},
                                                           {"lr", 3},
                                                           {"beta1", 4},
                                                           {"beta2", 5},
                                                           {"eps", 6},
                                                           {"weight_decay", 7},
                                                           {"grad", 8}}},
                                                         {"outputs", {}}};
const OptimParamNameToIndex kSGDNameToIdx = {
  {"inputs", {{kWeight, 0}, {kGradient, 1}, {kLearningRate, 2}, {kAccumulation, 3}, {kMomentum, 4}, {kStat, 5}}},
  {"outputs", {}}};

const std::map<std::string, OptimParamNameToIndex> kNameToIdxMap = {
  {kApplyMomentumOpName, kMomentumNameToIdx},     {kFusedSparseAdamName, kSparseAdamNameToIdx},
  {kSparseApplyFtrlOpName, kSparseFtrlNameToIdx}, {kApplyAdamOpName, kAdamNameToIdx},
  {"AdamWeightDecay", kAdamWeightDecayNameToIdx}, {kSGDName, kSGDNameToIdx}};

constexpr uint32_t kLeaderServerRank = 0;
constexpr size_t kWorkerMgrThreadPoolSize = 32;
constexpr size_t kWorkerMgrMaxTaskNum = 64;
constexpr size_t kCipherMgrThreadPoolSize = 32;
constexpr size_t kCipherMgrMaxTaskNum = 64;
constexpr size_t kExecutorThreadPoolSize = 32;
constexpr size_t kExecutorMaxTaskNum = 32;
constexpr size_t kNumberTypeFloat16Type = 2;
constexpr size_t kNumberTypeFloat32Type = 4;
constexpr size_t kNumberTypeUInt64Type = 8;
constexpr int kHttpSuccess = 200;
constexpr uint32_t kThreadSleepTime = 50;
constexpr auto kPBProtocol = "PB";
constexpr auto kFBSProtocol = "FBS";
constexpr auto kSuccess = "Success";
constexpr auto kFedAvg = "FedAvg";
constexpr auto kAggregationKernelType = "Aggregation";
constexpr auto kOptimizerKernelType = "Optimizer";
constexpr auto kCtxFuncGraph = "FuncGraph";
constexpr auto kCtxIterNum = "iteration";
constexpr auto kCtxDeviceMetas = "device_metas";
constexpr auto kCtxTotalTimeoutDuration = "total_timeout_duration";
constexpr auto kCtxIterationNextRequestTimestamp = "iteration_next_request_timestamp";
constexpr auto kCtxUpdateModelClientList = "update_model_client_list";
constexpr auto kCtxUpdateModelThld = "update_model_threshold";
constexpr auto kCtxClientsKeys = "clients_keys";
constexpr auto kCtxClientNoises = "clients_noises";
constexpr auto kCtxClientsEncryptedShares = "clients_encrypted_shares";
constexpr auto kCtxClientsReconstructShares = "clients_restruct_shares";
constexpr auto kCtxShareSecretsClientList = "share_secrets_client_list";
constexpr auto kCtxGetSecretsClientList = "get_secrets_client_list";
constexpr auto kCtxReconstructClientList = "reconstruct_client_list";
constexpr auto kCtxExChangeKeysClientList = "exchange_keys_client_list";
constexpr auto kCtxGetUpdateModelClientList = "get_update_model_client_list";
constexpr auto kCtxClientListSigns = "client_list_signs";
constexpr auto kCtxClientKeyAttestation = "client_key_attestation";
constexpr auto kCtxGetKeysClientList = "get_keys_client_list";
constexpr auto kCtxFedAvgTotalDataSize = "fed_avg_total_data_size";
constexpr auto kCtxCipherPrimer = "cipher_primer";
constexpr auto kCurrentIteration = "current_iteration";
constexpr auto kInstanceState = "instance_state";
const char PYTHON_MOD_SERIALIZE_MODULE[] = "luojianet_ms.train.serialization";
const char PYTHON_MOD_SAFE_WEIGHT[] = "_save_weight";

// This macro the current timestamp in milliseconds.
#define CURRENT_TIME_MILLI \
  std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch())

// This method returns the size in bytes of the given TypeId.
inline std::string GetTypeIdByte(const TypeId &type) {
  switch (type) {
    case kNumberTypeFloat16:
      return "Float16";
    case kNumberTypeFloat32:
      return "Float32";
    case kNumberTypeFloat64:
      return "Float64";
    default:
      MS_LOG(EXCEPTION) << "TypeId " << type << " not supported.";
  }
}

inline AddressPtr GenerateParameterNodeAddrPtr(const CNodePtr &kernel_node, size_t param_idx) {
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_node, nullptr);
  auto param_node = common::AnfAlgo::VisitKernelWithReturnType(common::AnfAlgo::GetInputNode(kernel_node, param_idx), 0)
                      .first->cast<ParameterPtr>();
  MS_ERROR_IF_NULL_W_RET_VAL(param_node, nullptr);
  auto param_tensor = param_node->default_param()->cast<tensor::TensorPtr>();
  MS_ERROR_IF_NULL_W_RET_VAL(param_tensor, nullptr);
  AddressPtr addr = std::make_shared<kernel::Address>();
  addr->addr = param_tensor->data_c();
  addr->size = param_tensor->data().nbytes();
  return addr;
}

template <typename T>
inline T JsonGetKeyWithException(const nlohmann::json &json, const std::string &key) {
  if (!json.contains(key)) {
    MS_LOG(EXCEPTION) << "The key " << key << "does not exist in json " << json.dump();
  }
  return json[key].get<T>();
}

// Definitions for Federated Learning.

constexpr auto kNetworkError = "Cluster networking failed.";
constexpr auto KTriggerCounterEventError = "Cluster trigger counter event failed.";

// The result code used for round kernels.
enum class ResultCode {
  // If the method is successfully called and round kernel's residual methods should be called, return kSuccess.
  kSuccess = 0,
  // If there's error happened, return kFail.
  kFail
};

inline std::string GetInstanceStateStr(const InstanceState &instance_state) {
  switch (instance_state) {
    case InstanceState::kRunning:
      return "kRunning";
    case InstanceState::kFinish:
      return "kFinish";
    case InstanceState::kDisable:
      return "kDisable";
    default:
      MS_LOG(EXCEPTION) << "InstanceState " << instance_state << " is not supported.";
  }
}

inline InstanceState GetInstanceState(const std::string &instance_state) {
  if (instance_state == "kRunning") {
    return InstanceState::kRunning;
  } else if (instance_state == "kFinish") {
    return InstanceState::kFinish;
  } else if (instance_state == "kDisable") {
    return InstanceState::kDisable;
  }

  MS_LOG(EXCEPTION) << "InstanceState " << instance_state << " is not supported.";
}
}  // namespace server
}  // namespace fl
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_FL_SERVER_COMMON_H_
