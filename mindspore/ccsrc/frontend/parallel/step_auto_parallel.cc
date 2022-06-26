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

#include "frontend/parallel/step_auto_parallel.h"

#include <cinttypes>
#include <ctime>
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "utils/hash_map.h"
#include "utils/hash_set.h"
#include "base/core_ops.h"
#include "frontend/optimizer/opt.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/parallel/auto_parallel/dp_algo_costmodel.h"
#include "frontend/parallel/auto_parallel/edge_costmodel.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_generate_strategy.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_parse_graph.h"
#include "frontend/parallel/auto_parallel/rec_core/rec_partition.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/graph_util/node_info.h"
#include "frontend/parallel/graph_util/graph_info.h"
#include "frontend/parallel/ops_info/reshape_info.h"
#include "frontend/parallel/ops_info/tmp_identity_info.h"
#include "frontend/parallel/step_parallel.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/strategy_checkpoint/parallel_strategy_checkpoint.h"
#include "ir/anf.h"
#include "ir/param_info.h"
#include "ir/tensor.h"
#if ((defined ENABLE_CPU) && (!defined _WIN32))
#include "ps/util.h"
#endif

namespace mindspore {
namespace parallel {
bool StepAutoParallel(const FuncGraphPtr &root, const opt::OptimizerPtr &) {
#if ((defined ENABLE_CPU) && (!defined _WIN32) && !defined(__APPLE__))
  if (ps::Util::IsRoleOfPServer() || ps::Util::IsRoleOfScheduler()) {
    return false;
  }
#endif
  MS_EXCEPTION_IF_NULL(root);
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  std::string parallel_mode = ParallelContext::GetInstance()->parallel_mode();
  // assume no change to graph
  bool changes = false;
  // control whether use model_parallel mode
  if (!root->has_flag(kAutoParallel) || (parallel_mode != kAutoParallel) ||
      root->has_flag(AUTO_PARALLEL_RUN_ONCE_ONLY)) {
    return changes;
  }

  std::string strategy_search_mode = ParallelContext::GetInstance()->strategy_search_mode();
  MS_LOG(INFO) << "search_mode: " << strategy_search_mode;

  struct timeval start_time {
    0
  }, end_time{0};
  (void)gettimeofday(&start_time, nullptr);
#ifdef ENABLE_DUMP_IR
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
    draw::Draw(STEP_AUTO_PARALLEL_BEGIN, root);
  }
#endif
  MS_LOG(INFO) << "Now entering step auto parallel";
  TOTAL_OPS = 0;
  AnfNodePtr ret = root->get_return();
  std::vector<AnfNodePtr> all_nodes = DeepScopedGraphSearch(ret);
  if (ParallelInit() != SUCCESS) {
    MS_LOG(EXCEPTION) << "Parallel init failed";
  }
  if (strategy_search_mode == kRecursiveProgramming &&
      ((g_device_manager->DeviceNum() & (g_device_manager->DeviceNum() - 1)) != 0)) {
    MS_LOG(EXCEPTION)
      << "The recursive auto parallel strategy searching mode requires the device num be the power of 2.";
  }
  // mark the forward cnodes, parallel only care these nodes
  MarkForwardCNode(root);
  if (IsInsertVirtualOutput(root)) {
    InsertVirtualOutput(root, all_nodes);
    AnfNodePtr ret_after = root->get_return();
    MS_EXCEPTION_IF_NULL(ret_after);
    all_nodes = DeepScopedGraphSearch(ret_after);
  }
  if (FindCommunicationOp(all_nodes)) {
    MS_LOG(EXCEPTION) << "The graph contain communication op";
  }

  // search parallelization strategy
  if ((strategy_search_mode == kDynamicProgramming) || (strategy_search_mode == kShardingPropagation)) {
    if (ParallelStrategySearch(all_nodes, root) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Auto-parallel strategy search failed when using " << strategy_search_mode
                        << " searching mode";
    }
  } else if (strategy_search_mode == kRecursiveProgramming) {
    if (ParallelStrategyRecSearch(all_nodes, root) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Auto-parallel strategy search failed when using RP searching mode";
    }
  } else {
    MS_LOG(EXCEPTION) << "Auto-parallel strategy searching mode unexpected: " << strategy_search_mode;
  }

  (void)gettimeofday(&end_time, nullptr);
  uint64_t time = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  time += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(INFO) << "Now leaving step auto parallel, used time: " << time << " us";

  root->set_flag(AUTO_PARALLEL_RUN_ONCE_ONLY, true);
  return changes;
}

bool IsElementWiseOperator(const std::string &op_name) {
  // clang-format off
  static const std::set<std::string> elementwise_op = {ACTIVATION, GELU,         TANH,
                                                       SOFTMAX,    LOG_SOFTMAX,  RELU,
                                                       SQRT,       CAST,         POW,
                                                       EXP,        LOG,          COS,
                                                       ACOS,       LOGICALNOT,   NEG,
                                                       SQUARE,     SIGMOID,      ABS,
                                                       ACOSH,      ASIN,         ASINH,
                                                       ATAN,       ATANH,        CEIL,
                                                       COSH,       EXPM1,        LOG1P,
                                                       SIN,        SINH,         TAN,
                                                       RSQRT,      RECIPROCAL,   INV,
                                                       ROUND,      FLOOR,        SIGN,
                                                       ERF,        ERFC,         ZEROSLIKE,
                                                       ONESLIKE,   BESSELI0E,    MOD,
                                                       ASSIGN,     ASSIGN_ADD,   ATAN2,
                                                       DIVNONAN,   LOGICALAND,   ELU,
                                                       LOGICALOR,  RELU6,        SOFTPLUS,
                                                       SOFTSIGN,   LESS,         LESSEQUAL,
                                                       BESSELI1E,  GREATEREQUAL, APPROXIMATEEQUAL,
                                                       REPEAT_ELEMENTS};
  // clang-format on
  auto iter = elementwise_op.find(op_name);
  return (iter != elementwise_op.end());
}

bool IsSplittableOperator(const std::string &op_name) {
  // clang-format off
  static const std::set<std::string> splittable_op =
    {MATMUL, TRANSPOSE, GELU, FAST_GELU, TANH, SOFTMAX, SUB, MUL, DIV, RESHAPE, GREATER, LOG_SOFTMAX, ACTIVATION, PRELU,
     FLOORDIV, L2_NORMALIZE, ADD, MAXPOOL, AVGPOOL, MAXPOOLV2, VIRTUAL_DATA_SET, RELU, ONEHOT, DROPOUT_DO_MASK,
     REDUCE_MAX, REDUCE_MIN, ARGMAXWITHVALUE, ARGMINWITHVALUE, REDUCE_SUM, CONV2D, FUSE_BATCH_NORM, POOLING,
     MAX_POOL_WITH_ARGMAX, SIMPLE_MEAN, FLATTEN, BATCH_NORM, LAYER_NORM, BIAS_ADD, ASSIGN_SUB, COS, ACOS, EXP, STACK,
     LOG, REDUCE_MEAN, REAL_DIV, SIGMOID, POW, MAXIMUM, MINIMUM, EQUAL, NOT_EQUAL, LOGICALNOT, GATHERV2, SQRT, CONCAT,
     STRIDEDSLICE, GET_NEXT, CAST, NEG, SQUARE, BATCH_MATMUL, EXPAND_DIMS, SQUEEZE, SPARSE_GATHERV2, TILE, DROPOUT,
     SOFTMAX_CROSS_ENTROPY_WITH_LOGITS, SIGMOID_CROSS_ENTROPY_WITH_LOGITS, SPARSE_SOFTMAX_CROSS_ENTROPY_WITH_LOGITS,
     EMBEDDING_LOOKUP, FUSE_BATCH_NORM_EX, SPLIT, BROADCAST_TO, ABS, ACOSH, ASIN, ASINH, ATAN, ATANH, CEIL, COSH,
     EXPM1, LOG1P, SIN, SINH, TAN, RSQRT, INV, RECIPROCAL, ROUND, FLOOR, SIGN, ERF, ERFC, ZEROSLIKE, ONESLIKE,
     BESSELI0E, BESSELI1E, FLOORMOD, ASSIGN, ASSIGN_ADD, ATAN2, DIVNONAN, LOGICALAND, LOGICALOR, ELU, RELU6, RELUV2,
     SOFTPLUS, SOFTSIGN, GREATEREQUAL, LESSEQUAL, LESS, APPROXIMATEEQUAL, MOD, UNIQUE, UNSORTED_SEGMENT_SUM,
     UNSORTED_SEGMENT_MIN, REPEAT_ELEMENTS, TENSOR_DOT, RANGE, UNIFORM_CANDIDATE_SAMPLER, SLICE, SELECT, GATHERD,
     UNSORTED_SEGMENT_MAX, GATHER_ND, TOPK, SCATTER_UPDATE, VIRTUAL_OUTPUT, CONV2D_BACK_PROP_INPUT, CONV2D_TRANSPOSE,
     MATMUL_DDS, DSD_MATMUL, UNIFORMREAL, RESIZE_BILINEAR, RESIZE_NEAREST_NEIGHBOR, FAST_GELU, IOU, BOUNDING_BOX_ENCODE,
     RANDOM_CHOICE_WITH_MASK, CROP_AND_RESIZE, ROI_ALIGN, REDUCE_PROD, REDUCE_ANY, REDUCE_ALL, ARGMAX, ARGMIN,
     UNSORTED_SEGMENT_PROD, SQUARE_SUM_ALL, MATMUL_DDS, DSD_MATMUL, UNIFORMREAL, RESIZE_BILINEAR,
     RESIZE_NEAREST_NEIGHBOR, CUM_SUM, FAST_GELU, IOU, BOUNDING_BOX_ENCODE, RANDOM_CHOICE_WITH_MASK, CROP_AND_RESIZE,
     ROI_ALIGN, IS_FINITE, RINT, HSHRINK, HSIGMOID, MISH, SELU, SOFT_SHRINK, XLOGY, XDIVY, CUM_PROD, BITWISE_AND,
     BITWISE_OR, BITWISE_XOR, MUL_NO_NAN, TRUNCATE_DIV, TRUNCATE_MOD, INPLACE_ADD, INPLACE_SUB, L2_LOSS, LERP, ADDN,
     CDIST};
  // clang-format on

  auto iter = splittable_op.find(op_name);
  return (iter != splittable_op.end());
}

bool IsAutoParallelCareNode(const CNodePtr &cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  ValueNodePtr prim_node = cnode->input(0)->cast<ValueNodePtr>();
  if (prim_node == nullptr) {
    return false;
  }
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_node);
  if (prim == nullptr) {
    return false;
  }
  if (prim->name() == SEND || prim->name() == RECEIVE) {
    return false;
  }
  bool bool_result = IsParallelCareNode(cnode) && !IsSplittableOperator(prim->name());
  if (bool_result && (prim->name() != MAKE_TUPLE) && (prim->name() != MAKE_LIST)) {
    MS_LOG(EXCEPTION) << "Should implementing OperatorInfo for: " << prim->name();
  } else if (prim->name() == CAST) {
    if (cnode->fullname_with_scope().find(OPTIMIZER_SUB_STRING) != std::string::npos) {
      // Do not care CASTs from optimizer
      return false;
    }
    return true;
  }
  return IsParallelCareNode(cnode) && IsSplittableOperator(prim->name());
}

// Recording the operators appearing in a for-loop.
// Currently, we assume that the operators in different for-loops are identical, and their traversal
// orderings are also identical.
// Therefore, we create OperatorInfo objects for the operators in a loop (say, loop-3), and reuse them in
// the rest of loops (loop-2, loop-1 and loop-0)
std::set<std::string> ops_in_a_loop_;
// Whether two operators are in different loops; if it is true, then return true.
// If at least one of the two operators is not in the loop, then return false.
// If two operators are in the same loop, the return false.
bool IsOperatorsInTwoSeparateLoops(const CNodePtr &a_cnode, const CNodePtr &b_cnode) {
  auto a_op_info = a_cnode->user_data<OperatorInfo>();
  MS_EXCEPTION_IF_NULL(a_op_info);
  auto b_op_info = b_cnode->user_data<OperatorInfo>();
  MS_EXCEPTION_IF_NULL(b_op_info);
  if ((ops_in_a_loop_.find(a_op_info->name()) == ops_in_a_loop_.end()) ||
      (ops_in_a_loop_.find(b_op_info->name()) == ops_in_a_loop_.end())) {
    return false;
  }
  size_t a_loop_index = 0, b_loop_index = 0;
  const auto &a_fullname = a_cnode->fullname_with_scope();
  if (!GetLoopIndexFromCNode(a_cnode, &a_loop_index)) {
    MS_LOG(EXCEPTION) << "The operator with fullname_with_scope: " << a_fullname << " was not included in the set.";
  }
  const auto &b_fullname = b_cnode->fullname_with_scope();
  if (!GetLoopIndexFromCNode(b_cnode, &b_loop_index)) {
    MS_LOG(EXCEPTION) << "The operator with fullname_with_scope: " << b_fullname << " was not included in the set.";
  }
  if (a_loop_index == b_loop_index) {
    return false;
  }
  return true;
}

// 'configured_stra_ops_' includes all operators that are configured sharding strategies.
std::map<OperatorInfoPtr, StrategyPtr, OpsPtrCompare> configured_stra_ops_;
std::set<OperatorInfoPtr> ignore_candidate_;
void InitCostGraph() {
  if (entire_costgraph == nullptr) {
    entire_costgraph = std::make_shared<CostGraph>();
  }
  MS_EXCEPTION_IF_NULL(CostModelContext::GetInstance());
  CostModelContext::GetInstance()->PrintCostModel();
  entire_costgraph->Init();
  configured_stra_ops_.clear();
  ignore_candidate_.clear();
}

void SetStrategyToOperator(const OperatorInfoPtr &operator_info, const PrimitivePtr &prim,
                           mindspore::HashMap<std::string, ValuePtr> attrs, bool, StrategyMap *stra_map,
                           const std::string &strategy_key_name) {
  // In this case, the configured strategy should be extracted to help setting cost
  StrategyPtr strategyPtr;
  if (StrategyFound(attrs)) {
    strategyPtr = parallel::ExtractStrategy(attrs[IN_STRATEGY]);
  } else {
    strategyPtr = (*stra_map)[strategy_key_name];
  }

  if (strategyPtr == nullptr) {
    return;
  }

  if (prim->name() == RESHAPE) {
    MS_LOG(EXCEPTION) << "Setting strategy for Reshape goes for nothing!";
    return;
  }

  // Set cost for this configured strategy
  if (operator_info->SetCostUnderStrategy(strategyPtr) != SUCCESS) {
    MS_LOG(EXCEPTION) << "Failure: operator " << prim->name() << " SetCostUnderStrategy failed";
    return;
  }

  const auto fully_use_devices = CostModelContext::GetInstance()->fully_use_device();
  if (fully_use_devices) {
    // If configured to fully use devices, then checking for the user-specified strategy
    int64_t used_devices = operator_info->used_devices();
    MS_EXCEPTION_IF_NULL(g_device_manager);
    auto total_device_num = g_device_manager->GetDeviceListByStageId(0).size();

    // 'used_devices == -1' means that 'used_devices_' is not set
    // 'used_devices == 1' means that ALL-1 strategy, which is valid in auto-parallel
    if (used_devices == -1 || (used_devices != 1 && LongToSize(used_devices) != total_device_num)) {
      MS_LOG(EXCEPTION) << "In current configuration 'fully_use_devices' = True, "
                        << "but the specified strategy uses device: " << used_devices
                        << ", total devices: " << total_device_num
                        << ", try to set 'set_algo_parameters(fully_use_devices=False)' "
                           "in package 'mindspore.parallel'.";
    }
  }
  (void)configured_stra_ops_.emplace(operator_info, strategyPtr);
}

void ApplyApproximationForNode(const OperatorInfoPtr &operator_info) {
  auto approximation = CostModelContext::GetInstance()->dp_algo_enable_approxi();
  if (approximation) {
    operator_info->ApproximateStrategies();
    MS_LOG(INFO) << "Approximated StrategyCost for: " << operator_info->name();
  }
}

void AddOperatorToIgnoreCandidates(const PrimitivePtr &prim, const OperatorInfoPtr &operator_info) {
  if (prim->name() == CAST) {
    // add CAST into ignore_candidate
    (void)ignore_candidate_.insert(operator_info);
  }
}

OperatorInfoPtr CreateTheOperatorInfo(const PrimitivePtr &prim, const CNodePtr &cnode, bool is_last_nodes,
                                      StrategyMap *stra_map) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_EXCEPTION_IF_NULL(cnode);
  auto attrs = prim->attrs();
  std::vector<Shapes> shape_list = ExtractShape(cnode);
  if (shape_list.empty()) {
    MS_LOG(EXCEPTION) << "Failure: node " << cnode->UniqueId() << " failed to extract shape";
  }
  // Create an OperatorInfo instance
  OperatorInfoPtr operator_info = NewOperatorInstance(prim, attrs, shape_list);
  MS_EXCEPTION_IF_NULL(operator_info);
  // Set the parameter information for this OperatorInfo (whether the inputs are parameters or not)
  std::vector<bool> parameter_info = ExtractInputParameterByNode(cnode);
  if (operator_info->set_is_parameter(parameter_info) != SUCCESS) {
    MS_LOG(ERROR) << "Initializing parameter information failed for operator: " << operator_info->name();
    return nullptr;
  }
  // Set the data type for inputs and outputs of this OperatorInfo
  auto inputs_type_length = ExtractInputTypeLengthByNode(cnode);
  auto outputs_type = ExtractOutputTypeByNode(cnode);
  std::vector<size_t> outputs_type_length;
  outputs_type_length.reserve(outputs_type.size());
  std::transform(outputs_type.begin(), outputs_type.end(), std::back_inserter(outputs_type_length),
                 GetLengthOfDataType);
  if (operator_info->SetInputAndOutputTypeLength(inputs_type_length, outputs_type_length) != SUCCESS) {
    MS_LOG(ERROR) << "Setting the lengths of inputs and outputs failed for operator: " << operator_info->name();
    return nullptr;
  }
  if (operator_info->set_outputs_type(outputs_type) != SUCCESS) {
    MS_LOG(ERROR) << "Setting the types of outputs failed for operator: " << operator_info->name();
    return nullptr;
  }
  // When the 'inputs' contains numerical values for some operators, these values should be extracted from
  // ANF graph
  auto &inputs = cnode->inputs();
  std::vector<ValuePtr> input_value;
  for (size_t index = 1; index < inputs.size(); ++index) {
    if (inputs[index]->isa<ValueNode>()) {
      input_value.push_back(GetValueNode(inputs[index]));
    } else {
      input_value.emplace_back(nullptr);
    }
  }
  operator_info->set_input_value(input_value);
  operator_info->set_outputs_dtype(cnode->Type());
  operator_info->set_cnode(cnode);
  operator_info->set_auto_parallel(true);

  AddOperatorToIgnoreCandidates(prim, operator_info);
  // key of strategy map
  std::string strategy_key_name = "";
  auto param_names = NodeParameterName(cnode, -1, 0);
  if (!param_names.empty()) {
    strategy_key_name = prim->name() + "_" + param_names[0].first;
  }
  bool load_strategy_from_ckpt =
    StrategyCheckpoint::GetInstance().LoadCheckPointOn() && stra_map->find(strategy_key_name) != stra_map->end();
  // If no strategy has been configured for this operator, then candidate strategies are generated for
  // auto-strategy searching; if this primitive is CAST, we ignore the user-specified strategy.
  // if strategy is set to load from checkpoint, it is prefer to load strategy from checkpoint .
  if ((StrategyFound(attrs) && prim->name() != CAST) || load_strategy_from_ckpt) {
    SetStrategyToOperator(operator_info, prim, attrs, is_last_nodes, stra_map, strategy_key_name);
    return operator_info;
  }

  // Compute split_flag_list_, indicating which input has batch dimension. This is ONLY used for preparation for
  // BatchParallelInfo operator
  operator_info->ComputeBatchSplitFlagList();
  Status retGenStra;
  if (AttrFound(attrs, STRATEGY_GEN_MODE) && GetValue<std::string>(attrs[STRATEGY_GEN_MODE]) == kDataParallel) {
    MS_LOG(INFO) << "generating batch parallel strategy...";
    StrategyPtr strategyPtr = parallel::GenerateBatchParallelStrategy(operator_info, prim);
    retGenStra = operator_info->SetCostUnderStrategy(strategyPtr);
    attrs = prim->attrs();
    operator_info->addAttr(IN_STRATEGY, attrs[GEN_STRATEGY]);  // for d-rec
  } else {
    MS_LOG(INFO) << "auto-searching strategy...";
    retGenStra = operator_info->GenerateStrategies(0);
  }

  if (retGenStra != SUCCESS) {
    MS_LOG(ERROR) << "Strategy search for Operator " << operator_info->name() << " failed.";
    return nullptr;
  }

  bool use_sp_and_dataset = ((ParallelContext::GetInstance()->strategy_search_mode() == kShardingPropagation) ||
                             (ParallelContext::GetInstance()->sharding_propagation())) &&
                            (operator_info->name().find(VIRTUAL_DATA_SET_INFO) != std::string::npos);
  if (use_sp_and_dataset) {
    const auto &swc_vec = operator_info->GetStrategyCost();
    if (swc_vec.empty()) {
      MS_LOG(EXCEPTION) << "No available strategy for: " << operator_info->name();
    }
    MS_EXCEPTION_IF_NULL(swc_vec[0]->strategy_ptr);
    (void)configured_stra_ops_.emplace(operator_info, swc_vec[0]->strategy_ptr);
  }
  // If 'approximation' is enabled, the 'strategy_cost' of each operator is approximated
  ApplyApproximationForNode(operator_info);
  return operator_info;
}

bool IsFindWrong(const OperatorInfoPtr current_op_ptr, const std::string &prim_name) {
  bool is_find_wrong = (current_op_ptr->name().find(VIRTUAL_DATA_SET_INFO) == std::string::npos) &&
                       (current_op_ptr->name().find(BATCH_PARALLEL) == std::string::npos) &&
                       (current_op_ptr->name().find(prim_name + "Info") == std::string::npos);
  if (prim_name == GATHERV2) {
    is_find_wrong = is_find_wrong && (current_op_ptr->name().find(prim_name + "PInfo") == std::string::npos);
  }
  return is_find_wrong;
}

void AddUsersUniqueIdWhenSharingParameter(
  const std::pair<std::string, std::pair<AnfNodePtr, AnfNodeIndexSet>> &parameter_users_info) {
  auto users_set = parameter_users_info.second.second;
  if (users_set.size() > 1) {
    MS_LOG(INFO) << "Parameter " << parameter_users_info.first << " has " << users_set.size() << " users.";
    std::vector<std::string> param_users_uniqueid;
    for (auto user : users_set) {
      MS_LOG(INFO) << "with ID: " << user.first->UniqueId();
      param_users_uniqueid.push_back(user.first->UniqueId());
    }
    entire_costgraph->add_param_users_uniqueid(param_users_uniqueid);
  }
}

// Using CNode's UniqueIds to construct nodes
Status ConstructCostGraphNodesByUniqueId(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &) {
  MS_LOG(INFO) << "Constructing nodes for cost graph begins.";
  // The map from CNode's UniqueId to its operatorInfo
  std::map<std::string, OperatorInfoPtr> from_cnode_to_info;
  // The operator_infos in a loop
  std::vector<OperatorInfoPtr> operators_in_forloop;
  // Key: i-th loop; Value: index of 'operators_in_forloop'
  std::map<size_t, size_t> loop_to_ops;
  // extract strategy from checkpoint for multi-train
  StrategyMap stra_map;
  if (StrategyCheckpoint::GetInstance().LoadCheckPointOn()) {
    if (StrategyCheckpoint::GetInstance().Load(&stra_map) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Load strategy checkpoint failed";
    }
  }

  for (auto &node : all_nodes) {
    // NOTE: we only care about splittable Primitive operators
    auto cnode = node->cast<CNodePtr>();
    bool bool_result = (cnode == nullptr) || (!IsValueNode<Primitive>(cnode->input(0)));
    if (bool_result) {
      continue;
    }
    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    if (!IsAutoParallelCareNode(cnode)) {
      // Needed by rec_parser
      if (ParallelContext::GetInstance()->strategy_search_mode() == kRecursiveProgramming) {
        auto prev_cnode = GetInternalOperatorInfo(cnode, prim_anf_node);
        if (prev_cnode != nullptr) {
          entire_costgraph->add_tuple_getitem(std::make_pair(cnode->UniqueId(), prev_cnode->UniqueId()));
        }
      }
      continue;
    }
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
    MS_EXCEPTION_IF_NULL(prim);

    auto search_cnode = from_cnode_to_info.find(cnode->UniqueId());
    if (search_cnode == from_cnode_to_info.end()) {
      size_t loop_index = 0;
      bool is_in_loop = GetLoopIndexFromCNode(cnode, &loop_index);
      const auto single_loop = CostModelContext::GetInstance()->dp_algo_single_loop();
      if (single_loop && is_in_loop && (loop_to_ops[loop_index] < operators_in_forloop.size())) {
        const auto &current_op_ptr = operators_in_forloop[loop_to_ops[loop_index]];
        if (IsFindWrong(current_op_ptr, prim->name())) {
          MS_LOG(EXCEPTION) << "The OperatorInfo: " << current_op_ptr->name()
                            << " does not match the Prim: " << prim->name()
                            << ". The fullname_with_scope: " << cnode->fullname_with_scope();
        }
        loop_to_ops[loop_index]++;
        cnode->set_user_data<OperatorInfo>(current_op_ptr);
        MS_LOG(INFO) << "The CNode with UniqueId: " << cnode->UniqueId()
                     << " and UniqueIdThroughCopy: " << cnode->UniqueIdThroughCopy()
                     << ", CNode fullname_with_scope: " << cnode->fullname_with_scope()
                     << " is set OperatorInfo: " << current_op_ptr->name() << ", Primitive: " << prim->name();
        (void)from_cnode_to_info.emplace(std::make_pair(cnode->UniqueId(), current_op_ptr));
        continue;
      }
      bool is_last_nodes = IsPrimitiveCNode(cnode, prim::kPrimVirtualOutput);
      auto operator_info = CreateTheOperatorInfo(prim, cnode, is_last_nodes, &stra_map);
      if (operator_info == nullptr) {
        return FAILED;
      }
      // Needed by rec_parser
      operator_info->set_type(prim->name());
      operator_info->set_last_node_flag(is_last_nodes);
      std::vector<std::string> inputs_tensor_name = ExtractInputsTensorName(cnode);

      entire_costgraph->AddOperator(operator_info);
      cnode->set_user_data<OperatorInfo>(operator_info);
      MS_LOG(INFO) << "The CNode with UniqueId: " << cnode->UniqueId()
                   << " and UniqueIdThroughCopy: " << cnode->UniqueIdThroughCopy()
                   << ", CNode fullname_with_scope: " << cnode->fullname_with_scope()
                   << " is set OperatorInfo: " << operator_info->name() << ", Primitive: " << prim->name();
      (void)from_cnode_to_info.emplace(std::make_pair(cnode->UniqueId(), operator_info));
      if (single_loop && is_in_loop) {
        operators_in_forloop.push_back(operator_info);
        ops_in_a_loop_.insert(operator_info->name());
        loop_to_ops[loop_index]++;
      }
      // Needed by rec_parser
      entire_costgraph->add_inputs_tensor_name(inputs_tensor_name);
    } else {
      // Two CNODEs' UniqueIds should not be equal
      MS_LOG(EXCEPTION) << "The CNode with UniqueId: " << cnode->UniqueId()
                        << " and UniqueIdThroughCopy: " << cnode->UniqueIdThroughCopy()
                        << " is set OperatorInfo: " << search_cnode->second->name() << ", Primitive: " << prim->name();
    }
  }

  MS_LOG(INFO) << "Constructing nodes for cost graph ends.";
  // Needed by rec_parser 2
  for (auto &node : all_nodes) {
    if (node->isa<Parameter>()) {
      ParameterUsersInfo parameter_users_info = FindParameterUsers(node, IsParallelCareNode);
      AddUsersUniqueIdWhenSharingParameter(parameter_users_info);
    }
  }

  return SUCCESS;
}

void SetOperatorToCNode(const OperatorInfoPtr &current_op_ptr, const PrimitivePtr &prim, const CNodePtr &cnode) {
  if (current_op_ptr == nullptr) {
    MS_LOG(EXCEPTION) << "Find " << prim->name() << " from CostGraph failed.";
  } else {
    if (IsFindWrong(current_op_ptr, prim->name())) {
      MS_LOG(EXCEPTION) << "The OperatorInfo: " << current_op_ptr->name()
                        << " does not match the Prim: " << prim->name();
    }

    // Needed by rec_parser
    ModifyInputsTensorNameListIfOperatorInfoCreated(current_op_ptr->name(), cnode->UniqueId());

    cnode->set_user_data<OperatorInfo>(current_op_ptr);
    current_op_ptr->set_cnode(cnode);
    MS_LOG(INFO) << "The CNode with UniqueId: " << cnode->UniqueId()
                 << " and UniqueIdThroughCopy: " << cnode->UniqueIdThroughCopy()
                 << ", CNode fullname_with_scope: " << cnode->fullname_with_scope()
                 << " is set OperatorInfo: " << current_op_ptr->name() << ", Primitive: " << prim->name();
  }
}

// Using CNode's UniqueIdThroughCopys to construct nodes
Status ConstructCostGraphNodesByUniqueIdTC(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &) {
  MS_LOG(INFO) << "Constructing nodes for cost graph begins.";
  // The map from CNode's UniqueIdThroughCopy to its operatorInfo
  std::map<std::string, OperatorInfoPtr> from_cnode_to_info;
  // The operator_infos in a loop
  std::vector<OperatorInfoPtr> operators_in_forloop;
  // Key: i-th loop; Value: index of 'operators_in_forloop'
  std::map<size_t, size_t> loop_to_ops;
  // extract strategy from checkpoint for multi-train
  StrategyMap stra_map;
  if (StrategyCheckpoint::GetInstance().LoadCheckPointOn() &&
      StrategyCheckpoint::GetInstance().Load(&stra_map) != SUCCESS) {
    MS_LOG(WARNING) << "Load strategy checkpoint failed";
    return FAILED;
  }
  for (auto &node : all_nodes) {
    // NOTE: we only care about splittable Primitive operators
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || (!IsValueNode<Primitive>(cnode->input(0)))) {
      continue;
    }
    ValueNodePtr prim_anf_node = cnode->input(0)->cast<ValueNodePtr>();
    if (!IsAutoParallelCareNode(cnode)) {
      // Needed by rec_parser
      if (ParallelContext::GetInstance()->strategy_search_mode() == kRecursiveProgramming) {
        auto prev_cnode = GetInternalOperatorInfo(cnode, prim_anf_node);
        if (prev_cnode != nullptr) {
          entire_costgraph->add_tuple_getitem(std::make_pair(cnode->UniqueId(), prev_cnode->UniqueId()));
        }
      }
      continue;
    }
    PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);

    // Find the operatorInfo if it exists
    auto search_cnode = from_cnode_to_info.find(cnode->UniqueIdThroughCopy());
    if (search_cnode == from_cnode_to_info.end()) {
      size_t loop_index = 0;
      bool is_in_loop = GetLoopIndexFromCNode(cnode, &loop_index);
      const auto single_loop = CostModelContext::GetInstance()->dp_algo_single_loop();
      bool is_op_created = single_loop && is_in_loop && (loop_to_ops[loop_index] < operators_in_forloop.size());
      if (is_op_created) {
        const auto &current_op_ptr = operators_in_forloop[loop_to_ops[loop_index]];
        if (IsFindWrong(current_op_ptr, prim->name())) {
          MS_LOG(EXCEPTION) << "The OperatorInfo: " << current_op_ptr->name()
                            << " does not match the Prim: " << prim->name()
                            << ". The fullname_with_scope: " << cnode->fullname_with_scope();
        }
        loop_to_ops[loop_index]++;
        cnode->set_user_data<OperatorInfo>(current_op_ptr);
        MS_LOG(INFO) << "The CNode with UniqueId: " << cnode->UniqueId()
                     << " and UniqueIdThroughCopy: " << cnode->UniqueIdThroughCopy()
                     << ", CNode fullname_with_scope: " << cnode->fullname_with_scope()
                     << " is set OperatorInfo: " << current_op_ptr->name() << ", Primitive: " << prim->name();
        (void)from_cnode_to_info.emplace(std::make_pair(cnode->UniqueIdThroughCopy(), current_op_ptr));
        continue;
      }
      // In this case, the corresponding OperatorInfo is not created, create the new one.
      bool is_last_nodes = IsPrimitiveCNode(cnode, prim::kPrimVirtualOutput);
      auto operator_info = CreateTheOperatorInfo(prim, cnode, is_last_nodes, &stra_map);
      MS_EXCEPTION_IF_NULL(operator_info);

      // Needed by rec_parser
      operator_info->set_type(prim->name());
      operator_info->set_last_node_flag(is_last_nodes);
      std::vector<std::string> inputs_tensor_name = ExtractInputsTensorName(cnode);

      entire_costgraph->AddOperator(operator_info);
      cnode->set_user_data<OperatorInfo>(operator_info);
      MS_LOG(INFO) << "The CNode with UniqueId: " << cnode->UniqueId()
                   << " and UniqueIdThroughCopy: " << cnode->UniqueIdThroughCopy()
                   << ", CNode fullname_with_scope: " << cnode->fullname_with_scope()
                   << " is set OperatorInfo: " << operator_info->name() << ", Primitive: " << prim->name();
      (void)from_cnode_to_info.emplace(std::make_pair(cnode->UniqueIdThroughCopy(), operator_info));
      if (single_loop && is_in_loop) {
        operators_in_forloop.push_back(operator_info);
        ops_in_a_loop_.insert(operator_info->name());
        loop_to_ops[loop_index]++;
      }
      // Needed by rec_parser
      entire_costgraph->add_inputs_tensor_name(inputs_tensor_name);
    } else {
      SetOperatorToCNode(search_cnode->second, prim, cnode);
    }
  }

  MS_LOG(INFO) << "Constructing nodes for cost graph ends.";
  // Needed by rec_parser 2
  for (auto &node : all_nodes) {
    if (node->isa<Parameter>()) {
      ParameterUsersInfo parameter_users_info = FindParameterUsers(node, IsParallelCareNode);
      AddUsersUniqueIdWhenSharingParameter(parameter_users_info);
    }
  }

  return SUCCESS;
}

void CreateEdgeBetweenTwoOps(const OperatorInfoPtr &prev_op_info, const OperatorInfoPtr &node_op_info,
                             const CNodePtr &cnode, const CNodePtr &prev_cnode, const PrimitivePtr &prim,
                             const PrimitivePtr &prev_prim, size_t output_index, size_t input_index,
                             size_t *edge_count) {
  std::string edge_name = prev_op_info->name() + OPERATOR_TO_OPERATOR_CONNECTOR + node_op_info->name();
  // If the edge between these two operators already has been added, then the edge will not be added again.
  if (entire_costgraph->IsEdgeInCostGraph(edge_name, output_index, input_index - 1)) {
    return;
  }
  EdgePtr edge_ptr;
  MS_LOG(INFO) << "Creating edge: " << edge_name;
  if (IsOperatorsInTwoSeparateLoops(prev_cnode, cnode)) {
    MS_LOG(INFO) << "prev_cnode_fullname: " << prev_cnode->fullname_with_scope()
                 << ", cnode_fullname: " << cnode->fullname_with_scope();
    MS_LOG(INFO) << "The two operators in two separate for-loops, thus skip the edge.";
    return;
  }
  const auto stra_follow = CostModelContext::GetInstance()->elementwise_stra_follow();
  bool follow_strategy = (prim->name() == RESHAPE) || (prev_prim->name() == RESHAPE) ||
                         (stra_follow && IsElementWiseOperator(prev_prim->name()));
  if (follow_strategy) {
    // Redistribution in not allowed on the edge.
    // Elementwise operators have the same strategy as their previous operators.
    edge_ptr =
      std::make_shared<Edge>(edge_name, prev_op_info, node_op_info, output_index, input_index - 1, false, true);
  } else {
    edge_ptr = std::make_shared<Edge>(edge_name, prev_op_info, node_op_info, output_index, input_index - 1, false);
  }

  // Init costs for this edge
  if (edge_ptr->InitEdgeCost() != SUCCESS) {
    MS_LOG(EXCEPTION) << "Edge cost initialization failed";
  }
  node_op_info->AddPrevEdge(edge_ptr);
  prev_op_info->AddSuccEdge(edge_ptr);
  entire_costgraph->AddEdge(prev_op_info, node_op_info, edge_ptr);
  bool use_sp = (ParallelContext::GetInstance()->strategy_search_mode() == kShardingPropagation) ||
                (ParallelContext::GetInstance()->sharding_propagation());
  if (use_sp && (prev_prim->name() == CAST) &&
      (configured_stra_ops_.find(node_op_info) != configured_stra_ops_.end())) {
    const auto next_op_stra = configured_stra_ops_[node_op_info];
    const auto cast_stra = edge_ptr->GetPrevOpStrategyByNextOpStrategyWithMiniComm(next_op_stra);
    if (cast_stra == nullptr) {
      MS_LOG(EXCEPTION) << "No available strategy for: " << prev_op_info->name();
    }
    prev_op_info->ClearStrategyCost();
    if (prev_op_info->SetCostUnderStrategy(cast_stra) != SUCCESS) {
      MS_LOG(EXCEPTION) << "Failure: operator " << prev_op_info->name() << " SetCostUnderStrategy failed";
    }
    if (edge_ptr->InitEdgeCost() != SUCCESS) {
      MS_LOG(EXCEPTION) << "Edge cost re-initialization failed.";
    }
    MS_LOG(INFO) << "Set strategy for: " << prev_op_info->name() << " under the strategy of: " << node_op_info->name();
    (void)configured_stra_ops_.emplace(prev_op_info, cast_stra);
  }
  MS_LOG(INFO) << "Successfully adding the edge between " << prev_op_info->name() << " and " << node_op_info->name();
  (*edge_count)++;
}

void ApplyApproximationForGraphs() {
  // If 'approximation' is enabled, the edges need to be checked have effective costs.
  auto approximation = CostModelContext::GetInstance()->dp_algo_enable_approxi();
  if (approximation) {
    entire_costgraph->CheckApproximateCostGraphEdges();
  }
}

static void ConstructCNodeCostGraphEdges(const mindspore::CNodePtr &cnode) {
  auto &inputs = cnode->inputs();
  ValueNodePtr prim_anf_node = inputs[0]->cast<ValueNodePtr>();
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  size_t edge_count = 0;
  auto node_op_info = cnode->user_data<OperatorInfo>();

  for (size_t i = 1; i < inputs.size(); ++i) {
    auto prev_cnode = inputs[i]->cast<CNodePtr>();
    bool bool_result_prev_cnode = (prev_cnode == nullptr) || (!IsValueNode<Primitive>(prev_cnode->input(0)));
    if (bool_result_prev_cnode) {
      continue;
    }
    ValueNodePtr prev_prim_anf_node = prev_cnode->input(0)->cast<ValueNodePtr>();
    PrimitivePtr prev_prim = prev_prim_anf_node->value()->cast<PrimitivePtr>();
    size_t output_index = 0;

    while ((IsAutoParallelCareNode(prev_cnode)) || (prev_prim->name() == prim::kTupleGetItem) ||
           (prev_prim->name() == DEPEND)) {
      if (IsAutoParallelCareNode(prev_cnode)) {
        auto prev_op_info = prev_cnode->user_data<OperatorInfo>();
        CreateEdgeBetweenTwoOps(prev_op_info, node_op_info, cnode, prev_cnode, prim, prev_prim, output_index, i,
                                &edge_count);
        break;
      } else if (prev_prim->name() == prim::kTupleGetItem) {
        // In this case, 'prev_anf_node' is 'tuple_getitem', the actual precursor node is node before
        // this 'tuple_getitem'
        MS_LOG(INFO) << "Jumping the 'tuple_getitem' operator.";
        output_index = LongToSize(GetValue<int64_t>(GetValueNode(prev_cnode->input(2))));
        prev_cnode = prev_cnode->input(1)->cast<CNodePtr>();
        bool bool_result_tuple = (prev_cnode == nullptr) || (!IsValueNode<Primitive>(prev_cnode->input(0)));
        if (bool_result_tuple) {
          break;
        }
        prev_prim_anf_node = prev_cnode->input(0)->cast<ValueNodePtr>();
        prev_prim = prev_prim_anf_node->value()->cast<PrimitivePtr>();
        if (!IsAutoParallelCareNode(prev_cnode)) {
          MS_LOG(EXCEPTION) << "Did not create OperatorInfo for : " << prev_prim->name();
        }
        MS_LOG(INFO) << "Jumped the 'tuple_getitem' operator, "
                     << "and creating an edge between the Operator before "
                     << "'tuple_getitem' and the Operator after 'tuple_getitem'.";
      } else if (prev_prim->name() == DEPEND) {
        // In this case, 'prev_anf_node' is 'depend', the actual precursor node is node before
        // this 'depend'
        MS_LOG(INFO) << "Jumping the 'depend' operator.";
        prev_cnode = prev_cnode->input(1)->cast<CNodePtr>();
        bool bool_result_depend = (prev_cnode == nullptr) || (!IsValueNode<Primitive>(prev_cnode->input(0)));
        if (bool_result_depend) {
          break;
        }
        prev_prim_anf_node = prev_cnode->input(0)->cast<ValueNodePtr>();
        prev_prim = prev_prim_anf_node->value()->cast<PrimitivePtr>();
        MS_LOG(INFO) << "Jumped the 'depend' operator, "
                     << "and creating an edge between the Operator before "
                     << "'depend' and the Operator after 'depend'.";
      }
    }
  }
  MS_LOG(INFO) << "Successfully created " << edge_count << " edges for: " << node_op_info->name();
}

void ConstructCostGraphEdges(const std::vector<AnfNodePtr> &all_nodes) {
  // Step 2
  MS_LOG(INFO) << "Constructing edges for cost graph begins.";
  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if ((cnode == nullptr) || !IsValueNode<Primitive>(cnode->input(0))) {
      continue;
    }
    if (!IsAutoParallelCareNode(cnode)) {
      continue;
    }
    ConstructCNodeCostGraphEdges(cnode);
  }
  ApplyApproximationForGraphs();

  MS_LOG(INFO) << "Constructing edges for cost graph ends.";
}

void ApplyApproximationForParaNode(const OperatorInfoPtr &target_op_info) {
  // If 'approximation' is enabled, the edges need to be checked have effective costs.
  auto approximation = CostModelContext::GetInstance()->dp_algo_enable_approxi();
  if (approximation) {
    target_op_info->ExactStrategiesAndRelatedEdges();
  }
}

void AugmentCostGraph(const std::vector<AnfNodePtr> &all_nodes) {
  // Step 3
  for (auto &node : all_nodes) {
    ParameterUsersInfo parameter_users_info = FindParameterUsers(node, IsAutoParallelCareNode);
    auto parameter_name = parameter_users_info.first;
    auto target_parameter = parameter_users_info.second.first;
    auto target_set = parameter_users_info.second.second;
    if (target_set.size() <= 1) {
      continue;
    }

    // Rule out the case when a Parameter being used by a Operator, but the Operator appears in multiple CNODEs
    std::set<std::string> target_without_duplicate;
    for (auto &target : target_set) {
      auto target_cnode = target.first->cast<CNodePtr>();
      // Eliminate the ops without cost.
      if (IsSomePrimitive(target_cnode, SEND)) {
        continue;
      }
      auto input_index = target.second;
      (void)target_without_duplicate.insert(std::to_string(input_index) +
                                            target_cnode->user_data<OperatorInfo>()->name());
    }
    if (target_without_duplicate.size() <= 1) {
      continue;
    }

    // Here, it is sure that this Parameter (RefKey) is being used by multiple Operators.
    OperatorInfoPtr tmp_identity_ptr;
    bool new_identity = false;
    std::string tmp_identity_name;
    auto returned_identity = entire_costgraph->FindTmpIdentityByParameterName(parameter_name);
    if (returned_identity != nullptr) {
      // In this case, the TmpIdentityInfo instance has already been created
      new_identity = false;
      tmp_identity_ptr = returned_identity;
      tmp_identity_name = tmp_identity_ptr->name();
    } else {
      // In the case, the TmpIdentityInfo instance has NOT been created. Thus, a new one is created.
      new_identity = true;
      // 1) extract input shape from this Parameter
      MS_EXCEPTION_IF_NULL(target_parameter);
      AbstractBasePtr abstract = target_parameter->abstract();
      if (abstract == nullptr) {
        MS_LOG(EXCEPTION) << "Failure: abstract is nullptr";
      }
      auto input_shape = dyn_cast<abstract::Shape>(abstract->GetShapeTrack());
      if (input_shape == nullptr) {
        MS_LOG(EXCEPTION) << "Failure: input_shape is nullptr";
      }
      Shape shape = input_shape->shape();
      Shapes inputs_shape = {shape};
      Shapes outputs_shape = {shape};
      // 2) init the attr
      mindspore::HashMap<std::string, ValuePtr> attr = {};

      // Create the TmpIdentity instance
      tmp_identity_ptr = std::make_shared<TmpIdentityInfo>(inputs_shape, outputs_shape, attr);
      tmp_identity_ptr->set_name(tmp_identity_ptr->name() + std::to_string(TOTAL_OPS));
      TOTAL_OPS++;
      tmp_identity_ptr->set_refkey_parameter_name(parameter_name);
      // Set the parameter and type lengths for inputs and outputs
      std::vector<bool> is_parameter;
      auto casted_target_parameter = target_parameter->cast<ParameterPtr>();
      MS_EXCEPTION_IF_NULL(casted_target_parameter);
      is_parameter.push_back(ParameterRequireGrad(casted_target_parameter));
      if (tmp_identity_ptr->set_is_parameter(is_parameter) != SUCCESS) {
        MS_LOG(EXCEPTION) << "Setting parameter for TmpIdentityInfo failed";
      }
      auto node_type = target_parameter->Type();
      if (node_type->isa<mindspore::TensorType>()) {
        auto input_element_type = node_type->cast<mindspore::TensorTypePtr>()->element();
        std::vector<size_t> type_length = {GetLengthOfDataType(input_element_type)};
        if (tmp_identity_ptr->SetInputAndOutputTypeLength(type_length, type_length) != SUCCESS) {
          MS_LOG(EXCEPTION) << "Setting input and output type length for TmpIdentityInfo failed";
        }
      } else {
        MS_LOG(EXCEPTION) << "Unknown type: " << node_type->type_name();
      }

      // Generate strategies for this TmpIdentityInfo instance;
      if (tmp_identity_ptr->GenerateStrategies(0) != SUCCESS) {
        MS_LOG(EXCEPTION) << "Strategy search for Operator failed : " << tmp_identity_ptr->name();
      }
    }
    // A flag recording whether new edges have been created or not
    bool add_identity_edge = false;

    // Create edges between this TmpIdentityInfo instance and subsequent Operator instances
    for (auto &target : target_set) {
      auto target_cnode = target.first->cast<CNodePtr>();
      auto input_index = target.second;
      auto target_op_info = target_cnode->user_data<OperatorInfo>();

      std::string edge_name = std::string(IDENTITY_INFO) + OPERATOR_TO_OPERATOR_CONNECTOR + target_op_info->name();
      // If the edge between these two operators already has been added, then the edge will not be added again.
      if (entire_costgraph->IsEdgeInCostGraph(edge_name, 0, LongToSize(input_index - 1))) {
        continue;
      }
      std::shared_ptr<Edge> edge_ptr =
        std::make_shared<Edge>(edge_name, tmp_identity_ptr, target_op_info, 0, input_index - 1, false, true);
      ApplyApproximationForParaNode(target_op_info);

      if (edge_ptr->InitEdgeCost() != SUCCESS) {
        MS_LOG(EXCEPTION) << "Edge cost initialization failed";
      }
      target_op_info->AddPrevEdge(edge_ptr);
      tmp_identity_ptr->AddSuccEdge(edge_ptr);
      entire_costgraph->AddEdge(tmp_identity_ptr, target_op_info, edge_ptr);
      MS_LOG(INFO) << "Successfully adding the edge between " << tmp_identity_ptr->name() << " and "
                   << target_op_info->name();
      add_identity_edge = true;
    }
    if (new_identity && add_identity_edge) {
      // Add the TmpIdentityInfo to CostGraph if BOTH two conditions are satisfied
      entire_costgraph->AddOperator(tmp_identity_ptr);
    }
  }
}

void ReshapeCostCompute(const std::vector<AnfNodePtr> &all_nodes) {
  mindspore::HashSet<std::string> op_cache;
  for (auto node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (!FindReshape(cnode, &op_cache)) {
      continue;
    }
    MS_ASSERT(cnode->inputs().size() == 3);
    // get previous node's strategy_cost_
    auto pre_node = cnode->input(1);
    if (IsPrimitiveCNode(pre_node, prim::kPrimLoad)) {
      pre_node = pre_node->cast<CNodePtr>()->input(1);
    }
    int64_t out_index = 0;
    OperatorInfoPtr pre_operator_info;
    std::vector<std::shared_ptr<StrategyWithCost>> pre_stra_costs;
    auto operator_info = cnode->user_data<OperatorInfo>();
    if (pre_node->isa<Parameter>()) {
      auto reshape_info1 = std::dynamic_pointer_cast<ReshapeInfo>(operator_info);
      reshape_info1->SetCostForReshapeWithParameter();
      pre_operator_info = reshape_info1;
      pre_stra_costs = reshape_info1->strategy_cost();
    } else {
      if (!FindReshapePreNodeStraCosts(pre_node, &pre_operator_info, &out_index, 0)) {
        MS_LOG(EXCEPTION) << "FindReshapePreNodeStraCosts for reshape failed";
      }
      pre_stra_costs = pre_operator_info->strategy_cost();
    }
    // get next node's strategy_cost_
    int64_t in_index = 0;
    OperatorInfoPtr next_operator_info;
    bool is_next_reshape = false;
    std::vector<std::shared_ptr<StrategyWithCost>> next_stra_costs;
    bool find_next_node = FindReshapeNextNodeStraCosts(cnode, &next_operator_info, &in_index, &is_next_reshape, 0);
    if (!find_next_node) {
      MS_LOG(INFO) << "FindReshapeNextNodeStraCosts for reshape failed";
    }
    // set input_layout and output_layout for reshape.
    // init reshape and set cost for each input_layout and output_layout.
    auto reshape_info = std::dynamic_pointer_cast<ReshapeInfo>(operator_info);
    reshape_info->set_pre_operator_name(pre_operator_info->name());
    reshape_info->set_pre_operator_index(out_index);
    if (find_next_node) {
      next_stra_costs = next_operator_info->strategy_cost();
      reshape_info->set_next_operator_name(next_operator_info->name());
      reshape_info->set_next_operator_index(in_index);
    }
    bool is_prev_param = pre_node->isa<Parameter>();
    if (reshape_info->GenerateStrategyCosts(pre_stra_costs, next_stra_costs, out_index, in_index, is_prev_param,
                                            is_next_reshape) != SUCCESS) {
      MS_LOG(EXCEPTION) << "reshape generate strategy_costs failed!";
    }
  }
}

Status IgnoreOperatorsInCostGraph() {
  for (const auto &op : ignore_candidate_) {
    auto cnodes = op->cnodes();
    for (auto &cnode : cnodes) {
      MS_EXCEPTION_IF_NULL(cnode);
      cnode->set_user_data<OperatorInfo>(nullptr);
    }
  }
  return SUCCESS;
}

Status ParallelStrategySearch(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root) {
  // There are 4 meta-steps to determine the parallelization strategy for the ANF graph.
  // Step 1: Traverse the ANF graph, and create NODEs for costgraph:
  //      create the OperatorInfo object for each primitive, and enumerate the parallelization strategies
  //      for each OperatorInfo;
  // Step 1.1: Deal with 'Reshape':
  //      For 'Reshape', it takes its previous operator's layout as its input layout, and takes its next operator's
  //      layout as its output layout.
  // Step 2: Traverse the ANF graph, and create EDGES for costgraph:
  //      create the Edge object for each pair of OperatorInfo, and enumerate the parallelization strategies
  //      for each edge, based on the strategies of two OperatorInfos;
  // Step 3: Augment the costgraph:
  //      taking care for the case of a single Parameter being used by multiple operators. Create a TmpIdentity
  //      operator for this Parameter, and add an edge for the use of this Parameter by each
  //      subsequent operator;
  // Step 3.1: Calculate memory usage:
  //      note the memory usage calculation is different in training phase and inference phase.
  // Step 4: Run the strategy searching algorithm:
  //      If 'sharding_propagation' is configured to be true, then the configured-sharding-strategies will propagate
  //      to the non-configured operators, with the goal of minimizing redistribution cost.
  //      Otherwise, DP algorithm is used to search strategy of the costgraph. Note that there may be several connected
  //      components in the costgraph, and the DP algorithm runs on each of them.
  //
  // OUTPUT: the determined strategy for each operator.

  InitCostGraph();
  // Step 1
  if (CostModelContext::GetInstance()->is_multi_subgraphs()) {
    if (ConstructCostGraphNodesByUniqueIdTC(all_nodes, root) == SUCCESS) {
      MS_LOG(INFO) << "Constructing nodes for cost graph succeeded. There are "
                   << entire_costgraph->GetOperators().size() << " operators.";
    } else {
      MS_LOG(EXCEPTION) << "Constructing nodes for cost graph failed.";
    }
  } else {
    if (ConstructCostGraphNodesByUniqueId(all_nodes, root) == SUCCESS) {
      MS_LOG(INFO) << "Constructing nodes for cost graph succeeded. There are "
                   << entire_costgraph->GetOperators().size() << " operators.";
    } else {
      MS_LOG(EXCEPTION) << "Constructing nodes for cost graph failed.";
    }
  }
  // Step 1.1
  ReshapeCostCompute(all_nodes);
  // Step 2
  ConstructCostGraphEdges(all_nodes);
  MS_LOG(INFO) << "Constructing edges for cost graph succeeded. There are " << entire_costgraph->GetOperators().size()
               << " operators, and " << entire_costgraph->GetNumEdges() << " edges.";

  // Step 3: Augment the costgraph.
  AugmentCostGraph(all_nodes);
  auto num_ops = entire_costgraph->GetOperators().size();
  SetOpsNumToExecutor(num_ops);
  auto num_edges = entire_costgraph->GetNumEdges();
  MS_LOG(INFO) << "After the augmenting procedure, there are " << num_ops << " operators, and " << num_edges
               << " edges.";

  // Step 3.1: Calculate the memory usage
  if (entire_costgraph->CalculateMemoryCost() != SUCCESS) {
    MS_LOG(EXCEPTION) << "Calculating memory cost failed.";
  }

  // Step 4: run the strategy searching algorithm
  bool use_sp = (ParallelContext::GetInstance()->strategy_search_mode() == kShardingPropagation) ||
                (ParallelContext::GetInstance()->sharding_propagation());
  if (use_sp) {
    entire_costgraph->StrategyPropagate(configured_stra_ops_);
  } else if (GetStrategy(entire_costgraph) != SUCCESS) {
    MS_LOG(ERROR) << "Strategy search for cost-graph fails";
    return FAILED;
  }
  MS_LOG(INFO) << "Searching strategy succeeded.";

  if (entire_costgraph->InitSelectedStrategy() == SUCCESS) {
    MS_LOG(INFO) << "Init selected strategy succeeded.";
  } else {
    MS_LOG(EXCEPTION) << "Init selected strategy failed.";
  }

  // print the selected strategy
  for (auto &op : entire_costgraph->GetOperators()) {
    StrategyPtr s_strategy = op->selected_strategy();
    MS_LOG(INFO) << op->name() << " : The strategy is:";
    PrintStrategy(s_strategy);
  }
  // Remove some operatorInfo from the CNODEs
  (void)IgnoreOperatorsInCostGraph();

  ops_in_a_loop_.clear();
  configured_stra_ops_.clear();
  ignore_candidate_.clear();

  return SUCCESS;
}

std::vector<std::vector<std::string>> RecInputTensorNames(const std::map<std::string, std::string>::iterator &it,
                                                          std::vector<std::vector<std::string>> input_tensor_names) {
  for (size_t j = 0; j < input_tensor_names.size(); j++) {
    for (size_t k = 0; k < input_tensor_names[j].size(); k++) {
      if (it->first == input_tensor_names[j][k]) {
        input_tensor_names[j][k] = it->second;
        break;
      }
    }
  }
  return input_tensor_names;
}

CNodePtr GetInternalOperatorInfo(const CNodePtr &cnode, const ValueNodePtr &prim_anf_node) {
  PrimitivePtr prim = GetValueNode<PrimitivePtr>(prim_anf_node);
  if (prim->name() == prim::kTupleGetItem || prim->name() == DEPEND) {
    auto prev_cnode = cnode->input(1)->cast<CNodePtr>();
    if (prev_cnode == nullptr || !IsValueNode<Primitive>(prev_cnode->input(0))) {
      return nullptr;
    }
    auto prev_prim = prev_cnode->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
    while (prev_prim->name() == prim::kTupleGetItem || prev_prim->name() == DEPEND) {
      prev_cnode = prev_cnode->input(1)->cast<CNodePtr>();
      if (prev_cnode == nullptr || !IsValueNode<Primitive>(prev_cnode->input(0))) {
        return nullptr;
      }
      prev_prim = prev_cnode->input(0)->cast<ValueNodePtr>()->value()->cast<PrimitivePtr>();
    }
    return prev_cnode;
  }
  return nullptr;
}

void ModifyInputsTensorNameListIfOperatorInfoCreated(const std::string &name, const std::string &uniqueid) {
  size_t iter_ops = 0;
  for (auto op : entire_costgraph->GetOperators()) {
    if (op->name() == name) {
      break;
    }
    iter_ops = iter_ops + 1;
  }

  std::vector<std::vector<std::string>> input_tensor_names = entire_costgraph->get_inputs_tensor_name_list();
  for (size_t i = 0; i < input_tensor_names.size(); i++) {
    for (size_t j = 0; j < input_tensor_names[i].size(); j++) {
      if (input_tensor_names[i][j] == uniqueid) {
        input_tensor_names[i][j] = input_tensor_names[iter_ops][0];
      }
    }
  }

  entire_costgraph->set_inputs_tensor_name_list(input_tensor_names);
}

size_t FindOperatorIndexById(const std::string &unique_id,
                             const std::vector<std::vector<std::string>> &input_tensor_names) {
  for (size_t i = 0; i < input_tensor_names.size(); i++) {
    if (input_tensor_names[i][0] == unique_id) {
      return i;
    }
  }
  return SIZE_MAX;
}

std::vector<std::vector<size_t>> GetIndexOfOpsSharingInputTensor(
  const std::vector<std::vector<std::string>> &param_users_uniqueid_list,
  const std::vector<std::vector<std::string>> &input_tensor_names) {
  std::vector<std::vector<size_t>> param_users_ops_index;
  for (auto users_uniqueid : param_users_uniqueid_list) {
    std::vector<size_t> users_index;
    for (size_t i = 0; i < users_uniqueid.size(); i++) {
      size_t user_index = FindOperatorIndexById(users_uniqueid[i], input_tensor_names);
      if (user_index != SIZE_MAX) {
        users_index.push_back(user_index);
      }
    }
    param_users_ops_index.push_back(users_index);
  }
  return param_users_ops_index;
}

Status ParallelStrategyRecSearch(const std::vector<AnfNodePtr> &all_nodes, const FuncGraphPtr &root) {
  InitCostGraph();
  if (CostModelContext::GetInstance()->is_multi_subgraphs()) {
    if (ConstructCostGraphNodesByUniqueIdTC(all_nodes, root) == SUCCESS) {
      MS_LOG(INFO) << "Constructing nodes for cost graph succeeded. There are "
                   << entire_costgraph->GetOperators().size() << " operators.";
    } else {
      MS_LOG(EXCEPTION) << "Constructing nodes for cost graph failed.";
    }
  } else {
    if (ConstructCostGraphNodesByUniqueId(all_nodes, root) == SUCCESS) {
      MS_LOG(INFO) << "Constructing nodes for cost graph succeeded. There are "
                   << entire_costgraph->GetOperators().size() << " operators.";
    } else {
      MS_LOG(EXCEPTION) << "Constructing nodes for cost graph failed.";
    }
  }
  ReshapeCostCompute(all_nodes);
  ConstructCostGraphEdges(all_nodes);

  auto ops = entire_costgraph->GetOperators();
  std::vector<std::vector<std::string>> input_tensor_names = entire_costgraph->get_inputs_tensor_name_list();
  // Needed by rec_parser 2
  auto param_users_uniqueid_list = entire_costgraph->get_param_users_uniqueid_list();
  auto tuple_getitem_list = entire_costgraph->get_tuple_getitem_list();
  for (auto it = tuple_getitem_list.begin(); it != tuple_getitem_list.end();) {
    input_tensor_names = RecInputTensorNames(it++, input_tensor_names);
  }
  std::shared_ptr<Graph> graph = ParseGraph(ops, input_tensor_names);
  std::vector<std::vector<size_t>> param_users_ops_index =
    GetIndexOfOpsSharingInputTensor(param_users_uniqueid_list, input_tensor_names);

  std::shared_ptr<std::vector<std::vector<size_t>>> eli_list = std::make_shared<std::vector<std::vector<size_t>>>();
  std::shared_ptr<std::vector<size_t>> index_list = std::make_shared<std::vector<size_t>>();
  graph = EliminateGraph(graph, eli_list, index_list);

  size_t num_device = g_device_manager->DeviceNum();
  const auto device_memory = CostModelContext::GetInstance()->device_memory_capacity();
  if (PartitionForAllDevices(num_device, device_memory, graph) == SUCCESS) {
    MS_LOG(INFO) << "Partition Success With " << num_device << " devices.";
  } else {
    MS_LOG(ERROR) << "PartitionForAllDevices failed.";
    return FAILED;
  }

  bool is_training = true;
  if (!root->has_flag(kTraining)) {
    is_training = false;
  }
  GenerateStrategy(graph, ops, eli_list, input_tensor_names, index_list, is_training, param_users_ops_index);

  if (entire_costgraph->InitSelectedStrategy() == SUCCESS) {
    MS_LOG(INFO) << "Init selected strategy succeeded.";
  } else {
    MS_LOG(ERROR) << "Init selected strategy failed.";
    return FAILED;
  }

  // print the selected strategy
  for (auto &op : entire_costgraph->GetOperators()) {
    StrategyPtr s_strategy = op->selected_strategy();
    MS_LOG(INFO) << op->name() << " : The strategy is:";
    PrintStrategy(s_strategy);
  }

  (void)IgnoreOperatorsInCostGraph();

  return SUCCESS;
}
}  // namespace parallel
}  // namespace mindspore
