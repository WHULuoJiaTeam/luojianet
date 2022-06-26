/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/static_analysis/evaluator.h"

#include <algorithm>
#include <utility>

#include "utils/hash_set.h"
#include "ir/func_graph_cloner.h"
#include "abstract/utils.h"
#include "pipeline/jit/debug/trace.h"
#include "utils/ms_context.h"
#include "pipeline/jit/static_analysis/stack_frame.h"
#include "pipeline/jit/static_analysis/async_eval_result.h"

namespace mindspore {
namespace abstract {
namespace {
string EvalEntryLogging(const EvaluatorPtr &evaluator, const AbstractBasePtrList &arg_spec_list,
                        const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(evaluator);
  std::stringstream ss;
  if (out_conf != nullptr) {
    ss << "Evaluator " << evaluator->ToString() << " run for " << out_conf->node()->scope()->name();
  }
  for (size_t i = 0; i < arg_spec_list.size(); i++) {
    ss << evaluator->ToString() << " input[" << i
       << "] abstract value: " << (arg_spec_list[i] ? arg_spec_list[i]->ToString() : "null abstract.");
  }
  return ss.str();
}

void EvalFailLogging(const EvaluatorPtr &evaluator, const AbstractBasePtrList &, const AnfNodeConfigPtr &out_conf) {
  MS_EXCEPTION_IF_NULL(evaluator);
  if (out_conf != nullptr) {
    auto node = out_conf->node();
    if (IsValueNode<Primitive>(node)) {
      MS_LOG(ERROR) << "Evaluator " << evaluator->ToString() << " run failed for node " << node->fullname_with_scope()
                    << ", with debug info: " << trace::GetDebugInfo(node->debug_info());
    } else {
      MS_LOG(ERROR) << "Evaluator " << evaluator->ToString() << " run failed for node " << node->DebugString()
                    << ", with debug info: " << trace::GetDebugInfo(node->debug_info());
    }
  }
}
}  // namespace

bool CheckIfAlwaysEval(const AnfNodeConfigPtr &conf, const AbstractBasePtr &arg) {
  auto new_sequence = dyn_cast<AbstractSequence>(arg);
  if (new_sequence != nullptr && new_sequence->sequence_nodes() != nullptr && new_sequence->size() != 0) {
    static AnalysisResultCacheMgr &cache_mgr = AnalysisResultCacheMgr::GetInstance();
    auto prev_result = cache_mgr.GetValue(conf);
    if (prev_result == nullptr) {
      return false;
    }
    auto prev_abs = prev_result->abstract();
    auto old_sequence = dyn_cast<AbstractSequence>(prev_abs);
    if (old_sequence != nullptr &&
        (old_sequence->sequence_nodes() == nullptr || old_sequence->sequence_nodes()->empty()) && *arg == *prev_abs) {
      MS_LOG(DEBUG) << "Always eval";
      return true;
    }
  }
  return false;
}

void BaseFuncGraphEvaluator::EnterStackFrame(const AnalysisEnginePtr &engine, const StackFramePtr &current_stack_frame,
                                             const StackFramePtr &new_stack_frame) {
  MS_EXCEPTION_IF_NULL(current_stack_frame);
  MS_EXCEPTION_IF_NULL(new_stack_frame);
  MS_EXCEPTION_IF_NULL(engine);
  // Enter new func graph.
  auto &current_node = current_stack_frame->CurrentNode();
  auto current_context = current_stack_frame->current_context();
  AnfNodeConfigPtr call_conf = engine->MakeConfig(current_node, current_context, current_context->func_graph());
  auto evaluator = new_stack_frame->evaluator();
  MS_EXCEPTION_IF_NULL(evaluator);
  auto new_context = new_stack_frame->current_context();
  trace::TraceGraphEvalEnter(new_context, call_conf);

  // Increase & Check the func graph call depth.
  IncreaseFunctionCallDepth();
  IncreaseStackFrameDepth();
  const uint32_t max_depth = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_MAX_CALL_DEPTH);
  if (FunctionCallDepth() > max_depth) {
    MS_LOG(EXCEPTION) << "Exceed function call depth limit " << max_depth
                      << ", (function call depth: " << FunctionCallDepth()
                      << ", simulate call depth: " << StackFrameDepth() << ").\n"
                      << "It's always happened with complex construction of code or infinite recursion or loop.\n"
                      << "Please check the code if it's has the infinite recursion "
                      << "or call 'context.set_context(max_call_depth=value)' to adjust this value.\n"
                      << "If max_call_depth is set larger, the system max stack depth should be set larger too "
                      << "to avoid stack overflow.\n"
                      << "For more details, please refer to the FAQ at https://www.mindspore.cn.";
  }
  MS_LOG(DEBUG) << evaluator << "(" << evaluator->type_name() << "/" << evaluator->ToString()
                << "), enter, function call depth: " << FunctionCallDepth() << " - " << StackFrameDepth();
}

void BaseFuncGraphEvaluator::LeaveStackFrame(const AnalysisEnginePtr &, const StackFramePtr &current_stack_frame) {
  MS_EXCEPTION_IF_NULL(current_stack_frame);
  // Leave current func graph.
  auto current_context = current_stack_frame->current_context();
  trace::TraceGraphEvalLeave(current_context);

  // Decrease the func graph call depth.
  DecreaseFunctionCallDepth();
  DecreaseStackFrameDepth();

  auto evaluator = current_stack_frame->evaluator();
  MS_EXCEPTION_IF_NULL(evaluator);
  MS_LOG(DEBUG) << evaluator << "(" << evaluator->type_name() << "/" << evaluator->ToString()
                << "), leave, function call depth: " << FunctionCallDepth() << " - " << StackFrameDepth();
}

// Start running stack frames in a Evaluator.
AbstractBasePtr BaseFuncGraphEvaluator::LaunchStackFrame(const AnalysisEnginePtr &engine, const FuncGraphPtr &fg,
                                                         const AnalysisContextPtr &context) {
  EvalResultPtr eval_result = nullptr;
  AbstractBasePtr res_base = nullptr;
  std::stack<StackFramePtr> stack_frames;
  auto current_stack_frame = std::make_shared<StackFrame>(shared_from_base<Evaluator>(), fg, context, parent_context_);
  MS_LOG(DEBUG) << "[" << this << "/StackFrame] Start at func graph, " << current_stack_frame;
  stack_frames.push(current_stack_frame);
  while (true) {
    current_stack_frame = stack_frames.top();
    if (current_stack_frame->Done()) {
      MS_EXCEPTION_IF_NULL(res_base);
      MS_LOG(DEBUG) << "[" << this << "/StackFrame] Leave from func graph, " << current_stack_frame;
      stack_frames.pop();
      if (stack_frames.empty()) {
        MS_LOG(DEBUG) << "[" << this << "/StackFrame] Finish at func graph, " << current_stack_frame
                      << ", res_base: " << res_base->ToString();
        break;
      }
      // Leave current func graph.
      LeaveStackFrame(engine, current_stack_frame);
      // Switch the stack frame.
      auto last_stack_frame = current_stack_frame;
      current_stack_frame = stack_frames.top();
      MS_LOG(DEBUG) << "[" << this << "/StackFrame] Back to func graph, " << current_stack_frame;
      current_stack_frame->Back(engine, last_stack_frame, eval_result);
      continue;
    }

    auto new_stack_frame = current_stack_frame->Jump(engine);
    if (new_stack_frame != nullptr) {
      // Enter new func graph.
      EnterStackFrame(engine, current_stack_frame, new_stack_frame);
      // Update current stack frame.
      stack_frames.push(new_stack_frame);
      MS_LOG(DEBUG) << "[" << this << "/StackFrame] Jump to new func graph, " << new_stack_frame;
      continue;
    }

    eval_result = current_stack_frame->Step(engine);
    MS_EXCEPTION_IF_NULL(eval_result);
    res_base = eval_result->abstract();
  }
  return res_base;
}

AbstractBasePtr BaseFuncGraphEvaluator::LaunchRecursiveEval(const AnalysisEnginePtr &engine, const FuncGraphPtr &fg,
                                                            const AnalysisContextPtr &context) {
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(engine);
  const AnfNodePtr &func_node = fg->get_return();
  const auto &all_nodes = TopoSort(func_node, SuccIncoming, [](const AnfNodePtr &node) -> IncludeType {
    MS_EXCEPTION_IF_NULL(node);
    if (node->isa<ValueNode>() || node->isa<Parameter>()) {
      return EXCLUDE;
    }
    return FOLLOW;
  });
  AbstractBasePtr res_base = nullptr;
  for (const auto &node : all_nodes) {
    AnfNodeConfigPtr node_conf = engine->MakeConfig(node, context, fg);
    MS_LOG(DEBUG) << "Analysis node begin, func graph: " << fg << "/" << fg->ToString()
                  << ", node_conf: " << node_conf->ToString();
    EvalResultPtr node_eval_result = nullptr;
    if (always_eval_flag()) {
      MS_LOG(DEBUG) << "Always eval node";
      node_eval_result = engine->ObtainEvalResultWithoutCache(node_conf);
    } else {
      node_eval_result = engine->ObtainEvalResultWithCache(node_conf);
    }
    MS_EXCEPTION_IF_NULL(node_eval_result);
    res_base = node_eval_result->abstract();
    MS_EXCEPTION_IF_NULL(res_base);
    MS_LOG(DEBUG) << GetInferThread() << "Eval ( " << node_conf->ToString() << ") = " << res_base->ToString();
  }
  MS_EXCEPTION_IF_NULL(res_base);
  return res_base;
}

EvalResultPtr BaseFuncGraphEvaluator::Eval(AnalysisEnginePtr engine, const AbstractBasePtrList &args_abs_list,
                                           const AnfNodeConfigPtr &out_conf) {
  auto eval_result = evaluator_cache_mgr_->GetValue(args_abs_list);
  if (eval_result != nullptr) {
    MS_LOG(ERROR) << ToString() << ArgsToString(args_abs_list) << " entered again. There is something wrong.";
    return eval_result;
  }
  MS_LOG(DEBUG) << ToString() << " entered first.";
  MS_EXCEPTION_IF_NULL(engine);
  // Increase & Check the func graph call depth.
  IncreaseFunctionCallDepth();
  const uint32_t max_depth = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_MAX_CALL_DEPTH);
  if (FunctionCallDepth() > max_depth) {
    MS_LOG(EXCEPTION) << "Exceed function call depth limit " << max_depth
                      << ", (function call depth: " << FunctionCallDepth()
                      << ", simulate call depth: " << StackFrameDepth() << ").\n"
                      << "It's always happened with complex construction of code or infinite recursion or loop.\n"
                      << "Please check the code if it's has the infinite recursion "
                      << "or call 'context.set_context(max_call_depth=value)' to adjust this value.\n"
                      << "If max_call_depth is set larger, the system max stack depth should be set larger too "
                      << "to avoid stack overflow.\n"
                      << "For more details, please refer to the FAQ at https://www.mindspore.cn.";
  }
  MS_LOG(DEBUG) << this << "(" << type_name() << "/" << ToString()
                << "), enter, function call depth: " << FunctionCallDepth() << " - " << StackFrameDepth();

  FuncGraphPtr fg = GetFuncGraph(engine, args_abs_list);
  MS_EXCEPTION_IF_NULL(fg);
  MS_EXCEPTION_IF_NULL(parent_context_);
  auto context = parent_context_->NewContext(fg, args_abs_list);
  trace::TraceGraphEvalEnter(context, out_conf);

  std::size_t nargs = fg->parameters().size();
  if (args_abs_list.size() != nargs) {
    MS_EXCEPTION(TypeError) << "The parameters number of the function is " << fg->parameters().size()
                            << ", but the number of provided arguments is " << args_abs_list.size() << ".\n"
                            << "FunctionGraph : " << fg->ToString()
                            << "\nNodeInfo: " << trace::GetDebugInfo(fg->debug_info());
  }
  MS_LOG(DEBUG) << GetInferThread() << "@" << fg->ToString() << ArgsToString(args_abs_list) << " { ";
  if (parent_context_->func_graph() != nullptr) {
    MS_LOG(DEBUG) << GetInferThread() << "graph_: " << AnalysisSchedule::thread_id() << ":"
                  << parent_context_->func_graph()->ToString() << "()->" << AnalysisSchedule::thread_id() << ":"
                  << fg->ToString() << "();";
  }

  auto func_graph_evaluator = dyn_cast<FuncGraphEvaluator>(shared_from_base<BaseFuncGraphEvaluator>());
  if (func_graph_evaluator != nullptr) {
    if (engine->root_func_graph() == func_graph_evaluator->func_graph()) {
      engine->set_root_context(context);
    }
  }
  bool always_eval_flag = false;
  const auto &parameters = fg->parameters();
  for (size_t i = 0; i < nargs; i++) {
    const auto &arg = args_abs_list[i];
    const auto &node = parameters[i];
    AnfNodeConfigPtr conf = engine->MakeConfig(node, context, fg);
    always_eval_flag = always_eval_flag || CheckIfAlwaysEval(conf, arg);
    engine->SaveEvalResultInCache(conf, std::make_shared<EvalResult>(arg, nullptr));
    MS_LOG(DEBUG) << GetInferThread() << ", Save argument[" << i << "] result for " << fg->ToString()
                  << ", NodeConfig: " << conf->ToString() << ", result: " << arg << "/" << arg->ToString();
  }
  PushAlwaysEvalFlag(always_eval_flag);

  MS_LOG(DEBUG) << "Analysis FuncGraph begin, func graph: " << fg << "/" << fg->ToString()
                << ", context: " << context->ToString() << ", return node: " << fg->get_return()->DebugString()
                << ", parent: " << (parent_context_->func_graph() ? parent_context_->func_graph()->ToString() : "NULL")
                << ", current function call depth: " << FunctionCallDepth();
  AbstractBasePtr res_base = nullptr;
  if (engine->enable_recursive_eval()) {
    res_base = LaunchRecursiveEval(engine, fg, context);
  } else {
    res_base = LaunchStackFrame(engine, fg, context);
  }
  PopAlwaysEvalFlag();

  MS_EXCEPTION_IF_NULL(res_base);
  MS_LOG(DEBUG) << "Analysis FuncGraph end, " << fg << "/" << fg->ToString()
                << ", evaluated abstract: " << res_base->ToString() << ", is stub: " << fg->stub();
  if (fg->stub()) {
    res_base = std::make_shared<AbstractUndetermined>();
  }
  MS_LOG(DEBUG) << GetInferThread() << "} //" << fg->ToString() << " = " << res_base->ToString();

  trace::TraceGraphEvalLeave(context);
  // Decrease the func graph call depth.
  DecreaseFunctionCallDepth();
  MS_LOG(DEBUG) << this << "(" << type_name() << "/" << ToString()
                << "), leave, function call depth: " << FunctionCallDepth() << " - " << StackFrameDepth();
  auto res = std::make_shared<EvalResult>(res_base, nullptr);
  return res;
}

void BroadenArgs(const AbstractBasePtrList &args_spec_list, AbstractBasePtrList *broaded_args) {
  MS_EXCEPTION_IF_NULL(broaded_args);
  (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(*broaded_args),
                       [](const AbstractBasePtr &arg) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(arg);
                         if (arg->GetValueTrack() != kAnyValue) {
                           return arg->Broaden();
                         }
                         return arg;
                       });
}

AbstractBasePtrList FuncGraphEvaluator::NormalizeArgs(const AbstractBasePtrList &args_spec_list) const {
  MS_EXCEPTION_IF_NULL(func_graph_);
  if (func_graph_->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE)) {
    AbstractBasePtrList broaded_list;
    BroadenArgs(args_spec_list, &broaded_list);
    MS_LOG(DEBUG) << func_graph_->ToString() << ", original: " << mindspore::ToString(args_spec_list)
                  << ", broadened: " << mindspore::ToString(broaded_list);
    return broaded_list;
  }
  return args_spec_list;
}

AbstractBasePtrList FuncGraphEvaluator::BroadenUndeterminedArgs(const AbstractBasePtrList &args_spec_list) {
  MS_EXCEPTION_IF_NULL(func_graph_);
  if (func_graph_->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE)) {
    return args_spec_list;
  }

  if (func_graph_->has_flag(kFuncGraphFlagUndetermined)) {
    func_graph_->set_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE, true);
    auto normalized_args_spec_list = NormalizeArgs(args_spec_list);
    MS_LOG(DEBUG) << "Set " << func_graph_->ToString() << " with IGNORE_VALUES flag.";
    MS_LOG(DEBUG) << "Normalized args " << mindspore::ToString(normalized_args_spec_list);
    return normalized_args_spec_list;
  }
  return args_spec_list;
}

FuncGraphPtr FuncGraphEvaluator::GetFuncGraph(AnalysisEnginePtr engine, const AbstractBasePtrList &args_spec_list) {
  auto iter = func_graph_cache_.find(args_spec_list);
  FuncGraphPtr res;
  if (iter == func_graph_cache_.end()) {
    auto fg = func_graph();
    MS_EXCEPTION_IF_NULL(fg);
    FuncGraphPtr generated_graph = fg->GenerateGraph(args_spec_list);
    func_graph_cache_[args_spec_list] = generated_graph;
    MS_LOG(DEBUG) << "Generate special instance of function graph: " << ToString()
                  << ", special function: " << generated_graph->ToString()
                  << ", args: " << ArgsToString(args_spec_list);

    MS_EXCEPTION_IF_NULL(engine);
    engine->func_graph_manager()->AddFuncGraph(generated_graph);
    res = generated_graph;
  } else {
    res = iter->second;
  }

  // For the top graph, if it is replaced by generated graph, update the top graph to the new one.
  if (parse::Parser::GetTopFuncGraph() == func_graph()) {
    if (res != func_graph()) {
      parse::Parser::UpdateTopFuncGraph(res);
    }
  }
  return res;
}

FuncGraphPtr MetaFuncGraphEvaluator::GetFuncGraph(AnalysisEnginePtr engine, const AbstractBasePtrList &args_spec_list) {
  auto iter = func_graph_cache_.find(args_spec_list);
  if (iter != func_graph_cache_.end()) {
    return iter->second;
  }

  MS_EXCEPTION_IF_NULL(meta_func_graph_);
  FuncGraphPtr generated_func_graph;
  if (this->bound_node() != nullptr) {
    TraceGuard trace_guard(std::make_shared<TraceGenMetaFuncGraph>(bound_node()->debug_info()));
    generated_func_graph = meta_func_graph_->GenerateFuncGraph(args_spec_list);
  } else {
    generated_func_graph = meta_func_graph_->GenerateFuncGraph(args_spec_list);
  }

  NodeDebugInfoPtr debug_info;
  if (this->bound_node() != nullptr) {
    debug_info = this->bound_node()->debug_info();
  }
  FuncGraphPtr cloned_func_graph =
    BasicClone(generated_func_graph, false, std::make_shared<UpdateInfo>(scope_, debug_info));
  func_graph_cache_[args_spec_list] = cloned_func_graph;
  MS_EXCEPTION_IF_NULL(engine);
  engine->func_graph_manager()->AddFuncGraph(cloned_func_graph);
  return cloned_func_graph;
}

EvalResultPtr Evaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                             const AnfNodeConfigPtr &out_conf) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->ObtainEvalResult()->abstract();
                       });
  args_spec_list = NormalizeArgs(args_spec_list);
  args_spec_list = BroadenUndeterminedArgs(args_spec_list);

  MS_LOG(DEBUG) << EvalEntryLogging(shared_from_base<Evaluator>(), args_spec_list, out_conf);
  EvalResultPtr eval_result = nullptr;
  const std::string &evaluator_name = ToString();
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto &cache = evaluator_cache_mgr_->GetCache();
  auto iter = cache.find(args_spec_list);
  if (iter == cache.end()) {
    MS_LOG(DEBUG) << "[" << this << "/" << evaluator_name << "] cache miss, call Eval(), args: " << args_spec_list;
    eval_result = Eval(engine, args_spec_list, out_conf);
    MS_EXCEPTION_IF_NULL(eval_result);
    if (eval_result->abstract() == nullptr) {
      EvalFailLogging(shared_from_base<Evaluator>(), args_spec_list, out_conf);
      MS_LOG(EXCEPTION) << "Evaluator " << evaluator_name << " result is nullptr.";
    }
    MS_LOG(DEBUG) << "[" << this << "/" << evaluator_name
                  << "] set cache. result: " << eval_result->abstract()->ToString();
    evaluator_cache_mgr_->SetValue(args_spec_list, eval_result);
  } else {
    eval_result = iter->second;
    MS_EXCEPTION_IF_NULL(eval_result->abstract());
    MS_LOG(DEBUG) << "[" << this << "/" << evaluator_name
                  << "] cache hit. result: " << eval_result->abstract()->ToString() << ", args: " << args_spec_list;
    // Update inputs sequence nodes info, if matched in cache.
    static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
    if (enable_eliminate_unused_element) {
      for (size_t i = 0; i < args_spec_list.size(); ++i) {
        auto new_sequence = dyn_cast<AbstractSequence>(args_spec_list[i]);
        auto old_sequence = dyn_cast<AbstractSequence>(iter->first[i]);
        if (old_sequence != nullptr && new_sequence != nullptr) {
          MS_LOG(DEBUG) << "Before synchronize sequence nodes use flags for NodeConfig: " << out_conf->ToString()
                        << ", old_sequence: " << old_sequence->ToString()
                        << ", new_sequence: " << new_sequence->ToString();
          SynchronizeSequenceElementsUseFlagsRecursively(old_sequence, new_sequence);
          MS_LOG(DEBUG) << "After synchronize sequence nodes use flags for NodeConfig: " << out_conf->ToString()
                        << ", old_sequence: " << old_sequence->ToString()
                        << ", new_sequence: " << new_sequence->ToString();
        }
      }
    }
  }
  return eval_result;
}

EvalResultPtr TrivialPrimEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                        const AnfNodeConfigPtr &) {
  AbstractBasePtrList args_spec_list;
  auto is_py_eval = (identifier_ == "PythonPrimEvaluator");
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [is_py_eval](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         auto abstract = conf->ObtainEvalResult()->abstract();
                         MS_EXCEPTION_IF_NULL(abstract);
                         // Broaden the ref_key, while infer python prim for cache
                         if (is_py_eval && abstract->isa<AbstractRef>()) {
                           auto abs_ref = abstract->cast<AbstractRefPtr>();
                           abstract = std::make_shared<AbstractRef>(abs_ref->ref_key()->Broaden(), abs_ref);
                         }
                         return abstract;
                       });
  return EvalPrim(engine, args_spec_list);
}

EvalResultPtr TransitionPrimEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                           const AnfNodeConfigPtr &out_conf) {
  if (args_conf_list.empty() && identifier_ != "MakeTupleEvaluator" && identifier_ != "MakeListEvaluator" &&
      identifier_ != "RaiseEvaluator") {
    MS_LOG(EXCEPTION) << "Size should be greater than 0, during running " << identifier_;
  }
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->ObtainEvalResult()->abstract();
                       });
  EvalResultPtr res = EvalPrim(engine, args_spec_list, args_conf_list[0], out_conf);
  // No need to cache.
  return res;
}

EvalResultPtr SymbolicPrimEvaluator::Run(AnalysisEnginePtr, const ConfigPtrList &args_conf_list,
                                         const AnfNodeConfigPtr &) {
  return EvalPrim(args_conf_list);
}

EvalResultPtr TrackedEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                    const AnfNodeConfigPtr &out_conf) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->ObtainEvalResult()->abstract();
                       });
  EvalResultPtr res = sub_evaluator_->Run(engine, args_conf_list, out_conf);
  // Don't lookup from cache, as different out_conf with same node but different context
  // may add different entry to anfnode_config_map_, like getattr primitive.
  evaluator_cache_mgr_->SetValue(args_spec_list, res);
  return res;
}

EvalResultPtr PartialAppEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                       const AnfNodeConfigPtr &out_conf) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->ObtainEvalResult()->abstract();
                       });
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto eval_result = evaluator_cache_mgr_->GetValue(args_spec_list);
  if (eval_result != nullptr) {
    return eval_result;
  }

  ConfigPtrList partial_args_conf_list;
  // Join arguments in partial and the rest arguments from args_conf_list.
  (void)std::transform(args_spec_list_.begin(), args_spec_list_.end(), std::back_inserter(partial_args_conf_list),
                       [](const AbstractBasePtr &arg) -> ConfigPtr { return std::make_shared<VirtualConfig>(arg); });

  (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(partial_args_conf_list),
                       [](const AbstractBasePtr &arg) -> ConfigPtr { return std::make_shared<VirtualConfig>(arg); });
  EvalResultPtr res = evaluator_->Run(engine, partial_args_conf_list, out_conf);
  evaluator_cache_mgr_->SetValue(args_spec_list, res);
  return res;
}

EvalResultPtr JEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list, const AnfNodeConfigPtr &) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->ObtainEvalResult()->abstract();
                       });
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto eval_result = evaluator_cache_mgr_->GetValue(args_spec_list);
  if (eval_result != nullptr) {
    return eval_result;
  }

  // Call the original evaluator, get the result: y = f(x)
  EvalResultPtr result = evaluator_->Run(engine, args_conf_list, nullptr);
  MS_EXCEPTION_IF_NULL(result);
  // If the primal func graph's output is sequence, set its elements use flags all true.
  SetSequenceElementsUseFlagsRecursively(result->abstract(), true);
  // Build a virtual function: bprop_f which use sense of y as input, return sense of function free variable and input
  // parameters. (sense_f, sense_x, ...)(*bpro_f) (sense_y)
  AbstractBasePtrList bparams;
  bparams.push_back(SensitivityTransform(orig_func_));
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool enable_sparse = context->get_param<bool>(MS_CTX_ENABLE_SPARSE);
  (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(bparams),
                       [&enable_sparse](const AbstractBasePtr &arg_spec) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(arg_spec);
                         if (enable_sparse && arg_spec->isa<AbstractTensor>()) {
                           return std::make_shared<AbstractUndetermined>();
                         }
                         return SensitivityTransform(arg_spec);
                       });
  AbstractBasePtr bparams_final = std::make_shared<AbstractTuple>(bparams);
  AbstractFunctionPtr bprop =
    std::make_shared<VirtualAbstractClosure>(SensitivityTransform(result->abstract()), bparams_final);

  // J(f)(J(x)) return a tuple (y, bprop_f)
  AbstractBasePtrList jargs = {result->abstract(), bprop};
  AbstractBasePtr jtuple = std::make_shared<AbstractTuple>(jargs);
  auto res = std::make_shared<EvalResult>(jtuple, std::make_shared<AttrValueMap>());
  evaluator_cache_mgr_->SetValue(args_spec_list, res);
  return res;
}

EvalResultPtr TaylorEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                   const AnfNodeConfigPtr &) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->ObtainEvalResult()->abstract();
                       });
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto eval_result = evaluator_cache_mgr_->GetValue(args_spec_list);
  if (eval_result != nullptr) {
    return eval_result;
  }

  // Call the original evaluator, get the result: y = f(x)
  EvalResultPtr result = evaluator_->Run(engine, args_conf_list, nullptr);
  MS_EXCEPTION_IF_NULL(result);
  evaluator_cache_mgr_->SetValue(args_spec_list, result);
  return result;
}

EvalResultPtr ShardEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                  const AnfNodeConfigPtr &) {
  AbstractBasePtrList args_spec_list;
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         return conf->ObtainEvalResult()->abstract();
                       });
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto eval_result = evaluator_cache_mgr_->GetValue(args_spec_list);
  if (eval_result != nullptr) {
    return eval_result;
  }

  // Call the original evaluator, get the result: y = f(x)
  EvalResultPtr result = evaluator_->Run(engine, args_conf_list, nullptr);
  MS_EXCEPTION_IF_NULL(result);
  auto res = std::make_shared<EvalResult>(result->abstract(), std::make_shared<AttrValueMap>());
  evaluator_cache_mgr_->SetValue(args_spec_list, res);
  return res;
}

namespace {
AbstractBasePtr ReduceDim(int *axis, const AbstractBasePtr &orig_abs, int *axis_size) {
  if (!orig_abs->isa<abstract::AbstractTensor>()) {
    MS_LOG(EXCEPTION) << "ValueError: orig_abs should be AbstractTensor, but got a " << orig_abs->ToString() << ".";
  }
  ShapeVector orig_shape = dyn_cast<abstract::Shape>(orig_abs->BuildShape())->shape();
  int shape_len = SizeToInt(orig_shape.size());
  if (*axis < -shape_len || *axis >= shape_len) {
    MS_LOG(EXCEPTION) << "ValueError: The axis: " << *axis << " in 'in_axes' is out of bounds for array of dimension ["
                      << -shape_len << "," << shape_len << ").";
  }
  *axis = *axis < 0 ? shape_len + *axis : *axis;
  auto temp_axes_size = orig_shape[IntToSize(*axis)];
  if (*axis_size == -1) {
    *axis_size = LongToInt(temp_axes_size);
  } else if (*axis_size != temp_axes_size) {
    MS_LOG(EXCEPTION) << "The 'axis_size' of each argument in the scope of 'vmap' should be equal, but got "
                      << *axis_size << " and " << temp_axes_size << ".";
  }
  (void)orig_shape.erase(orig_shape.begin() + *axis);
  BaseShapePtr new_shape = std::make_shared<abstract::Shape>(orig_shape);
  AbstractBasePtr abs_clone = orig_abs->Clone()->Broaden();
  abs_clone->set_shape(new_shape);
  return abs_clone;
}

AbstractBasePtr GetLogicalViewAbs(const AbstractBasePtr &physical_view_abs, const ValuePtr &in_axes, int *axis_size) {
  MS_EXCEPTION_IF_NULL(physical_view_abs);
  MS_EXCEPTION_IF_NULL(in_axes);
  auto physical_view_abs_sequence = dyn_cast<abstract::AbstractSequence>(physical_view_abs);
  if (physical_view_abs_sequence != nullptr) {
    AbstractBasePtrList abs_list = physical_view_abs_sequence->elements();
    AbstractBasePtrList logical_view_abs_list;
    auto in_axes_seq = dyn_cast<ValueSequeue>(in_axes);
    int index = 0;
    (void)std::transform(
      abs_list.begin(), abs_list.end(), std::back_inserter(logical_view_abs_list),
      [&axis_size, &index, &in_axes_seq, in_axes](const AbstractBasePtr &sub_abs) -> AbstractBasePtr {
        ValuePtr sub_in_axes = in_axes;
        if (in_axes->isa<ValueSequeue>()) {
          sub_in_axes = (*in_axes_seq)[index];
          index++;
        }
        return GetLogicalViewAbs(sub_abs, sub_in_axes, axis_size);
      });
    if (physical_view_abs->isa<AbstractList>()) {
      return std::make_shared<AbstractList>(logical_view_abs_list);
    }
    return std::make_shared<AbstractTuple>(logical_view_abs_list);
  }
  ValuePtr in_axis = in_axes;
  if (in_axis->isa<Int64Imm>()) {
    int axis = dyn_cast<Int64Imm>(in_axis)->value();
    auto logical_view_abs = ReduceDim(&axis, physical_view_abs, axis_size);
    return logical_view_abs;
  }
  if (!in_axis->isa<None>()) {
    MS_LOG(EXCEPTION) << "The axis in vmap's 'in_axes' should be a None or a scalar of type Int64Imm, but got a "
                      << in_axis->ToString() << ".";
  }
  // in_axis is None.
  return physical_view_abs;
}

AbstractBasePtr ExtendDim(int *axis, const AbstractBasePtr &orig_abs, int axis_size) {
  MS_EXCEPTION_IF_NULL(orig_abs);
  AbstractBasePtr out_abs = nullptr;
  ShapeVector orig_shape;
  if (orig_abs->isa<abstract::AbstractTensor>()) {
    orig_shape = dyn_cast<abstract::Shape>(orig_abs->BuildShape())->shape();
  }
  int shape_len = SizeToInt(orig_shape.size() + 1);
  if (*axis < -shape_len || *axis >= shape_len) {
    MS_LOG(EXCEPTION) << "ValueError: The axis: " << *axis << " in 'out_axes' is out of bounds for array of dimension ["
                      << -shape_len << "," << shape_len << ").";
  }
  *axis = *axis < 0 ? shape_len + *axis : *axis;
  (void)orig_shape.insert(orig_shape.begin() + *axis, axis_size);
  BaseShapePtr new_shape = std::make_shared<abstract::Shape>(orig_shape);
  if (orig_abs->isa<abstract::AbstractTensor>()) {
    out_abs = orig_abs->Clone()->Broaden();
    out_abs->set_shape(new_shape);
  } else if (orig_abs->isa<AbstractScalar>()) {
    out_abs = std::make_shared<abstract::AbstractTensor>(orig_abs, new_shape);
  } else {
    MS_LOG(EXCEPTION) << "The outputs of vmap's 'fn' should be consisting of tensors or constants, but got "
                      << orig_abs->ToString() << ".";
  }
  return out_abs;
}

AbstractBasePtr GetPhysicalViewAbs(const AbstractBasePtr &logical_view_abs, const ValuePtr &out_axes, int axis_size) {
  MS_EXCEPTION_IF_NULL(logical_view_abs);
  auto logical_view_abs_sequence = dyn_cast<abstract::AbstractSequence>(logical_view_abs);
  if (logical_view_abs_sequence != nullptr) {
    AbstractBasePtrList logical_view_abs_list = logical_view_abs_sequence->elements();
    AbstractBasePtrList physical_view_abs_list;
    auto out_axes_seq = dyn_cast<ValueSequeue>(out_axes);
    if (out_axes_seq != nullptr) {
      if (logical_view_abs_list.size() != out_axes_seq->size()) {
        MS_LOG(EXCEPTION) << "The size of vmap's 'out_axes' should be equal to the number of results of 'fn': "
                          << logical_view_abs_list.size() << ", but got size: " << out_axes_seq->size() << ".";
      }
    }
    int index = 0;
    (void)std::transform(
      logical_view_abs_list.begin(), logical_view_abs_list.end(), std::back_inserter(physical_view_abs_list),
      [&axis_size, &index, &out_axes_seq, out_axes](const AbstractBasePtr &arg_spec) -> AbstractBasePtr {
        ValuePtr sub_out_axes = out_axes;
        if (out_axes->isa<ValueSequeue>()) {
          sub_out_axes = (*out_axes_seq)[index];
          index++;
        }
        if (arg_spec->isa<AbstractSequence>()) {
          return GetPhysicalViewAbs(arg_spec, sub_out_axes, axis_size);
        }
        if (sub_out_axes->isa<Int64Imm>()) {
          int axis = dyn_cast<Int64Imm>(sub_out_axes)->value();
          return ExtendDim(&axis, arg_spec, axis_size);
        } else if (sub_out_axes->isa<None>()) {
          return arg_spec;
        }
        MS_LOG(EXCEPTION) << "The axis in vmap's 'out_axes' should be a None or a scalar of type Int64Imm, but got a "
                          << sub_out_axes->ToString() << ".";
      });
    if (logical_view_abs->isa<AbstractList>()) {
      return std::make_shared<AbstractList>(physical_view_abs_list);
    }
    return std::make_shared<AbstractTuple>(physical_view_abs_list);
  }

  // for the single output case, outputs: A, and out_axes: 1 or (1,).
  ValuePtr sub_out_axes = out_axes;
  ValueSequeuePtr out_axes_seq = dyn_cast<ValueSequeue>(out_axes);
  if (out_axes_seq != nullptr) {
    if (out_axes_seq->size() != 1) {
      MS_LOG(EXCEPTION) << "The  size of vmap's 'out_axes' should be equal to the result size: 1, but got size: "
                        << out_axes_seq->size() << ".";
    }
    sub_out_axes = (*out_axes_seq)[0];
  }

  int axis = 0;
  auto axis_int_ptr = dyn_cast<Int64Imm>(sub_out_axes);
  if (axis_int_ptr != nullptr) {
    axis = LongToInt(axis_int_ptr->value());
  } else {
    MS_LOG(EXCEPTION) << "The axis in vmap's 'out_axes' should be a None or a scalar of type Int64Imm, but got a "
                      << sub_out_axes->ToString() << ".";
  }
  return ExtendDim(&axis, logical_view_abs, axis_size);
}
}  // namespace

// According to the in_axes (e.g. (1,(None,3))), the abstraction of input parameters with the
// physical view (e.g. (A,(B,C))) are converted into that with the logical view (e.g.(a,(b,c))),
// more specific, the input `A` with shape (32, 16, 8) fitting the axis index `1` is converted in to
// `a` with shape (32, 8). And then leverage the original graph to perform the evaluation.
// Finally, the outputs with the logical view are converted back into the physical view in
// combination with the out_axes. The inferring result is consistent with that after eliminating
// the VmapOperator.
EvalResultPtr VmapEvaluator::Run(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                 const AnfNodeConfigPtr &) {
  AbstractBasePtrList args_spec_list;
  int axis_size = -1;
  int index = 0;
  auto in_axes = in_axes_;
  auto in_axes_seq = dyn_cast<ValueSequeue>(in_axes);
  (void)std::transform(args_conf_list.begin(), args_conf_list.end(), std::back_inserter(args_spec_list),
                       [&axis_size, &index, &in_axes_seq, in_axes](const ConfigPtr &conf) -> AbstractBasePtr {
                         MS_EXCEPTION_IF_NULL(conf);
                         AbstractBasePtr abs = conf->ObtainEvalResult()->abstract();
                         // Drop the side effect tag parameters, because it has no mapping axis.
                         // e.g. args=(A,(B,C),U), in_axes=(1,(None,3))
                         if (abs->isa<AbstractMonad>()) {
                           return abs;
                         }
                         ValuePtr sub_in_axes = in_axes;
                         if (in_axes->isa<ValueSequeue>()) {
                           sub_in_axes = (*in_axes_seq)[index];
                           index++;
                         }
                         auto arg_abs = GetLogicalViewAbs(abs, sub_in_axes, &axis_size);
                         return arg_abs;
                       });
  MS_EXCEPTION_IF_NULL(evaluator_cache_mgr_);
  auto eval_result = evaluator_cache_mgr_->GetValue(args_spec_list);
  if (eval_result != nullptr) {
    return eval_result;
  }
  ConfigPtrList virtual_conf_list;
  (void)std::transform(args_spec_list.begin(), args_spec_list.end(), std::back_inserter(virtual_conf_list),
                       [](const AbstractBasePtr &arg) -> ConfigPtr { return std::make_shared<VirtualConfig>(arg); });

  // Call the original evaluator, get the result: y = f(x)
  EvalResultPtr result = evaluator_->Run(engine, virtual_conf_list, nullptr);
  MS_EXCEPTION_IF_NULL(result);

  // If the primal func graph's output is sequence, set its elements use flags all true.
  SetSequenceElementsUseFlagsRecursively(result->abstract(), true);

  AbstractBasePtr result_abs = result->abstract();
  AbstractBasePtr after_vmap = GetPhysicalViewAbs(result_abs, out_axes_, axis_size);

  auto res = std::make_shared<EvalResult>(after_vmap, std::make_shared<AttrValueMap>());
  evaluator_cache_mgr_->SetValue(args_spec_list, res);
  return res;
}

EvalResultPtr VirtualEvaluator::Eval(AnalysisEnginePtr, const AbstractBasePtrList &args_spec_list,
                                     const AnfNodeConfigPtr &out_conf) {
  if (args_spec_list.size() != args_spec_list_.size()) {
    MS_LOG(EXCEPTION) << "Arguments mismatch, parameters no: " << args_spec_list_.size()
                      << ", arguments no: " << args_spec_list.size();
  }
  static const auto enable_eliminate_unused_element = (common::GetEnv("MS_DEV_ENABLE_DDE") != "0");
  // Check each parameter and argument match;
  for (std::size_t i = 0; i < args_spec_list.size(); i++) {
    MS_EXCEPTION_IF_NULL(args_spec_list[i]);
    // For VirtualAbstractClosure, likely J's bprop, we just set its tuple arguments as used before really grad.
    if (enable_eliminate_unused_element && args_spec_list[i]->isa<abstract::AbstractSequence>()) {
      MS_LOG(INFO) << "Notice: For VirtualAbstractClosure, update all use flags as true for arguments[" << i
                   << "]: " << args_spec_list[i]->ToString();
      SetSequenceElementsUseFlagsRecursively(args_spec_list[i], true);
    }
    (void)args_spec_list[i]->Join(args_spec_list_[i]);
  }
  return std::make_shared<EvalResult>(output_, std::make_shared<AttrValueMap>());
}
EvalResultPtr Evaluator::SingleRun(AnalysisEnginePtr engine, const ConfigPtrList &args_conf_list,
                                   const AnfNodeConfigPtr &out_conf) {
  EvalResultPtr result;
  try {
    result = this->Run(engine, args_conf_list, out_conf);
  } catch (const std::exception &ex) {
    MS_LOG(INFO) << "Eval " << ToString() << " throw exception.";
    AnalysisSchedule::GetInstance().HandleException(ex);
  }
  AnalysisSchedule::GetInstance().Wait();
  return result;
}
}  // namespace abstract
}  // namespace mindspore
