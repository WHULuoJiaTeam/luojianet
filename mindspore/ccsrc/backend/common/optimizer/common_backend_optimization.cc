/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#include "backend/common/optimizer/common_backend_optimization.h"
#include <memory>
#include <string>
#include "backend/common/optimizer/optimizer.h"
#include "backend/common/pass/eliminate_func_data_type.h"
#include "backend/common/pass/convert_const_input_to_attr.h"
#include "backend/common/pass/custom_op_const_input_to_attr.h"
#include "backend/common/pass/custom_op_reg_info_to_attr.h"
#include "backend/common/pass/convert_tuple_output_to_maketuple.h"
#include "backend/common/pass/convert_const_input_to_tensor_input.h"
#include "backend/common/pass/convert_tuple_input_to_dynamic_input.h"
#include "backend/common/pass/convert_const_scalar_to_tensor.h"
#include "backend/common/pass/convert_attr_to_unify_mindir.h"
#include "backend/common/pass/add_training_attr.h"
#include "backend/common/pass/optimize_updatestate.h"
#include "backend/common/pass/conv_transpose_to_conv_bp.h"
#include "backend/common/pass/reduce_sum_optimizer.h"
#include "backend/common/pass/add_dynamic_shape_attr.h"
#include "backend/common/pass/add_akg_kernel_attrs.h"
#include "backend/common/pass/sparse_process.h"
#include "backend/common/pass/insert_assign_for_custom_op.h"
#include "backend/common/optimizer/dynamic_shape/convert_custom_op.h"
#include "backend/common/optimizer/dynamic_shape/link_custom_op.h"
#include "utils/ms_context.h"
#include "include/common/debug/anf_ir_dump.h"

namespace mindspore {
namespace opt {
void BackendCommonOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_LOG(INFO) << "Status record: start common optimization. graph id: " << kernel_graph->graph_id();
  PROF_START(backend_common_optimization);
  MS_EXCEPTION_IF_NULL(kernel_graph);
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string file_name = "hwopt_common_before_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto common_pm = std::make_shared<PassManager>("common_pm");
  common_pm->AddPass(std::make_shared<AddDynamicShapeAttr>());
  common_pm->AddPass(std::make_shared<ReduceSumOptimizer>());
  common_pm->AddPass(std::make_shared<ConvertConstInputToAttr>());
  common_pm->AddPass(std::make_shared<CustomOpConstInputToAttr>());
  common_pm->AddPass(std::make_shared<SparseProcess>());
  common_pm->AddPass(std::make_shared<ConvertAttrToUnifyMindIR>());
  common_pm->AddPass(std::make_shared<ConvertConstInputToTensorInput>());
  common_pm->AddPass(std::make_shared<ConvertTupleOutputToMaketuple>());
  common_pm->AddPass(std::make_shared<ConvertConstScalarToTensor>());
  common_pm->AddPass(std::make_shared<ConvertTupleInputToDynamicInput>());
  common_pm->AddPass(std::make_shared<AddTrainingAttr>());
  optimizer->AddPassManager(common_pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
  PROF_END(backend_common_optimization);
#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name = "hwopt_common_after_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  MS_LOG(INFO) << "Status record: end common optimization. graph id: " << kernel_graph->graph_id();
}

void CommonFinalOptimization(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  // Run optimizer passes.
  auto optimizer = std::make_shared<GraphOptimizer>();
  auto pm = std::make_shared<PassManager>("final_opt");
  pm->AddPass(std::make_shared<OptimizeUpdateState>());
  pm->AddPass(std::make_shared<AddAkgKernelAttrs>());
  optimizer->AddPassManager(pm);
  (void)optimizer->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  // Dump IR if save_graphs is set.
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const bool save_graphs = context->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string filename = "hwopt_common_final_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(filename, kernel_graph);
  }
#endif
}

void CommonUnifyMindIR(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "start common unify mindir opt graph:" << kernel_graph->graph_id();
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string file_name =
      "hwopt_common_unify_mindir_before_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  auto opt = std::make_shared<GraphOptimizer>();
  auto pm = std::make_shared<PassManager>("common_unify_mindir_pm");
  pm->AddPass(std::make_shared<ConvTransposeToConvBackpropInputPass>());
  pm->AddPass(std::make_shared<CustomOpRegInfoToAttr>());
  pm->AddPass(std::make_shared<InsertAssignForCustomOp>());
  opt->AddPassManager(pm);
  (void)opt->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name = "hwopt_common_unify_mindir_after_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
}

void AddDynamicShapeAttrPass(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  auto opt = std::make_shared<GraphOptimizer>();
  auto pm = std::make_shared<PassManager>("add_dynamic_shape_attr");
  pm->AddPass(std::make_shared<AddDynamicShapeAttr>());
  opt->AddPassManager(pm);
  (void)opt->Optimize(kernel_graph);
}

void EliminateIllegalDataTypePass(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Start eliminate illegal data type for kernel graph id:" << kernel_graph->graph_id();
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string file_name =
      "hwopt_common_eliminate_illegal_data_type_before_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  auto opt = std::make_shared<GraphOptimizer>();
  auto pm = std::make_shared<PassManager>("common_eliminate_illegal_data_type_pm");
  pm->AddPass(std::make_shared<EliminateFuncDataType>());
  opt->AddPassManager(pm);
  (void)opt->Optimize(kernel_graph);
  kernel_graph->SetExecOrderByDefault();
#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name =
      "hwopt_common_eliminate_illegal_data_type_after_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
}

void DynamicShapeConvertPass(const std::shared_ptr<session::KernelGraph> &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Start dynamic shape convert for kernel graph id:" << kernel_graph->graph_id();
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string file_name =
      "hwopt_d_before_dynamic_shape_convert_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto dynamic_shape_convert_pm = std::make_shared<opt::PassManager>("dynamic_shape_convert_pm");
  dynamic_shape_convert_pm->AddPass(std::make_shared<opt::dynamic_shape::ConvertCustomOp>());
  dynamic_shape_convert_pm->AddPass(std::make_shared<opt::dynamic_shape::LinkCustomOp>());
  optimizer->AddPassManager(dynamic_shape_convert_pm);
  (void)optimizer->Optimize(kernel_graph);
#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name =
      "hwopt_d_after_dynamic_shape_convert_graph_" + std::to_string(kernel_graph->graph_id()) + ".ir";
    DumpIR(file_name, kernel_graph);
  }
#endif
}
}  // namespace opt
}  // namespace mindspore
