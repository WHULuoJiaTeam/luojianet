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

#ifndef COMMON_GRAPH_DEBUG_GE_OP_TYPES_H_
#define COMMON_GRAPH_DEBUG_GE_OP_TYPES_H_

#include "graph/types.h"
#include "graph/compiler_options.h"

#define GE_REGISTER_OPTYPE(var_name, str_name) static const ge::char_t *(var_name) METADEF_ATTRIBUTE_UNUSED = (str_name)
namespace ge {
GE_REGISTER_OPTYPE(DATA, "Data");
GE_REGISTER_OPTYPE(AIPPDATA, "AippData");
GE_REGISTER_OPTYPE(MATMUL, "MatMul");
GE_REGISTER_OPTYPE(RESHAPE, "Reshape");
GE_REGISTER_OPTYPE(PERMUTE, "Permute");
GE_REGISTER_OPTYPE(NETOUTPUT, "NetOutput");
GE_REGISTER_OPTYPE(_WHILE, "_While");
GE_REGISTER_OPTYPE(WHILE, "While");
GE_REGISTER_OPTYPE(STATELESSWHILE, "StatelessWhile");
GE_REGISTER_OPTYPE(SQUEEZE, "Squeeze");
GE_REGISTER_OPTYPE(EXPANDDIMS, "ExpandDims");
GE_REGISTER_OPTYPE(SWITCH, "Switch");
GE_REGISTER_OPTYPE(REFSWITCH, "RefSwitch");
GE_REGISTER_OPTYPE(SWITCHN, "SwitchN");
GE_REGISTER_OPTYPE(MERGE, "Merge");
GE_REGISTER_OPTYPE(REFMERGE, "RefMerge");
GE_REGISTER_OPTYPE(STREAMMERGE, "StreamMerge");
GE_REGISTER_OPTYPE(STREAMSWITCH, "StreamSwitch");
GE_REGISTER_OPTYPE(STREAMACTIVE, "StreamActive");
GE_REGISTER_OPTYPE(LABELSET, "LabelSet");
GE_REGISTER_OPTYPE(LABELGOTOEX, "LabelGotoEx");
GE_REGISTER_OPTYPE(LABELSWITCHBYINDEX, "LabelSwitchByIndex");
GE_REGISTER_OPTYPE(ENTER, "Enter");
GE_REGISTER_OPTYPE(REFENTER, "RefEnter");
GE_REGISTER_OPTYPE(NEXTITERATION, "NextIteration");
GE_REGISTER_OPTYPE(REFNEXTITERATION, "RefNextIteration");
GE_REGISTER_OPTYPE(CONSTANT, "Const");
GE_REGISTER_OPTYPE(PLACEHOLDER, "PlaceHolder");
GE_REGISTER_OPTYPE(END, "End");
GE_REGISTER_OPTYPE(FRAMEWORKOP, "FrameworkOp");
GE_REGISTER_OPTYPE(GETNEXT, "GetNext");
GE_REGISTER_OPTYPE(INITDATA, "InitData");
GE_REGISTER_OPTYPE(REFIDENTITY, "RefIdentity");
GE_REGISTER_OPTYPE(ANN_DATA, "AnnData");
GE_REGISTER_OPTYPE(ATOMICADDRCLEAN, "AtomicAddrClean");

GE_REGISTER_OPTYPE(CONSTANTOP, "Constant");
GE_REGISTER_OPTYPE(VARIABLE, "Variable");
GE_REGISTER_OPTYPE(VARIABLEV2, "VariableV2");
GE_REGISTER_OPTYPE(PARTITIONEDCALL, "PartitionedCall");

GE_REGISTER_OPTYPE(INPUT_TYPE, "Input");
GE_REGISTER_OPTYPE(FILECONSTANT, "FileConstant");

// Horovod operator
GE_REGISTER_OPTYPE(HVDCALLBACKALLREDUCE, "hvdCallbackAllreduce");
GE_REGISTER_OPTYPE(HVDCALLBACKALLGATHER, "hvdCallbackAllgather");
GE_REGISTER_OPTYPE(HVDCALLBACKBROADCAST, "hvdCallbackBroadcast");
GE_REGISTER_OPTYPE(HVDWAIT, "hvdWait");

GE_REGISTER_OPTYPE(NODE_NAME_NET_OUTPUT, "Node_Output");

GE_REGISTER_OPTYPE(RECV, "Recv");
GE_REGISTER_OPTYPE(SEND, "Send");
GE_REGISTER_OPTYPE(NOOP, "NoOp");
}  // namespace ge
#endif  // COMMON_GRAPH_DEBUG_GE_OP_TYPES_H_
