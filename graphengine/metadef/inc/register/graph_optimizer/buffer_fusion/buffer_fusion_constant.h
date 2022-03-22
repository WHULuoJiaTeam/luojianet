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

#ifndef INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_CONSTANT_H_
#define INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_CONSTANT_H_
#include <string>
#include <vector>

namespace fe {
const std::string UB_FUSION_OP_TYPE = "_ub_fusion_op_type";
// add the op pattern
const std::string TBE_PATTERN_INPUT_NODE = "InputData";
const std::string TBE_PATTERN_OP_TYPE_ANY = "OpTypeAny";
const std::string TBE_PATTERN_OUTPUT_NODE = "OutputData";
const std::string OP_PATTERN_ELEMWISE = "ElemWise";
const std::string OP_PATTERN_COMMONREDUCE = "CommReduce";
const std::string OP_PATTERN_BROAD_CAST = "Broadcast";
const std::string OP_PATTERN_SEGMENT = "Segment";
const std::string OP_PATTERN_MAXPOOL = "MaxPool";
const std::string OP_PATTERN_CONV = "Convolution";
const std::string OP_PATTERN_MATMUL = "Matmul";
const std::string OP_PATTERN_BNUPDATE = "bn_update";
const std::string OP_PATTERN_BNREDUCE = "bn_reduce";
const std::string OP_PATTERN_CONV_BACKPROP_INPUT = "Conv2d_backprop_input";
const std::string OP_PATTERN_DEPTHWISE_CONV = "DepthwiseConvolution";
const std::string OP_PATTERN_QUANT = "quant";
const std::string OP_PATTERN_DEQUANT = "dequant";
const std::string OP_PATTERN_REQUANT = "requant";
const std::string OP_PATTERN_POOL2D = "Pool2d";
const std::string OP_PATTERN_ANTIQUANT = "anti_quant";
const std::string OP_PATTERN_STRIDED_WRITE = "strided_write";
const std::string OP_PATTERN_STRIDED_READ = "strided_read";
const std::string OP_PATTERN_AIPP = "aipp";
const std::string OP_PATTERN_CONFUSION_TRANSPOSE = "confusiontranspose";
const std::string OP_PATTERN_DEQUANTS16 = "dequant_s16";
const std::string OP_PATTERN_REQUANTS16 = "requant_s16";
const std::string OP_PATTERN_READ_SELECT = "read_select";
const std::string OP_PATTERN_WRITE_SELECT = "write_select";
const std::string OP_PATTERN_BATCH_MATMUL = "BatchMatmul";
const std::string OP_PATTERN_CONV3D = "Conv3d";
const std::string OP_PATTERN_DROPOUTDOMASKV3D = "DropOutDoMaskV3D";
const std::string OP_PATTERN_CONV3D_BACKPROP_INPUT = "Conv3d_backprop_input";
const std::string OP_PATTERN_CONV_BACKPROP_FILTER = "Conv2d_backprop_filter";
const std::string OP_PATTERN_GEMM = "GEMM";

const std::vector<std::string> OP_PATTERN_VEC{OP_PATTERN_ELEMWISE,
                                              OP_PATTERN_COMMONREDUCE,
                                              OP_PATTERN_BROAD_CAST,
                                              OP_PATTERN_SEGMENT,
                                              OP_PATTERN_MAXPOOL,
                                              OP_PATTERN_CONV,
                                              OP_PATTERN_MATMUL,
                                              OP_PATTERN_BNUPDATE,
                                              OP_PATTERN_BNREDUCE,
                                              OP_PATTERN_CONV_BACKPROP_INPUT,
                                              OP_PATTERN_DEPTHWISE_CONV,
                                              OP_PATTERN_QUANT,
                                              OP_PATTERN_DEQUANT,
                                              OP_PATTERN_REQUANT,
                                              OP_PATTERN_POOL2D,
                                              OP_PATTERN_ANTIQUANT,
                                              OP_PATTERN_STRIDED_WRITE,
                                              OP_PATTERN_STRIDED_READ,
                                              OP_PATTERN_AIPP,
                                              OP_PATTERN_CONFUSION_TRANSPOSE,
                                              OP_PATTERN_DEQUANTS16,
                                              OP_PATTERN_REQUANTS16,
                                              OP_PATTERN_READ_SELECT,
                                              OP_PATTERN_WRITE_SELECT,
                                              OP_PATTERN_BATCH_MATMUL,
                                              OP_PATTERN_CONV3D,
                                              OP_PATTERN_DROPOUTDOMASKV3D,
                                              OP_PATTERN_CONV3D_BACKPROP_INPUT,
                                              OP_PATTERN_CONV_BACKPROP_FILTER,
                                              OP_PATTERN_GEMM
};
}  // namespace fe

#endif  // INC_REGISTER_GRAPH_OPTIMIZER_BUFFER_FUSION_CONSTANT_H_
