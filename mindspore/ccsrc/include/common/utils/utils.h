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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_UTILS_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_UTILS_H_

#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <algorithm>

#include "utils/log_adapter.h"
#include "ir/dtype/type.h"

namespace mindspore {
// op name. Op which not exists in operator/ops.h, so define it's name here
constexpr auto kConcatOpName = "Concat";
constexpr auto kUniqueOpName = "Unique";
constexpr auto kMaskedSelectOpName = "MaskedSelect";
constexpr auto kComputeAccidentalHitsOpName = "ComputeAccidentalHits";
constexpr auto kCTCGreedyDecoderOpName = "CTCGreedyDecoder";
constexpr auto kDynamicStitchOpName = "DynamicStitch";
constexpr auto kFour2FiveOpName = "Four2Five";
constexpr auto kFive2FourOpName = "Five2Four";
constexpr auto kConv3DOpName = "Conv3D";
constexpr auto kConv3DBackpropFilterOpName = "Conv3DBackpropFilter";
constexpr auto kConv3DBackpropInputOpName = "Conv3DBackpropInput";
constexpr auto kConv2DOpName = "Conv2D";
constexpr auto kConvBN1OpName = "ConvBN1";
constexpr auto kBN2AddReluOpName = "BN2AddRelu";
constexpr auto kBN2ReLUOpName = "BN2Relu";
constexpr auto kBN2OpName = "BN2";
constexpr auto kFusedBN1OpName = "FusedBN1";
constexpr auto kFusedBN2OpName = "FusedBN2";
constexpr auto kFusedBN3OpName = "FusedBN3";
constexpr auto kBNGrad1OpName = "BNGrad1";
constexpr auto kBNGrad2OpName = "BNGrad2";
constexpr auto kBNGrad3OpName = "BNGrad3";
constexpr auto kBatchNorm = "BatchNorm";
constexpr auto kInstanceNorm = "InstanceNorm";
constexpr auto kBatchNormWithActivation = "BatchNormWithActivation";
constexpr auto kBatchNormWithAddAndActivation = "BatchNormWithAddAndActivation";
constexpr auto kBatchNormGradWithActivation = "BatchNormGradWithActivation";
constexpr auto kBatchNormGradWithAddAndActivation = "BatchNormGradWithAddAndActivation";
constexpr auto kClearZeroOpName = "ClearZero";
constexpr auto kAtomicAddrCleanOpName = "AtomicAddrClean";
constexpr auto kGetNextOpName = "GetNext";
constexpr auto kInitDatasetQueueOpName = "InitDataSetQueue";
constexpr auto kEndOfSequence = "EndOfSequence";
constexpr auto kAllToAllVOpName = "AllToAllv";
constexpr auto kAllReduceOpName = "AllReduce";
constexpr auto kAllGatherOpName = "AllGather";
constexpr auto kHostAllGatherOpName = "HostAllGather";
constexpr auto kBroadcastOpName = "Broadcast";
constexpr auto kReceiveOpName = "Receive";
constexpr auto kHcomSendOpName = "Send";
constexpr auto kReduceScatterOpName = "ReduceScatter";
constexpr auto kHostReduceScatterOpName = "HostReduceScatter";
constexpr auto kMemCpyAsyncOpName = "memcpy_async";
constexpr auto kTopKOpName = "TopK";
constexpr auto kLinSpaceOpName = "LinSpace";
constexpr auto kExtractImagePatchesOpName = "ExtractImagePatches";
constexpr auto kBNTrainingReduceOpName = "BNTrainingReduce";
constexpr auto kBNTrainingUpdateOpName = "BNTrainingUpdate";
constexpr auto kBNTrainingUpdateV2OpName = "BNTrainingUpdateV2";
constexpr auto kBNTrainingUpdateV3OpName = "BNTrainingUpdateV3";
constexpr auto kNonMaxSuppressionV3OpName = "NonMaxSuppressionV3";
constexpr auto kSimpleMeanGradOpName = "SimpleMeanGrad";
constexpr auto kMeanGradOpName = "MeanGrad";
constexpr auto kSliceOpName = "Slice";
constexpr auto kSliceGradOpName = "SliceGrad";
constexpr auto kCoalesceOpName = "Coalesce";
constexpr auto kTileOpName = "Tile";
constexpr auto kScatterNdOpName = "ScatterNd";
constexpr auto kStridedSliceAssignOpName = "StridedSliceAssign";
constexpr auto kStridedSliceOpName = "StridedSlice";
constexpr auto kStridedSliceGradOpName = "StridedSliceGrad";
constexpr auto kSparseGatherV2OpName = "SparseGatherV2";
constexpr auto kUnsortedSegmentProdOpName = "UnsortedSegmentProd";
constexpr auto kUnsortedSegmentMinOpName = "UnsortedSegmentMin";
constexpr auto kFlattenGradOpName = "FlattenGrad";
constexpr auto kExpandDimsOpName = "ExpandDims";
constexpr auto kReshapeOpName = "Reshape";
constexpr auto kTransposeOpName = "Transpose";
constexpr auto kTransposeNODOpName = "TransposeNOD";
constexpr auto kSplitOpName = "Split";
constexpr auto kSplitVOpName = "SplitV";
constexpr auto kSparseApplyAdagradOpName = "SparseApplyAdagrad";
constexpr auto kMomentumOpName = "Momentum";
constexpr auto kApplyMomentumOpName = "ApplyMomentum";
constexpr auto kCombineMomentumOpName = "CombineMomentum";
constexpr auto kCombineMomentumWeightOpName = "CombineMomentumWeight";
constexpr auto kApplyAdadeltaOpName = "ApplyAdadelta";
constexpr auto kApplyAdagradOpName = "ApplyAdagrad";
constexpr auto kApplyAdagradDAName = "ApplyAdagradDA";
constexpr auto kApplyAdamOpName = "Adam";
constexpr auto kApplyAdaMaxOpName = "ApplyAdaMax";
constexpr auto kApplyAddSignOpName = "ApplyAddSign";
constexpr auto kApplyCenteredRMSPOpName = "ApplyCenteredRMSP";
constexpr auto kApplyCenteredRMSPropOpName = "ApplyCenteredRMSProp";
constexpr auto kApplyFtrlOpName = "ApplyFtrl";
constexpr auto kApplyFtrlV2OpName = "ApplyFtrlV2";
constexpr auto kApplyGradientDescentOpName = "ApplyGradientDescent";
constexpr auto kApplyPowerSignOpName = "ApplyPowerSign";
constexpr auto kApplyProximalAdagradOpName = "ApplyProximalAdagrad ";
constexpr auto kApplyProximalGradientDescentOpName = "ApplyProximalGradientDescent";
constexpr auto kApplyRMSPropOpName = "ApplyRMSProp";
constexpr auto kTransDataOpName = "TransData";
constexpr auto kTransDataRNNOpName = "TransDataRNN";
constexpr auto kStackInitOpName = "StackInit";
constexpr auto kStackPushOpName = "StackPush";
constexpr auto kStackPopOpName = "StackPop";
constexpr auto kStackOpName = "Stack";
constexpr auto kStackDestroyOpName = "StackDestroy";
constexpr auto kBNTrainingUpdateGradOpName = "BNTrainingUpdateGrad";
constexpr auto kBNTrainingReduceGradOpName = "BNTrainingReduceGrad";
constexpr auto kSquareSumV1OpName = "SquareSumV1";
constexpr auto kSquareSumV2OpName = "SquareSumV2";
constexpr auto kClipByNormNoDivSumOpName = "ClipByNormNoDivSum";
constexpr auto kPReluOpName = "PReLU";
constexpr auto kGreaterOpName = "Greater";
constexpr auto kSqrtOpName = "Sqrt";
constexpr auto kRsqrtOpName = "Rsqrt";
constexpr auto kRsqrtGradOpName = "RsqrtGrad";
constexpr auto kErfOpName = "Erf";
constexpr auto kRealDivOpName = "RealDiv";
constexpr auto kLambUpdateWithLROpName = "LambUpdateWithLR";
constexpr auto kLambNextMVWithDecayOpName = "LambNextMVWithDecay";
constexpr auto kLambNextMVWithDecayV1OpName = "LambNextMVWithDecayV1";
constexpr auto kClipByValueOpName = "ClipByValue";
constexpr auto kLambNextRightOpName = "LambNextRight";
constexpr auto kConfusionSoftmaxGradOpName = "ConfusionSoftmaxGrad";
constexpr auto kLambUpdateWithLrV2OpName = "LambUpdateWithLrV2";
constexpr auto kLayerNormXBackpropOpName = "LayerNormXBackprop";
constexpr auto kLayerNormXBackpropV2OpName = "LayerNormXBackpropV2";
constexpr auto kLayerNormBetaGammaBackpropOpName = "LayerNormBetaGammaBackprop";
constexpr auto kLayerNormBetaGammaBackpropV2OpName = "LayerNormBetaGammaBackpropV2";
constexpr auto kLambNextMVOpName = "LambNextMV";
constexpr auto kConfusionTransposeDOpName = "ConfusionTransposeD";
constexpr auto kAdamApplyOneWithDecayOpName = "AdamApplyOneWithDecay";
constexpr auto kAdamApplyOneWithDecayAssignOpName = "AdamApplyOneWithDecayAssign";
constexpr auto kBatchNormGradOpName = "BatchNormGrad";
constexpr auto kBNInferOpName = "BNInfer";
constexpr auto kAdamApplyOneOpName = "AdamApplyOne";
constexpr auto kAdamApplyOneAssignOpName = "AdamApplyOneAssign";
constexpr auto kResizeNearestNeighborGradOpName = "ResizeNearestNeighborGrad";
constexpr auto kFusedMulAddOpName = "FusedMulAdd";
constexpr auto kFusedMulAddNOpName = "FusedMulAddN";
constexpr auto kFusedMulApplyMomentumOpName = "FusedMulApplyMomentum";
constexpr auto kBiasAddOpName = "BiasAdd";
constexpr auto kConfusionMulGradOpName = "ConfusionMulGrad";
constexpr auto kStreamSwitchOpName = "StreamSwitch";
constexpr auto kStreamActiveOpName = "StreamActive";
constexpr auto kAssignAddOpName = "AssignAdd";
constexpr auto kSendOpName = "StreamSend";
constexpr auto kRecvOpName = "StreamRecv";
constexpr auto kRpcSendOpName = "RpcSend";
constexpr auto kRpcRecvOpName = "RpcRecv";
constexpr auto kReluV2OpName = "ReLUV2";
constexpr auto kReluGradV2OpName = "ReluGradV2";
constexpr auto kAddOpName = "Add";
constexpr auto kAddNOpName = "AddN";
constexpr auto kResizeNearestNeighborV2OpName = "ResizeNearestNeighborV2";
constexpr auto kResizeNearestNeighborV2GradOpName = "ResizeNearestNeighborV2Grad";
constexpr auto kApplyRMSPropOpname = "ApplyRMSProp";
constexpr auto kCumsumOpName = "Cumsum";
constexpr auto kInplaceAddOpName = "InplaceAdd";
constexpr auto kInplaceSubOpName = "InplaceSub";
constexpr auto kResizeBilinearV2OpName = "kResizeBilinearV2";
constexpr auto kReduceProdOpName = "ReduceProd";
constexpr auto kCumprodOpName = "Cumprod";
constexpr auto kSpaceToBatchOpName = "SpaceToBatch";
constexpr auto kBatchToSpaceOpName = "BatchToSpace";
constexpr auto kSpaceToDepthOpName = "SpaceToDepth";
constexpr auto kPadOpName = "Pad";
constexpr auto kConv2DTransposeOpName = "Conv2DTranspose";
constexpr auto kConv2DBackpropInputOpName = "Conv2DBackpropInput";
constexpr auto kConv2DBackpropFilterOpName = "Conv2DBackpropFilter";
constexpr auto kDepthwiseConv2dNativeOpName = "DepthwiseConv2dNative";
constexpr auto kDepthwiseConv2dNativeBackpropInputOpName = "DepthwiseConv2dNativeBackpropInput";
constexpr auto kDepthwiseConv2dNativeBackpropFilterOpName = "DepthwiseConv2dNativeBackpropFilter";
constexpr auto kFusionOpConv2DBackpropInputReluGradV2Name = "FusionOp_Conv2DBackpropInput_ReluGradV2";
constexpr auto kFusionOpConv2DBackpropInputAddNReluGradV2Name = "FusionOp_Conv2DBackpropInput_AddN_ReluGradV2";
constexpr auto kLabelSetOpName = "LabelSet";
constexpr auto kLabelSwitchOpName = "LabelSwitch";
constexpr auto kLabelGotoOpName = "LabelGoto";
constexpr auto kBNInferGradOpName = "BNInferGrad";
constexpr auto kCallOpName = "call";
constexpr auto kPartialOpName = "partial";
constexpr auto kSwitchOpName = "Switch";
constexpr auto kReturnOpName = "Return";
constexpr auto kBpropCutOpName = "bprop_cut";
constexpr auto kLarsV2OpName = "LarsV2";
constexpr auto kLarsV2UpdateOpName = "LarsV2Update";
constexpr auto kSquareSumAllOpName = "SquareSumAll";
constexpr auto kNMSWithMaskOpName = "NMSWithMask";
constexpr auto kSoftmaxGradExtOpName = "SoftmaxGradExt";
constexpr auto kStridedReadOpName = "StridedRead";
constexpr auto kStridedWriteOpName = "StridedWrite";
constexpr auto kFusedAdamWeightDecayName = "FusedAdamWeightDecay";
constexpr auto kAdamWeightDecayName = "AdamWeightDecay";
constexpr auto kFusedCastAdamWeightDecayName = "FusedCastAdamWeightDecay";
constexpr auto kFusedAdamName = "FusedAdam";
constexpr auto kFusedAdaFactorName = "FusedAdaFactor";
constexpr auto kFusedAdaFactorWithGlobalNormName = "FusedAdaFactorWithGlobalNorm";
constexpr auto kFusedSparseAdamName = "FusedSparseAdam";
constexpr auto kFusedMatMulBiasAddName = "FusedMatMulBiasAdd";
constexpr auto kDeadNodeName = "DeadNode";
constexpr auto kPolyNodeName = "PolyNode";
constexpr auto kApplyAdagradV2OpName = "ApplyAdagradV2";
constexpr auto kSparseApplyAdagradV2OpName = "SparseApplyAdagradV2";
constexpr auto kSparseApplyFtrlOpName = "SparseApplyFtrl";
constexpr auto kSparseApplyFtrlV2OpName = "SparseApplyFtrlV2";
constexpr auto kApplyKerasMomentumOpName = "ApplyKerasMomentum";
constexpr auto kSparseApplyProximalAdagradOpName = "SparseApplyProximalAdagrad";
constexpr auto kSparseApplyRMSPropOpName = "SparseApplyRMSProp";
constexpr auto kSparseApplyAdadeltaOpName = "SparseApplyAdadelta";
constexpr auto kApplyAdamWithAmsgradOpName = "ApplyAdamWithAmsgrad";
constexpr auto kTensorMoveOpName = "TensorMove";
constexpr auto kTensorCopySlicesOpName = "TensorCopySlices";
constexpr auto kTensorScatterUpdateOpName = "TensorScatterUpdate";
constexpr auto kScatterNdUpdateOpName = "ScatterNdUpdate";
constexpr auto kPushOpName = "Push";
constexpr auto kPullOpName = "Pull";
constexpr auto kPushWeightOpName = "PushWeight";
constexpr auto kPullWeightOpName = "PullWeight";
constexpr auto kFusedPushWeightOpName = "FusedPushWeight";
constexpr auto kFusedPullWeightOpName = "FusedPullWeight";
constexpr auto kUpdateCacheOpName = "UpdateCache";
constexpr auto kCacheSwapTableOpName = "CacheSwapTable";
constexpr auto kEmbeddingLookupOpName = "EmbeddingLookup";
constexpr auto kEmbeddingLookupProxyOpName = "EmbeddingLookupProxy";
constexpr auto kGatherV2OpName = "Gather";
constexpr auto kPaddingOpName = "Padding";
constexpr auto kAvgPoolOpName = "AvgPool";
constexpr auto kAvgPoolGradOpName = "AvgPoolGrad";
constexpr auto kAvgPool3DOpName = "AvgPool3D";
constexpr auto kAvgPool3DGradOpName = "AvgPool3DGrad";
constexpr auto kAvgPoolGradVmOpName = "AvgPoolGradVm";
constexpr auto kMaxPoolOpName = "MaxPool";
constexpr auto kMaxPoolGradOpName = "MaxPoolGrad";
constexpr auto kMaxPool3DOpName = "MaxPool3D";
constexpr auto kMaxPool3DGradOpName = "MaxPool3DGrad";
constexpr auto kMaxPoolWithArgmaxOpName = "MaxPoolWithArgmax";
constexpr auto kMaxPoolGradWithArgmaxOpName = "MaxPoolGradWithArgmax";
constexpr auto kTensorAddOpName = "Add";
constexpr auto kMaxPool3DGradGradOpName = "MaxPool3DGradGrad";
constexpr auto kCastOpName = "Cast";
constexpr auto kGreaterEqualOpName = "GreaterEqual";
constexpr auto kAbsOpName = "Abs";
constexpr auto kExpOpName = "Exp";
constexpr auto kNegOpName = "Neg";
constexpr auto kMinimumOpName = "Minimum";
constexpr auto kMaximumOpName = "Maximum";
constexpr auto kMulOpName = "Mul";
constexpr auto kSubOpName = "Sub";
constexpr auto kLogOpName = "Log";
constexpr auto kPowOpName = "Pow";
constexpr auto kReciprocalOpName = "Reciprocal";
constexpr auto kEqualOpName = "Equal";
constexpr auto kLessOpName = "Less";
constexpr auto kLessEqualOpName = "LessEqual";
constexpr auto kSquareOpName = "Square";
constexpr auto kSelectOpName = "Select";
constexpr auto kReduceSumOpName = "ReduceSum";
constexpr auto kReduceMinOpName = "ReduceMin";
constexpr auto kReduceMaxOpName = "ReduceMax";
constexpr auto kReduceMeanOpName = "ReduceMean";
constexpr auto kReduceAnyOpName = "ReduceAny";
constexpr auto kReduceAllOpName = "ReduceAll";
constexpr auto kFusedWeightScaleApplyMomentum = "FusedWeightScaleApplyMomentum";
constexpr auto kFusedWeightApplyMomentum = "FusedWeightApplyMomentum";
constexpr auto kFusedScaleApplyMomentum = "FusedScaleApplyMomentum";
constexpr auto kBasicLSTMCellWeightGradOpName = "BasicLSTMCellWeightGrad";
constexpr auto kBasicLSTMCellInputGradOpName = "BasicLSTMCellInputGrad";
constexpr auto kBasicLSTMCellOpName = "BasicLSTMCell";
constexpr auto kDynamicRNNOpName = "DynamicRNN";
constexpr auto kLSTMOpName = "LSTM";
constexpr auto kLSTMGradOpName = "LSTMGrad";
constexpr auto kLSTMInputGradOpName = "LSTMInputGrad";
constexpr auto kDynamicGRUV2OpName = "DynamicGRUV2";
constexpr auto kGRUV2HiddenGradOpName = "GRUV2HiddenGrad";
constexpr auto kGRUV2HiddenGradCellOpName = "GRUV2HiddenGradCell";
constexpr auto kFusedSparseFtrlName = "FusedSparseFtrl";
constexpr auto kFusedSparseProximalAdagradName = "FusedSparseProximalAdagrad";
constexpr auto kFusedSparseLazyAdamName = "FusedSparseLazyAdam";
constexpr auto kSparseApplyFtrlName = "SparseApplyFtrl";
constexpr auto kSparseApplyFtrlV2Name = "SparseApplyFtrlV2";
constexpr auto kSGDName = "SGD";
constexpr auto kLARSUpdateName = "LARSUpdate";
constexpr auto kBasicLSTMCellCStateGradOpName = "BasicLSTMCellCStateGrad";
constexpr auto kBasicLSTMCellCStateGradV2OpName = "BasicLSTMCellCStateGradV2";
constexpr auto kMatMulOpName = "MatMul";
constexpr auto kMatMulV2OpName = "MatMulV2";
constexpr auto kBatchMatMulOpName = "BatchMatMul";
constexpr auto kBatchMatMulV2OpName = "BatchMatMulV2";
constexpr auto kBroadcastToOpName = "BroadcastTo";
constexpr auto kFusedAddReluV2Name = "FusedAddReluV2";
constexpr auto kFusedAddReluGradV2Name = "FusedAddReluGradV2";
constexpr auto kDropoutOpName = "Dropout";
constexpr auto kDropoutGradOpName = "DropoutGrad";
constexpr auto kDropoutGenMaskOpName = "DropoutGenMask";
constexpr auto kDropoutGenMaskV3OpName = "DropoutGenMaskV3";
constexpr auto kDropoutDoMaskOpName = "DropoutDoMask";
constexpr auto kDropoutDoMaskV3OpName = "DropoutDoMaskV3";
constexpr auto kSoftmaxV2WithDropoutDoMaskV3OpName = "SoftmaxV2WithDropoutDoMaskV3";
constexpr auto kGammaOpName = "Gamma";
constexpr auto kPoissonOpName = "Poisson";
constexpr auto kStandardLaplaceOpName = "StandardLaplace";
constexpr auto kStandardNormalOpName = "StandardNormal";
constexpr auto kUniformIntOpName = "UniformInt";
constexpr auto kUniformRealOpName = "UniformReal";
constexpr auto kSubAndFilterOpName = "SubAndFilter";
constexpr auto kPadAndShiftOpName = "PadAndShift";
constexpr auto kSparseSoftmaxCrossEntropyWithLogitsOpName = "SparseSoftmaxCrossEntropyWithLogits";
constexpr auto kOneHotOpName = "OneHot";
constexpr auto kSoftmaxCrossEntropyWithLogitsOpName = "SoftmaxCrossEntropyWithLogits";
constexpr auto kUniformCandidateSamplerOpName = "UniformCandidateSampler";
constexpr auto kLogSoftmaxGradOpName = "LogSoftmaxGrad";
constexpr auto kLayerNormGradOpName = "LayerNormGrad";
constexpr auto kMinimumGradOpName = "MinimumGrad";
constexpr auto kMaximumGradOpName = "MaximumGrad";
constexpr auto kFusedDbnDwOpName = "FusedDbnDw";
constexpr auto kNPUAllocFloatStatusOpName = "NPUAllocFloatStatus";
constexpr auto kNPUGetFloatStatusOpName = "NPUGetFloatStatus";
constexpr auto kNPUClearFloatStatusOpName = "NPUClearFloatStatus";
constexpr auto kAssignOpName = "Assign";
constexpr auto kScatterAddOpName = "ScatterAdd";
constexpr auto kScatterUpdateOpName = "ScatterUpdate";
constexpr auto kEnvironCreateOpName = "EnvironCreate";
constexpr auto kEnvironSetOpName = "EnvironSet";
constexpr auto kEnvironGetOpName = "EnvironGet";
constexpr auto kEnvironDestroyAllOpName = "EnvironDestroyAll";
constexpr auto kUpdateStateOpName = "UpdateState";
constexpr auto kPriorityReplayBufferCreate = "PriorityReplayBufferCreate";
constexpr auto kPriorityReplayBufferPush = "PriorityReplayBufferPush";
constexpr auto kPriorityReplayBufferSample = "PriorityReplayBufferSample";
constexpr auto kPriorityReplayBufferUpdate = "PriorityReplayBufferUpdate";

// Communication world group
constexpr auto kNcclWorldGroup = "nccl_world_group";
constexpr auto kHcclWorldGroup = "hccl_world_group";
constexpr auto kSyncBnGroup = "sync_bn_group";
constexpr auto kRankID = "RANK_ID";

// Hcom Op Type
constexpr auto kHcomOpTypeAllReduce = "HcomAllReduce";
constexpr auto kHcomOpTypeAllGather = "HcomAllGather";
constexpr auto kHcomOpTypeBroadcast = "HcomBroadcast";
constexpr auto kHcomOpTypeSend = "HcomSend";
constexpr auto kHcomOpTypeReceive = "HcomReceive";
constexpr auto kHcomOpTypeReduceScatter = "HcomReduceScatter";

// attr key name
constexpr auto kAttrInputNames = "input_names";
constexpr auto kAttrAttrNames = "attr_names";
constexpr auto kAttrIsAiCpuKernel = "is_AICPU_kernel";
constexpr auto kIsBackendCast = "is_backed_cast";
constexpr auto kAttrOutputNames = "output_names";
constexpr auto kAttrAsync = "async";
constexpr auto kAttrOffload = "offload";
constexpr auto kAttrVisited = "visited";
constexpr auto kAttrShape = "shape";
constexpr auto kAttrMomentum = "momentum";
constexpr auto kAttrEps = "eps";
constexpr auto kAttrEpsilon = "epsilon";
constexpr auto kAttrFactor = "factor";
constexpr auto kAttrIsRef = "isRef";
constexpr auto kAttrDataShape = "data_shape";
constexpr auto kAttrFormat = "format";
constexpr auto kAttrReshapeType = "reshape_type";
constexpr auto kAttrAxis = "axis";
constexpr auto kAttrAxes = "axes";
constexpr auto kAttrKeepDims = "keep_dims";
constexpr auto kAttrShapeGamma = "shape_gamma";
constexpr auto kAttrPerm = "perm";
constexpr auto kAttrTransposeFirst = "transpose_first";
constexpr auto kAttrAtomicAddMemSize = "automic_add_mem_size";
constexpr auto kAttrAtomicOutputIndexs = "atomic_output_clean_indexs";
constexpr auto kAttrNeedAtomic = "need_atomic";
constexpr auto kAttrAtomicWorkspaceIndexs = "atomic_workspace_clean_indexs";
constexpr auto kAttrSwitchCondition = "switch_condition";
constexpr auto kAttrDataType = "data_type";
constexpr auto kAttrDType = "dtype";
constexpr auto kAttrActiveTarget = "active_target";
constexpr auto kAttrActiveStreamId = "active_stream_id";
constexpr auto kAttrActiveStreamList = "active_stream_list";
constexpr auto kAttrTrueBranchStream = "true_branch_stream";
constexpr auto kAttrStreamSwitchKind = "stream_switch_kind";
constexpr auto kAttrEventId = "event_id";
constexpr auto kAttrLabelId = "label_id";
constexpr auto kAttrLogicId = "logic_id";
constexpr auto kAttrNodeInfo = "node_info";
constexpr auto kAttrNodeName = "node_name";
constexpr auto kAttrDynInput = "dynamic";
constexpr auto kAttrDynInputSizes = "dyn_input_sizes";
constexpr auto kAttrSrcFormat = "src_format";
constexpr auto kAttrDstFormat = "dst_format";
constexpr auto kAttrMultiples = "multiples";
constexpr auto kAttrFixPrecision = "fix_precision";
constexpr auto kAttrOutputPrecision = "output_precision";
constexpr auto kAttrOutputUsedNum = "output_used_num";
constexpr auto kAttrHasBias = "has_bias";
constexpr auto kAttrN = "n";
constexpr auto kAttrLabelForInsertStreamActive = "label_for_insert_stream_active";
constexpr auto kAttrFpBpEnd = "fpbp_end";
constexpr auto kAttrFusion = "fusion";
constexpr auto kAttrNotDelayFusion = "not_delay_fusion";
constexpr auto kAttrGroup = "group";
constexpr auto kAttrGroups = "groups";
constexpr auto kAttrGroupBack = "group_back";
constexpr auto kAttrFracZGroup = "fracz_group";
constexpr auto kAttrFracZGroupIdx = "fracz_group_idx";
constexpr auto kAttrOp = "op";
constexpr auto kAttrDestRank = "dest_rank";
constexpr auto kAttrSrcRank = "src_rank";
constexpr auto kAttrSrTag = "sr_tag";
constexpr auto kAttrRootRank = "root_rank";
constexpr auto kAttrIsTraining = "is_training";
constexpr auto kAttrFusionId = "fusion_id";
constexpr auto kAttrDuplicated = "duplicated";
constexpr auto kAttrBucketId = "bucket_id";
constexpr auto kAttrGradOutputIndex = "grad_output_index";
constexpr auto kAttrLabelIndex = "label_index";
constexpr auto kAttrLabelSwitchList = "label_switch_list";
constexpr auto kAttrBeginMask = "begin_mask";
constexpr auto kAttrEndMask = "end_mask";
constexpr auto kAttrEllipsisMask = "ellipsis_mask";
constexpr auto kAttrNewAxisMask = "new_axis_mask";
constexpr auto kAttrShrinkAxisMask = "shrink_axis_mask";
constexpr auto kAttrDatadumpOriginalNames = "_datadump_original_names";
constexpr auto kAttrDatadumpIsMultiop = "_datadump_is_multiop";
constexpr auto kAttrNeedRecordEvent = "need_record_event";
constexpr auto kAttrStreamId = "stream_id";
constexpr auto kAttrRecordEvent = "record_event";
constexpr auto kAttrWaitEvent = "wait_event";
constexpr auto kAttrRecordEventStream = "record_event_stream";
constexpr auto kAttrWaitEventStream = "wait_event_stream";
constexpr auto kAttrIndex = "index";
constexpr auto kAttrSplitDim = "split_dim";
constexpr auto kAttrNumSplit = "num_split";
constexpr auto kAttrReduction = "reduction";
constexpr auto kAttrOutputNum = "output_num";
constexpr auto kAttrSizeSplits = "size_splits";
constexpr auto kAttrOutputDefault = "output_default";
constexpr auto kAttrPrimitiveTarget = "primitive_target";
constexpr auto kAttrUseLocking = "use_locking";
constexpr auto kAttrReduceScatterFlag = "reduce_scatter_flag";
constexpr auto kAttrOffset = "offset";
constexpr auto kAttrCacheEnable = "cache_enable";
constexpr auto kAttrPsKey = "ps_key";
constexpr auto kAttrOptimizerType = "optim_type";
constexpr auto kAttrChildGraph = "child_graph";
constexpr auto kAttrInputNums = "inputNums";
constexpr auto kAttrT = "T";
constexpr auto kAttrNum = "num";
constexpr auto kAttrRecvType = "recv_type";
constexpr auto kAttrConcatDim = "concat_dim";
constexpr auto kAttrSplitCount = "split_count";
constexpr auto kAttrSendRankIds = "send_rank_ids";
constexpr auto kAttrRecvRankIds = "recv_rank_ids";
constexpr auto kAttrSendLens = "send_lens";
constexpr auto kAttrRecvLens = "recv_lens";
constexpr auto kAttrRankSize = "rank_size";
constexpr auto kAttrPadDimSize = "pad_dim_size";
constexpr auto kAttrPaddings = "paddings";
constexpr auto kAttrNumSegments = "num_segments";
constexpr auto kAttrStackOpName = "stack_op_name";
constexpr auto kAttrBegin = "begin";
constexpr auto kAttrEnd = "end";
constexpr auto kAttrSize = "size";
constexpr auto kAttrIsDynamicShape = "is_dynamic_shape";
constexpr auto kAttrInputIsDynamicShape = "input_is_dynamic_shape";
constexpr auto kAttrOutputIsDynamicShape = "output_is_dynamic_shape";
constexpr auto kAttrPynativeNextOpName = "next_op";
constexpr auto kAttrPynativeNextIndex = "next_index";
constexpr auto kAttrCompileInfo = "compile_info";
constexpr auto kAttrFusionType = "fusion_type";
constexpr auto kAttrStride = "stride";
constexpr auto kAttrStrides = "strides";
constexpr auto kAttrKernelSize = "kernel_size";
constexpr auto kAttrDilation = "dilation";
constexpr auto kAttrPadMode = "pad_mode";
constexpr auto kAttrPad = "pad";
constexpr auto kAttrPadding = "padding";
constexpr auto kAttrNonTask = "non_task";
constexpr auto kAttrIsGrad = "is_grad";
constexpr auto kAttrRecompute = "recompute";
constexpr auto kAttrSliceActivation = "slice_activation";
constexpr auto kAttrNeedCseAfterRecompute = "need_cse_after_recompute";
constexpr auto kAttrParallelDimInfo = "parallel_dim_info";
constexpr auto kAttrParallelFusionType = "parallel_fusion_type";
constexpr auto kAttrParallelTypeInfo = "parallel_type_info";
constexpr auto kAttrCompositeType = "composite_type";
constexpr auto kAttrStitch = "stitch";
constexpr auto kAttrTopoSortRhsFirst = "topo_sort_rhs_first";
constexpr auto kAttrIgnoreSideEffect = "ignore_side_effect";
constexpr auto kAttrSwitchLayer = "switch_layer";
constexpr auto kAttrReturn = "return";
constexpr auto kAttrRecursiveStart = "recursive_start";
constexpr auto kAttrRecursiveEnd = "recursive_end";
constexpr auto kAttrRecursive = "recursive";
constexpr auto kAttrMultiCallEnd = "multicall_end";
constexpr auto kAttrProfilingIterEnd = "PROFILING_ITER_END";
constexpr auto kAttrHiddenSize = "hidden_size";
constexpr auto kAttrInputSize = "input_size";
constexpr auto kAttrDstType = "dst_type";
constexpr auto kAttrDump = "dump";
constexpr auto kAttrSkipNopOpAddr = "skip_nop_op_addr";
constexpr auto kAttrSkipNopOpExecution = "skip_nop_op_execution";
constexpr auto kAttrFixedInputFormat = "fixed_input_format";
constexpr auto kAttrFixedOutputFormat = "fixed_output_format";
constexpr auto kAttrFixedInputDeviceShape = "fixed_input_device_shape";
constexpr auto kAttrFixedOutputDeviceShape = "fixed_output_device_shape";
constexpr auto kAttrFuncType = "func_type";
constexpr auto kAttrCustAicpu = "cust_aicpu";
constexpr auto kAttrIsInternalOutputNopNode = "is_internal_output_nop_node";
constexpr auto kAttrIsUBFusionOp = "is_ub_fusion_op";
constexpr auto kAttrNopOp = "nop_op";
constexpr auto kAttrPlaceHolderIndex = "placeholder_index";
constexpr auto kAttrMicro = "micro";
constexpr auto kAttrJsonFileName = "json_file_name";
constexpr auto kAttrNeedDropInput = "need_drop_input";
constexpr auto kAttrNeedConvertToValueNode = "need_convert_to_value_node";
constexpr auto kAttrSendSrcNodeName = "send_src_node_name";
constexpr auto kAttrSendDstNodeName = "send_dst_node_name";
constexpr auto kAttrSendDstRanks = "send_dst_ranks";
constexpr auto kAttrSendDstRoles = "send_dst_roles";
constexpr auto kAttrRecvSrcNodeName = "recv_src_node_name";
constexpr auto kAttrRecvDstNodeName = "recv_dst_node_name";
constexpr auto kAttrRecvSrcRanks = "recv_src_ranks";
constexpr auto kAttrRecvSrcRoles = "recv_src_roles";
constexpr auto kAttrInterProcessEdgeName = "inter_process_edge_name";
constexpr auto kAttrForwardOpOutputId = "forward_op_output_id";
constexpr auto kAttrGroupRankIds = "group_rank_ids";
constexpr auto kAttrReuseCommunication = "reuse_communication_node";
constexpr auto kAttrPrecisionFlag = "precision_flag";

// TODO(dsj): for ms_function running in graph_mode. should be delete later
constexpr auto kAttrMSFunction = "ms_function_graph";
// for single_op_compile_and_run condition and set_context GraphMode. should be delete later
constexpr auto kAttrSingleOpCompile = "compile_and_run_in_single_op";

// custom operator func type
constexpr auto kCustomTypeAOT = "aot";
constexpr auto kCustomTypeJULIA = "julia";
constexpr auto kCustomTypePyfunc = "pyfunc";
constexpr auto kCustomTypeTbe = "tbe";
constexpr auto kCustomTypeAICPU = "aicpu";
constexpr auto kCustomTypeHybrid = "hybrid";
const std::set<std::string> kCustomTypeAkg = {"ir_builder", "tvm_compute", "hybrid"};

// primal attr key name
constexpr auto kPrimalAttrForwardNodeName = "forward_node_name";

// attr value
constexpr auto kValueTargetSwitch = "target_switch";
constexpr auto kValueTargetOther = "target_other";
constexpr auto kValueTrue = "true";

// env key
constexpr auto kGraphOpRun = "GRAPH_OP_RUN";

// some size
const size_t kShape4dDims = 4;
const size_t kShape3dDims = 3;
const size_t kShape2dDims = 2;
const size_t kShape5dDims = 5;
const size_t kShape1dDims = 1;
const size_t kCubeSize = 16;
const size_t kCubeSize_C04 = 4;
const size_t kNiSize = 16;
const size_t kMemAlignSize = 512;
const size_t kBNChannelMultipleFactor = 4;
const int kParameterDataTensorMask = 0;
const int kParameterWeightTensorMask = 1;
const int kValueNodeTensorMask = 2;
constexpr auto kNCHWShapeSize = 4;

// define special index in special node
constexpr auto kDefaultStreamIndex = 0;
constexpr auto kIndependentStreamIndex = 1;
constexpr auto kWorldGroupStreamIndex = 2;
constexpr auto kAnfPrimitiveIndex = 0;
constexpr auto kFirstDataInputIndex = 1;
constexpr auto kRealInputNodeIndexInTupleGetItem = 1;
constexpr auto kInputNodeOutputIndexInTupleGetItem = 2;
constexpr auto kSparseGetAttrInputSize = 2;
constexpr auto kTupleGetItemInputSize = 3;
// index define of partial
constexpr auto kPartialMinInputSize = 2;
constexpr auto kPartialGraphIndex = 1;

// index define of switch
constexpr auto kSwitchInputSize = 4;
constexpr auto kSwitchTrueBranchIndex = 2;
constexpr auto kSwitchFalseBranchIndex = 3;
constexpr auto kSwitchBranchesNum = 2;

// index define of switch_layer
constexpr auto kSwitchLayerInputSize = 3;
constexpr auto kSwitchLayerSelectIndex = 1;
constexpr auto kSwitchLayerBranchesIndex = 2;

// index define of depend
constexpr auto kRealInputIndexInDepend = 1;
constexpr auto kDependAttachNodeIndex = 2;
constexpr auto kDependInputSize = 3;
// index define of UpdateState
constexpr auto kUpdateStateStateInput = 1;
constexpr auto kUpdateStateRealInput = 2;
// index define of Load
constexpr auto kLoadRealInput = 1;
constexpr auto kLoadStateInput = 2;
// time transfer unit
constexpr int kBasicTimeTransferUnit = 1000;
constexpr int kMaxVectorSize = 10000;
// index of input or output
constexpr size_t kIndex0 = 0;
constexpr size_t kIndex1 = 1;
constexpr size_t kIndex2 = 2;
constexpr size_t kIndex3 = 3;
constexpr size_t kIndex4 = 4;
constexpr size_t kIndex5 = 5;
constexpr size_t kIndex6 = 6;
constexpr size_t kIndex7 = 7;
constexpr size_t kIndex8 = 8;
constexpr size_t kIndex9 = 9;
constexpr size_t kIndex10 = 10;
constexpr size_t kIndex11 = 11;
constexpr size_t kIndex12 = 12;
constexpr size_t kIndex13 = 13;
constexpr size_t kIndex14 = 14;
constexpr size_t kIndex15 = 15;
constexpr size_t kIndex16 = 16;
// dim of shape
constexpr size_t kDim0 = 0;
constexpr size_t kDim1 = 1;
constexpr size_t kDim2 = 2;
constexpr size_t kDim3 = 3;
constexpr size_t kDim4 = 4;
constexpr size_t kDim5 = 5;
constexpr size_t kDim6 = 6;
// format
constexpr auto kOpFormat_DEFAULT = "DefaultFormat";
constexpr auto kOpFormat_ChannelFirst = "ChannelFirst";
constexpr auto kOpFormat_ChannelLast = "ChannelLast";
constexpr auto kOpFormat_NC1KHKWHWC0 = "NC1KHKWHWC0";
constexpr auto kOpFormat_ND = "ND";
constexpr auto kOpFormat_NCHW = "NCHW";
constexpr auto kOpFormat_NHWC = "NHWC";
constexpr auto kOpFormat_HWCN = "HWCN";
constexpr auto kOpFormat_NC1HWC0 = "NC1HWC0";
constexpr auto kOpFormat_FRAC_Z = "FracZ";
constexpr auto kOpFormat_FRACTAL_Z = "FRACTAL_Z";
constexpr auto kOpFormat_FRAC_NZ = "FRACTAL_NZ";
constexpr auto kOpFormat_C1HWNCoC0 = "C1HWNCoC0";
constexpr auto kOpFormat_NC1HWC0_C04 = "NC1HWC0_C04";
constexpr auto kOpFormat_FRACTAL_Z_C04 = "FRACTAL_Z_C04";
constexpr auto kOpFormat_NDHWC = "NDHWC";
constexpr auto kOpFormat_NCDHW = "NCDHW";
constexpr auto kOpFormat_DHWNC = "DHWNC";
constexpr auto kOpFormat_DHWCN = "DHWCN";
constexpr auto kOpFormat_NDC1HWC0 = "NDC1HWC0";
constexpr auto kOpFormat_FRACTAL_Z_3D = "FRACTAL_Z_3D";
constexpr auto kOpFormat_FRACTAL_ZN_LSTM = "FRACTAL_ZN_LSTM";
constexpr auto kOpFormat_FRACTAL_ZN_RNN = "FRACTAL_ZN_RNN";
constexpr auto kOpFormat_ND_RNN_BIAS = "ND_RNN_BIAS";

const std::set<std::string> kOpFormatList = {kOpFormat_DEFAULT,
                                             kOpFormat_NC1KHKWHWC0,
                                             kOpFormat_ND,
                                             kOpFormat_NCHW,
                                             kOpFormat_NHWC,
                                             kOpFormat_HWCN,
                                             kOpFormat_NC1HWC0,
                                             kOpFormat_FRAC_Z,
                                             kOpFormat_C1HWNCoC0,
                                             kOpFormat_FRAC_NZ,
                                             kOpFormat_NC1HWC0_C04,
                                             kOpFormat_FRACTAL_Z_C04,
                                             kOpFormat_NDHWC,
                                             kOpFormat_FRACTAL_ZN_LSTM,
                                             kOpFormat_FRACTAL_ZN_RNN,
                                             kOpFormat_ND_RNN_BIAS,
                                             kOpFormat_NDC1HWC0,
                                             kOpFormat_NCDHW,
                                             kOpFormat_FRACTAL_Z_3D,
                                             kOpFormat_DHWNC,
                                             kOpFormat_DHWCN};

constexpr auto kSliceStart = "start";
constexpr auto kSliceStop = "stop";
constexpr auto kSliceStep = "step";
const std::map<std::string, size_t> kSliceAttrToIndex = {{kSliceStart, 1}, {kSliceStop, 2}, {kSliceStep, 3}};

const std::set<std::string> kDefaultCompatibleFormat = {kOpFormat_ND, kOpFormat_NCHW, kOpFormat_NHWC, kOpFormat_HWCN,
                                                        kOpFormat_NCDHW};

const std::set<std::string> kOptOperatorSet = {kMomentumOpName,
                                               kApplyMomentumOpName,
                                               kApplyAdadeltaOpName,
                                               kApplyAdagradOpName,
                                               kApplyAdagradDAName,
                                               kApplyAdamOpName,
                                               kApplyAdaMaxOpName,
                                               kApplyAddSignOpName,
                                               kApplyCenteredRMSPOpName,
                                               kApplyFtrlOpName,
                                               kApplyFtrlV2OpName,
                                               kApplyGradientDescentOpName,
                                               kApplyPowerSignOpName,
                                               kApplyProximalAdagradOpName,
                                               kApplyProximalGradientDescentOpName,
                                               kApplyRMSPropOpName,
                                               kAdamApplyOneWithDecayOpName,
                                               kAdamApplyOneWithDecayAssignOpName,
                                               kFusedAdamWeightDecayName,
                                               kAdamWeightDecayName,
                                               kFusedCastAdamWeightDecayName,
                                               kFusedAdamName,
                                               kFusedAdaFactorName,
                                               kFusedAdaFactorWithGlobalNormName,
                                               kFusedSparseAdamName,
                                               kFusedMulApplyMomentumOpName,
                                               kFusedWeightScaleApplyMomentum,
                                               kFusedScaleApplyMomentum,
                                               kApplyCenteredRMSPropOpName,
                                               kFusedSparseFtrlName,
                                               kFusedSparseProximalAdagradName,
                                               kFusedSparseLazyAdamName,
                                               kSparseApplyFtrlName,
                                               kSparseApplyFtrlV2Name,
                                               kSGDName,
                                               kLARSUpdateName,
                                               kCombineMomentumWeightOpName,
                                               kCombineMomentumOpName,
                                               kScatterAddOpName,
                                               kScatterUpdateOpName,
                                               kSparseApplyProximalAdagradOpName};

const std::set<std::string> kNodeWithSeedOperators = {kGammaOpName,          kPoissonOpName,    kStandardLaplaceOpName,
                                                      kStandardNormalOpName, kUniformIntOpName, kUniformRealOpName};
const std::set<std::string> kPosteriorOperatorSet = {kPullOpName};

const std::set<std::string> kOpCacheBlackList = {kUniformCandidateSamplerOpName, kInitDatasetQueueOpName,
                                                 kGetNextOpName};

const std::set<std::string> kOpNotSupportMultiThreadExecList = {kAvgPoolOpName, kAvgPoolGradOpName, kMaxPoolOpName,
                                                                kBatchNorm, kBatchNormGradOpName};

const std::set<std::string> kHWSpecialFormatSet = {
  kOpFormat_FRACTAL_Z_3D,   kOpFormat_NC1KHKWHWC0, kOpFormat_NC1HWC0,       kOpFormat_FRAC_NZ,
  kOpFormat_C1HWNCoC0,      kOpFormat_NC1HWC0_C04, kOpFormat_FRACTAL_Z_C04, kOpFormat_FRACTAL_ZN_LSTM,
  kOpFormat_FRACTAL_ZN_RNN, kOpFormat_NDC1HWC0,    kOpFormat_FRAC_Z};

const std::set<TypeId> kFloatDataTypeSet = {kNumberTypeFloat16, kNumberTypeFloat32};

const std::set<std::string> kComputeDepend = {
  kUniqueOpName,           kComputeAccidentalHitsOpName, kSubAndFilterOpName, kPadAndShiftOpName,
  kCTCGreedyDecoderOpName, kDropoutGenMaskOpName,        kMaskedSelectOpName, kDynamicStitchOpName,
  kGetNextOpName,          kNonMaxSuppressionV3OpName,   kCoalesceOpName};

const std::set<std::string> k3DFormatSet = {kOpFormat_NCDHW, kOpFormat_NDC1HWC0, kOpFormat_FRACTAL_Z_3D,
                                            kOpFormat_NDHWC, kOpFormat_DHWCN,    kOpFormat_DHWNC};

const std::set<std::string> kNoPaddingFormatSet = {kOpFormat_ChannelLast, kOpFormat_FRAC_NZ, kOpFormat_FRACTAL_ZN_RNN,
                                                   kOpFormat_ND_RNN_BIAS};

const std::set<std::string> DynamicShapeConstInputToAttr = {kCastOpName,      kExpandDimsOpName, kEmbeddingLookupOpName,
                                                            kReduceMinOpName, kReduceMeanOpName, kReduceMaxOpName,
                                                            kReduceAllOpName, kReduceAnyOpName,  kConcatOpName};

const std::set<std::string> NeedConvertToValueNodeSet = {kStridedSliceGradOpName};

const std::set<std::string> DynamicShapeConstInputToAttrCPU = {
  kCastOpName,      kExpandDimsOpName, kEmbeddingLookupOpName, kReduceMinOpName, kReduceMeanOpName,
  kReduceMaxOpName, kReduceAllOpName,  kReduceAnyOpName,       kConcatOpName,    kReduceSumOpName};

const std::set<std::string> DynamicShapeConstInputToAttrGPU = {
  kCastOpName,      kExpandDimsOpName, kReshapeOpName,   kEmbeddingLookupOpName, kTransposeOpName, kReduceSumOpName,
  kReduceMinOpName, kReduceMeanOpName, kReduceMaxOpName, kReduceAllOpName,       kReduceAnyOpName, kConcatOpName};

// The map between kernel's output and input ref relationship.
// Key is the output index while the value is input index which will be used as the reference of output.
using OutputInputRefMap = std::map<size_t, size_t>;

static inline void ChangeFileMode(const std::string &file_name, mode_t mode) {
  if (access(file_name.c_str(), F_OK) == -1) {
    return;
  }
  try {
    if (chmod(common::SafeCStr(file_name), mode) != 0) {
      MS_LOG(WARNING) << "Change file `" << file_name << "` to mode " << std::oct << mode << " fail.";
    }
  } catch (std::exception &e) {
    MS_LOG(WARNING) << "File `" << file_name << "` change mode failed! May be not exist.";
  }
}

static inline uint64_t GetCurrentUSec() {
  struct timeval tv;
  int ret = gettimeofday(&tv, nullptr);
  if (ret != 0) {
    MS_LOG(EXCEPTION) << "Fail gettimeofday, ret = " << ret;
  }
  return static_cast<uint64_t>(tv.tv_usec + tv.tv_sec * 1000000);
}

#define PROF_START(stage) uint64_t start_usec_##stage = mindspore::GetCurrentUSec()
#define PROF_END(stage)                                                                                     \
  do {                                                                                                      \
    uint64_t end_usec_##stage = mindspore::GetCurrentUSec();                                                \
    MS_LOG(INFO) << "[PROF]" << #stage << " costs " << (end_usec_##stage - start_usec_##stage) << " usec."; \
  } while (0)

#define PROF_MULTI_DEFINE(stage)       \
  do {                                 \
    static uint64_t total_##stage = 0; \
    static uint64_t count_##stage = 0; \
  } while (0)

#define PROF_LOCAL_DEFINE(stage) \
  do {                           \
    uint64_t total_##stage = 0;  \
    uint64_t count_##stage = 0;  \
  } while (0)

#define PROF_MULTI_START(stage) uint64_t start_usec_##stage = mindspore::GetCurrentUSec()

#define PROF_MULTI_END(stage)                                 \
  do {                                                        \
    ++count_##stage;                                          \
    uint64_t end_usec_##stage = mindspore::GetCurrentUSec();  \
    total_##stage += (end_usec_##stage - start_usec_##stage); \
  } while (0)

#define PROF_MULTI_PRINT(stage)                                                                             \
  do {                                                                                                      \
    MS_LOG(INFO) << #stage << " called " << count_##stage << " times, costs " << total_##stage << " usec."; \
  } while (0)
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_UTILS_UTILS_H_
