/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <memory>
#include <utility>
#include "src/ops/ops_utils.h"
#include "mindapi/base/shared_ptr.h"

#ifdef PRIMITIVE_WRITEABLE
#include "mindspore/core/ir/anf.h"

namespace mindspore {
namespace lite {
std::unique_ptr<schema::PrimitiveT> GetPrimitiveT(const AnfNodePtr &node) {
  auto prim = GetValueNode<std::shared_ptr<Primitive>>(node);
  if (prim == nullptr) {
    MS_LOG(DEBUG) << "primitive is nullptr";
    return nullptr;
  }

  if (prim->name().empty()) {
    MS_LOG(ERROR) << "the name of primitive is null";
    return nullptr;
  }

  MS_LOG(DEBUG) << "export prim: " << prim->name();
  auto creator = MSOpsRegistry::GetInstance()->GetPrimitiveCreator(prim->name());
  if (creator != nullptr) {
    return creator(node);
  } else {
    MS_LOG(WARNING) << "can not find MSOpsRegistry for op: " << prim->name();
    return nullptr;
  }
}

template <typename T>
api::SharedPtr<T> GetOperator(const AnfNodePtr &node) {
  auto prim = GetValueNode<PrimitivePtr>(node);
  if (prim == nullptr) {
    return nullptr;
  }
  return api::MakeShared<T>(prim);
}

std::unique_ptr<schema::PrimitiveT> AbsPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Abs>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> AbsGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::AbsGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ActivationPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Activation>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ActivationGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ActivationGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> AdamPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Adam>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> AdderFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::AdderFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> AddFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::AddFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> AddGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::AddGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> AddNPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::AddN>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> AllPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::All>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ApplyMomentumPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ApplyMomentum>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ArgMaxFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ArgMaxFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ArgMinFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ArgMinFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> AssertPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Assert>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> AssignPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Assign>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> AssignAddPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::AssignAdd>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> AudioSpectrogramPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::AudioSpectrogram>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> AvgPoolFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::AvgPoolFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> AvgPoolGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::AvgPoolGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> BatchNormPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::BatchNorm>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> BatchToSpacePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::BatchToSpace>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> BatchToSpaceNDPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::BatchToSpaceND>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> BiasAddPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::BiasAdd>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> BiasAddGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::BiasAddGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> BNGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::BatchNormGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> BroadcastToPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::BroadcastTo>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> CastPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Cast>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> CeilPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Ceil>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ClipPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Clip>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ConcatPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Concat>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ConstantOfShapePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ConstantOfShape>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> Conv2DBackpropFilterFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Conv2DBackpropFilterFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> Conv2DBackpropInputFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Conv2DBackpropInputFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> Conv2DFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Conv2DFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> Conv2dTransposeFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Conv2dTransposeFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> CosPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Cos>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> CropPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Crop>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> CropAndResizePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::CropAndResize>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> CustomExtractFeaturesPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::CustomExtractFeatures>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> CustomNormalizePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::CustomNormalize>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> CustomPredictPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::CustomPredict>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> DependPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Depend>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> DepthToSpacePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::DepthToSpace>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> DetectionPostProcessPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::DetectionPostProcess>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> DivFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::DivFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> DivGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::DivGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> DropoutPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Dropout>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> DropoutGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::DropoutGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> GRUPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::GRU>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> EltwisePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Eltwise>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> EluPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Elu>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> EmbeddingLookupFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::EmbeddingLookupFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> EqualPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Equal>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ExpandDimsPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ExpandDims>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ExpFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ExpFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> FftImagPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::FftImag>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> FftRealPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::FftReal>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> FillPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Fill>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> FlattenPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Flatten>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> FlattenGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::FlattenGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> FloorPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Floor>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> FloorDivPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::FloorDiv>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> FloorModPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::FloorMod>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> FullConnectionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::FullConnection>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> FusedBatchNormPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::FusedBatchNorm>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> GatherPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Gather>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> GatherNdPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::GatherNd>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> GreaterPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Greater>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> GreaterEqualPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::GreaterEqual>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> HashtableLookupPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::HashtableLookup>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> InstanceNormPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::InstanceNorm>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> InvertPermutationPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::InvertPermutation>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LayerNormFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LayerNormFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LayerNormGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LayerNormGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LeakyReluPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LeakyRelu>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LessPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Less>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LessEqualPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LessEqual>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LogPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Log>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LogGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LogGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LogicalAndPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LogicalAnd>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LogicalNotPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LogicalNot>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LogicalOrPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LogicalOr>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LrnPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LRN>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LpNormalizationPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LpNormalization>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LshProjectionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LshProjection>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LSTMPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LSTM>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LSTMGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LSTMGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LSTMGradDataPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LSTMGradData>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> LSTMGradWeightPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LSTMGradWeight>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> L2NormalizeFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::L2NormalizeFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> MatMulFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::MatMulFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> MaximumPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Maximum>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> MaximumGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::MaximumGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> MaxPoolFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::MaxPoolFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> MaxPoolGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::MaxPoolGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SwitchLayerPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SwitchLayer>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> MfccPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Mfcc>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> MinimumPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Minimum>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> MinimumGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::MinimumGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ModPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Mod>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> MulFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::MulFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> MulGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::MulGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> NegPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Neg>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> NegGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::NegGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> NotEqualPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::NotEqual>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> NonMaxSuppressionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::NonMaxSuppression>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> OneHotPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::OneHot>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> OnesLikePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::OnesLike>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> PadFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::PadFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> PartialFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::PartialFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> PowerGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::PowerGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> PowFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::PowFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> PReLUFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::PReLUFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> QuantDTypeCastPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::QuantDTypeCast>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> RaggedRangePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::RaggedRange>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> RangePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Range>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> RandomStandardNormalPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::RandomStandardNormal>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> RankPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Rank>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> RealDivPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::RealDiv>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ReciprocalPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Reciprocal>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ReduceFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ReduceFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ReshapePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Reshape>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ResizePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Resize>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ResizeGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ResizeGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ReverseV2PrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ReverseV2>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ReverseSequencePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ReverseSequence>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> RfftPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Rfft>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ROIPoolingPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ROIPooling>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> RoundPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Round>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> RsqrtPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Rsqrt>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> RsqrtGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::RsqrtGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ScaleFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ScaleFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ScatterNdPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ScatterNd>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SelectPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Select>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SGDPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SGD>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ShapePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Shape>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SigmoidCrossEntropyWithLogitsPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SigmoidCrossEntropyWithLogits>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SigmoidCrossEntropyWithLogitsGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SigmoidCrossEntropyWithLogitsGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SinPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Sin>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SizePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Size>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SkipGramPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SkipGram>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SliceFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SliceFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SmoothL1LossPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SmoothL1Loss>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SmoothL1LossGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SmoothL1LossGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SoftmaxPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Softmax>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SoftmaxCrossEntropyWithLogitsPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SoftmaxCrossEntropyWithLogits>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SpaceToBatchPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SpaceToBatch>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SpaceToBatchNDPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SpaceToBatchND>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SpaceToDepthPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SpaceToDepth>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SparseSoftmaxCrossEntropyWithLogitsPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SparseSoftmaxCrossEntropyWithLogits>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SparseToDensePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SparseToDense>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SplitPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Split>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SqrtPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Sqrt>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SqrtGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SqrtGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SquarePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Square>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SquaredDifferencePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SquaredDifference>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SqueezePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Squeeze>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> StackPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Stack>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> StridedSlicePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::StridedSlice>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> StridedSliceGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::StridedSliceGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SubFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SubFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SubGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SubGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> SwitchPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Switch>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> TensorListFromTensorPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::TensorListFromTensor>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> TensorListGetItemPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::TensorListGetItem>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> TensorListReservePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::TensorListReserve>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> TensorListSetItemPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::TensorListSetItem>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> TensorListStackPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::TensorListStack>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> TileFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::TileFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> TopKFusionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::TopKFusion>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> TransposePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Transpose>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> UniquePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Unique>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> UnstackPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Unstack>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> UnsortedSegmentSumPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::UnsortedSegmentSum>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> UnsqueezePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Unsqueeze>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> WherePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Where>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ZerosLikePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ZerosLike>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}
std::unique_ptr<schema::PrimitiveT> ErfPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Erf>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> SplicePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Splice>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> LogSoftmaxPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::LogSoftmax>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> CallPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Call>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> CumSumPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::CumSum>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> SplitWithOverlapPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::SplitWithOverlap>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> GluPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::GLU>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> TensorArrayPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::TensorArray>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> TensorArrayReadPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::TensorArrayRead>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> TensorArrayWritePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::TensorArrayWrite>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> AffinePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Affine>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> AttentionPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Attention>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> ScatterNdUpdatePrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ScatterNdUpdate>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> AllGatherPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::AllGather>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> ReduceScatterPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::ReduceScatter>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> DynamicQuantPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::DynamicQuant>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> RandomNormalPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::RandomNormal>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> NLLLossPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::NLLLoss>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> NLLLossGradPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::NLLLossGrad>(node);
  return ms_primc != nullptr ? ops::MSOp2SchemaOp(ms_primc.get()) : nullptr;
}

std::unique_ptr<schema::PrimitiveT> CustomPrimitiveCreator(const AnfNodePtr &node) {
  auto ms_primc = GetOperator<mindspore::ops::Custom>(node);
  auto schema_op = std::make_unique<schema::CustomT>();
  if (schema_op == nullptr) {
    return nullptr;
  }
  if (ms_primc->GetAttr("type") != nullptr) {
    schema_op->type = ms_primc->get_type();
  }
  if (ms_primc->GetAttr("attr") != nullptr) {
    auto attr_map = ms_primc->get_attr();
    for (const auto &attr_item : attr_map) {
      auto attr = std::make_unique<schema::AttributeT>();
      if (attr == nullptr) {
        return nullptr;
      }
      attr->name = attr_item.first;
      attr->data = attr_item.second;
      schema_op->attr.emplace_back(std::move(attr));
    }
  }

  auto prim = std::make_unique<schema::PrimitiveT>();
  if (prim == nullptr) {
    return nullptr;
  }
  prim->value.value = schema_op.release();
  prim->value.type = schema::PrimitiveType_Custom;
  return prim;
}

RegistryMSOps g_absPrimitiveCreatorRegistry("Abs", AbsPrimitiveCreator);
RegistryMSOps g_absGradPrimitiveCreatorRegistry("AbsGrad", AbsGradPrimitiveCreator);
RegistryMSOps g_activationPrimitiveCreatorRegistry("Activation", ActivationPrimitiveCreator);
RegistryMSOps g_activationGradPrimitiveCreatorRegistry("ActivationGrad", ActivationGradPrimitiveCreator);
RegistryMSOps g_reluGradPrimitiveCreatorRegistry("ReluGrad", ActivationGradPrimitiveCreator);  // ?
RegistryMSOps g_addPrimitiveCreatorRegistry("Add", AddFusionPrimitiveCreator);
RegistryMSOps g_addFusionPrimitiveCreatorRegistry("AddFusion", AddFusionPrimitiveCreator);
RegistryMSOps g_addGradPrimitiveCreatorRegistry("AddGrad", AddGradPrimitiveCreator);
RegistryMSOps g_adamPrimitiveCreatorRegistry("Adam", AdamPrimitiveCreator);
RegistryMSOps g_adderPrimitiveCreatorRegistry("Adder", AdderFusionPrimitiveCreator);
RegistryMSOps g_adderFusionPrimitiveCreatorRegistry("AdderFusion", AdderFusionPrimitiveCreator);
RegistryMSOps g_addNPrimitiveCreatorRegistry("AddN", AddNPrimitiveCreator);
RegistryMSOps g_allPrimitiveCreatorRegistry("All", AllPrimitiveCreator);
RegistryMSOps g_applyMomentumPrimitiveCreatorRegistry("ApplyMomentum", ApplyMomentumPrimitiveCreator);
RegistryMSOps g_argMaxPrimitiveCreatorRegistry("ArgMax", ArgMaxFusionPrimitiveCreator);
RegistryMSOps g_argMaxFusionPrimitiveCreatorRegistry("ArgMaxFusion", ArgMaxFusionPrimitiveCreator);
RegistryMSOps g_argMinPrimitiveCreatorRegistry("ArgMin", ArgMinFusionPrimitiveCreator);
RegistryMSOps g_argMinFusionPrimitiveCreatorRegistry("ArgMinFusion", ArgMinFusionPrimitiveCreator);
RegistryMSOps g_assertPrimitiveCreatorRegistry("Assert", AssertPrimitiveCreator);
RegistryMSOps g_assignPrimitiveCreatorRegistry("Assign", AssignPrimitiveCreator);
RegistryMSOps g_assignAddPrimitiveCreatorRegistry("AssignAdd", AssignAddPrimitiveCreator);
RegistryMSOps g_audioSpectrogramPrimitiveCreatorRegistry("AudioSpectrogram", AudioSpectrogramPrimitiveCreator);
RegistryMSOps g_avgPoolPrimitiveCreatorRegistry("AvgPool", AvgPoolFusionPrimitiveCreator);
RegistryMSOps g_avgPoolFusionPrimitiveCreatorRegistry("AvgPoolFusion", AvgPoolFusionPrimitiveCreator);
RegistryMSOps g_avgPoolGradPrimitiveCreatorRegistry("AvgPoolGrad", AvgPoolGradPrimitiveCreator);
RegistryMSOps g_avgPoolGradGpuPrimitiveCreatorRegistry("AvgPoolGradGpu", AvgPoolGradPrimitiveCreator);
RegistryMSOps g_avgPoolGradCpuPrimitiveCreatorRegistry("AvgPoolGradCpu", AvgPoolGradPrimitiveCreator);
RegistryMSOps g_batchNormPrimitiveCreatorRegistry("BatchNorm", BatchNormPrimitiveCreator);
RegistryMSOps g_batchToSpacePrimitiveCreatorRegistry("BatchToSpace", BatchToSpacePrimitiveCreator);
RegistryMSOps g_batchToSpaceNDPrimitiveCreatorRegistry("BatchToSpaceND", BatchToSpaceNDPrimitiveCreator);
RegistryMSOps g_biasAddPrimitiveCreatorRegistry("BiasAdd", BiasAddPrimitiveCreator);
RegistryMSOps g_biasGradAddPrimitiveCreatorRegistry("BiasGrad", BiasAddGradPrimitiveCreator);
RegistryMSOps g_biasAddGradAddPrimitiveCreatorRegistry("BiasAddGrad", BiasAddGradPrimitiveCreator);
RegistryMSOps g_bNGradPrimitiveCreatorRegistry("BatchNormGrad", BNGradPrimitiveCreator);
RegistryMSOps g_broadcastToPrimitiveCreatorRegistry("BroadcastTo", BroadcastToPrimitiveCreator);
RegistryMSOps g_castPrimitiveCreatorRegistry("Cast", CastPrimitiveCreator);
RegistryMSOps g_ceilPrimitiveCreatorRegistry("Ceil", CeilPrimitiveCreator);
RegistryMSOps g_clipPrimitiveCreatorRegistry("Clip", ClipPrimitiveCreator);
RegistryMSOps g_concatPrimitiveCreatorRegistry("Concat", ConcatPrimitiveCreator);
RegistryMSOps g_conv2DBackpropFilterFusionPrimitiveCreatorRegistry("Conv2DBackpropFilterFusion",
                                                                   Conv2DBackpropFilterFusionPrimitiveCreator);
RegistryMSOps g_conv2DBackpropInputFusionPrimitiveCreatorRegistry("Conv2DBackpropInputFusion",
                                                                  Conv2DBackpropInputFusionPrimitiveCreator);
RegistryMSOps g_conv2DPrimitiveCreatorRegistry("Conv2D", Conv2DFusionPrimitiveCreator);
RegistryMSOps g_conv2DFusionPrimitiveCreatorRegistry("Conv2DFusion", Conv2DFusionPrimitiveCreator);
RegistryMSOps g_conv2dTransposePrimitiveCreatorRegistry("Conv2dTranspose", Conv2dTransposeFusionPrimitiveCreator);
RegistryMSOps g_conv2dTransposeFusionPrimitiveCreatorRegistry("Conv2dTransposeFusion",
                                                              Conv2dTransposeFusionPrimitiveCreator);
RegistryMSOps g_constantOfShapePrimitiveCreatorRegistry("ConstantOfShape", ConstantOfShapePrimitiveCreator);
RegistryMSOps g_cosPrimitiveCreatorRegistry("Cos", CosPrimitiveCreator);
RegistryMSOps g_cropPrimitiveCreatorRegistry("Crop", CropPrimitiveCreator);
RegistryMSOps g_cropAndResizePrimitiveCreatorRegistry("CropAndResize", CropAndResizePrimitiveCreator);
RegistryMSOps g_customExtractFeaturesPrimitiveCreatorRegistry("CustomExtractFeatures",
                                                              CustomExtractFeaturesPrimitiveCreator);
RegistryMSOps g_customNormalizePrimitiveCreatorRegistry("CustomNormalize", CustomNormalizePrimitiveCreator);
RegistryMSOps g_customPredictPrimitiveCreatorRegistry("CustomPredict", CustomPredictPrimitiveCreator);
RegistryMSOps g_dependPrimitiveCreatorRegistry("Depend", DependPrimitiveCreator);
RegistryMSOps g_depthToSpacePrimitiveCreatorRegistry("DepthToSpace", DepthToSpacePrimitiveCreator);
RegistryMSOps g_detectionPostProcessPrimitiveCreatorRegistry("DetectionPostProcess",
                                                             DetectionPostProcessPrimitiveCreator);
RegistryMSOps g_divPrimitiveCreatorRegistry("Div", DivFusionPrimitiveCreator);
RegistryMSOps g_divFusionPrimitiveCreatorRegistry("DivFusion", DivFusionPrimitiveCreator);
RegistryMSOps g_divGradPrimitiveCreatorRegistry("DivGrad", DivGradPrimitiveCreator);
RegistryMSOps g_dropoutPrimitiveCreatorRegistry("Dropout", DropoutPrimitiveCreator);
RegistryMSOps g_dropoutGradPrimitiveCreatorRegistry("DropoutGrad", DropoutGradPrimitiveCreator);
RegistryMSOps g_eltwisePrimitiveCreatorRegistry("Eltwise", EltwisePrimitiveCreator);
RegistryMSOps g_eluPrimitiveCreatorRegistry("Elu", EluPrimitiveCreator);
RegistryMSOps g_eluGradPrimitiveCreatorRegistry("EluGrad", ActivationGradPrimitiveCreator);
RegistryMSOps g_equalPrimitiveCreatorRegistry("Equal", EqualPrimitiveCreator);
RegistryMSOps g_embeddingLookupFusionPrimitiveCreatorRegistry("EmbeddingLookupFusion",
                                                              EmbeddingLookupFusionPrimitiveCreator);
RegistryMSOps g_expandDimsPrimitiveCreatorRegistry("ExpandDims", ExpandDimsPrimitiveCreator);
RegistryMSOps g_expPrimitiveCreatorRegistry("Exp", ExpFusionPrimitiveCreator);
RegistryMSOps g_expFusionPrimitiveCreatorRegistry("ExpFusion", ExpFusionPrimitiveCreator);
RegistryMSOps g_fftImagPrimitiveCreatorRegistry("FftImag", FftImagPrimitiveCreator);
RegistryMSOps g_fftRealPrimitiveCreatorRegistry("FftReal", FftRealPrimitiveCreator);
RegistryMSOps g_fillPrimitiveCreatorRegistry("Fill", FillPrimitiveCreator);
RegistryMSOps g_flattenPrimitiveCreatorRegistry("Flatten", FlattenPrimitiveCreator);
RegistryMSOps g_flattenGradPrimitiveCreatorRegistry("FlattenGrad", FlattenGradPrimitiveCreator);
RegistryMSOps g_floorPrimitiveCreatorRegistry("Floor", FloorPrimitiveCreator);
RegistryMSOps g_floorDivPrimitiveCreatorRegistry("FloorDiv", FloorDivPrimitiveCreator);
RegistryMSOps g_floorModPrimitiveCreatorRegistry("FloorMod", FloorModPrimitiveCreator);
RegistryMSOps g_fullConnectionPrimitiveCreatorRegistry("FullConnection", FullConnectionPrimitiveCreator);
RegistryMSOps g_fusedBatchNormPrimitiveCreatorRegistry("FusedBatchNorm", FusedBatchNormPrimitiveCreator);
RegistryMSOps g_gatherPrimitiveCreatorRegistry("Gather", GatherPrimitiveCreator);
RegistryMSOps g_gatherNdPrimitiveCreatorRegistry("GatherNd", GatherNdPrimitiveCreator);
RegistryMSOps g_greaterPrimitiveCreatorRegistry("Greater", GreaterPrimitiveCreator);
RegistryMSOps g_greaterEqualPrimitiveCreatorRegistry("GreaterEqual", GreaterEqualPrimitiveCreator);
RegistryMSOps g_gRUPrimitiveCreatorRegistry("GRU", GRUPrimitiveCreator);
RegistryMSOps g_hashtableLookupPrimitiveCreatorRegistry("HashtableLookup", HashtableLookupPrimitiveCreator);
RegistryMSOps g_instanceNormPrimitiveCreatorRegistry("InstanceNorm", InstanceNormPrimitiveCreator);
RegistryMSOps g_invertPermutationPrimitiveCreatorRegistry("InvertPermutation", InvertPermutationPrimitiveCreator);
RegistryMSOps g_layerNormPrimitiveCreatorRegistry("LayerNorm", LayerNormFusionPrimitiveCreator);
RegistryMSOps g_layerNormFusionPrimitiveCreatorRegistry("LayerNormFusion", LayerNormFusionPrimitiveCreator);
RegistryMSOps g_layerNormGradPrimitiveCreatorRegistry("LayerNormGrad", LayerNormGradPrimitiveCreator);
RegistryMSOps g_leakyReluPrimitiveCreatorRegistry("LeakyRelu", LeakyReluPrimitiveCreator);
RegistryMSOps g_lessPrimitiveCreatorRegistry("Less", LessPrimitiveCreator);
RegistryMSOps g_lessEqualPrimitiveCreatorRegistry("LessEqual", LessEqualPrimitiveCreator);
RegistryMSOps g_logPrimitiveCreatorRegistry("Log", LogPrimitiveCreator);
RegistryMSOps g_logGradPrimitiveCreatorRegistry("LogGrad", LogGradPrimitiveCreator);
RegistryMSOps g_logicalAndPrimitiveCreatorRegistry("LogicalAnd", LogicalAndPrimitiveCreator);
RegistryMSOps g_logicalNotPrimitiveCreatorRegistry("LogicalNot", LogicalNotPrimitiveCreator);
RegistryMSOps g_logicalOrPrimitiveCreatorRegistry("LogicalOr", LogicalOrPrimitiveCreator);
RegistryMSOps g_lpNormalizationPrimitiveCreatorRegistry("LpNormalization", LpNormalizationPrimitiveCreator);
RegistryMSOps g_lrnPrimitiveCreatorRegistry("LRN", LrnPrimitiveCreator);
RegistryMSOps g_lshProjectionPrimitiveCreatorRegistry("LshProjection", LshProjectionPrimitiveCreator);
RegistryMSOps g_lSTMPrimitiveCreatorRegistry("LSTM", LSTMPrimitiveCreator);
RegistryMSOps g_lSTMGradPrimitiveCreatorRegistry("LSTMGrad", LSTMGradPrimitiveCreator);
RegistryMSOps g_lSTMGradDataPrimitiveCreatorRegistry("LSTMGradData", LSTMGradDataPrimitiveCreator);
RegistryMSOps g_lSTMGradWeightPrimitiveCreatorRegistry("LSTMGradWeight", LSTMGradWeightPrimitiveCreator);
RegistryMSOps g_l2NormalizeFusionPrimitiveCreatorRegistry("L2NormalizeFusion", L2NormalizeFusionPrimitiveCreator);
RegistryMSOps g_matMulFusionPrimitiveCreatorRegistry("MatMulFusion", MatMulFusionPrimitiveCreator);
RegistryMSOps g_matMulPrimitiveCreatorRegistry("MatMul", MatMulFusionPrimitiveCreator);
RegistryMSOps g_maximumPrimitiveCreatorRegistry("Maximum", MaximumPrimitiveCreator);
RegistryMSOps g_maximumGradPrimitiveCreatorRegistry("MaximumGrad", MaximumGradPrimitiveCreator);
RegistryMSOps g_maxPoolPrimitiveCreatorRegistry("MaxPool", MaxPoolFusionPrimitiveCreator);
RegistryMSOps g_maxPoolFusionPrimitiveCreatorRegistry("MaxPoolFusion", MaxPoolFusionPrimitiveCreator);
RegistryMSOps g_maxPoolGradPrimitiveCreatorRegistry("MaxPoolGrad", MaxPoolGradPrimitiveCreator);
RegistryMSOps g_mergePrimitiveCreatorRegistry("switch_layer", SwitchLayerPrimitiveCreator);
RegistryMSOps g_mfccPrimitiveCreatorRegistry("Mfcc", MfccPrimitiveCreator);
RegistryMSOps g_minimumPrimitiveCreatorRegistry("Minimum", MinimumPrimitiveCreator);
RegistryMSOps g_minimumGradPrimitiveCreatorRegistry("MinimumGrad", MinimumGradPrimitiveCreator);
RegistryMSOps g_modPrimitiveCreatorRegistry("Mod", ModPrimitiveCreator);
RegistryMSOps g_mulPrimitiveCreatorRegistry("Mul", MulFusionPrimitiveCreator);
RegistryMSOps g_mulMulFusionPrimitiveCreatorRegistry("MulFusion", MulFusionPrimitiveCreator);
RegistryMSOps g_mulGradPrimitiveCreatorRegistry("MulGrad", MulGradPrimitiveCreator);
RegistryMSOps g_negPrimitiveCreatorRegistry("Neg", NegPrimitiveCreator);
RegistryMSOps g_negGradPrimitiveCreatorRegistry("NegGrad", NegGradPrimitiveCreator);
RegistryMSOps g_nonMaxSuppressionPrimitiveCreatorRegistry("NonMaxSuppression", NonMaxSuppressionPrimitiveCreator);
RegistryMSOps g_notEqualPrimitiveCreatorRegistry("NotEqual", NotEqualPrimitiveCreator);
RegistryMSOps g_oneHotPrimitiveCreatorRegistry("OneHot", OneHotPrimitiveCreator);
RegistryMSOps g_onesLikePrimitiveCreatorRegistry("OnesLike", OnesLikePrimitiveCreator);
RegistryMSOps g_padPrimitiveCreatorRegistry("Pad", PadFusionPrimitiveCreator);
RegistryMSOps g_padFusionPrimitiveCreatorRegistry("PadFusion", PadFusionPrimitiveCreator);
RegistryMSOps g_partialFusionPrimitiveCreatorRegistry("PartialFusion", PartialFusionPrimitiveCreator);
RegistryMSOps g_powerGradPrimitiveCreatorRegistry("PowerGrad", PowerGradPrimitiveCreator);
RegistryMSOps g_powFusionPrimitiveCreatorRegistry("PowFusion", PowFusionPrimitiveCreator);
RegistryMSOps g_pReLUFusionPrimitiveCreatorRegistry("PReLUFusion", PReLUFusionPrimitiveCreator);
RegistryMSOps g_RandomStandardNormalPrimitiveCreatorRegistry("RandomStandardNormal",
                                                             RandomStandardNormalPrimitiveCreator);
RegistryMSOps g_StandardNormalPrimitiveCreatorRegistry("StandardNormal", RandomStandardNormalPrimitiveCreator);
RegistryMSOps g_raggedRangePrimitiveCreatorRegistry("RaggedRange", RaggedRangePrimitiveCreator);
RegistryMSOps g_rangePrimitiveCreatorRegistry("Range", RangePrimitiveCreator);
RegistryMSOps g_rankPrimitiveCreatorRegistry("Rank", RankPrimitiveCreator);
RegistryMSOps g_reciprocalPrimitiveCreatorRegistry("Reciprocal", ReciprocalPrimitiveCreator);
RegistryMSOps g_realDivPrimitiveCreatorRegistry("RealDiv", RealDivPrimitiveCreator);
RegistryMSOps g_reducePrimitiveCreatorRegistry("Reduce", ReduceFusionPrimitiveCreator);
RegistryMSOps g_reduceFusionPrimitiveCreatorRegistry("ReduceFusion", ReduceFusionPrimitiveCreator);
RegistryMSOps g_reshapePrimitiveCreatorRegistry("Reshape", ReshapePrimitiveCreator);
RegistryMSOps g_resizePrimitiveCreatorRegistry("Resize", ResizePrimitiveCreator);
RegistryMSOps g_resizeGradPrimitiveCreatorRegistry("ResizeGrad", ResizeGradPrimitiveCreator);
RegistryMSOps g_resizeBilinearGradPrimitiveCreatorRegistry("ResizeBilinearGrad", ResizeGradPrimitiveCreator);
RegistryMSOps g_resizeNearestNeighborGradPrimitiveCreatorRegistry("ResizeNearestNeighborGrad",
                                                                  ResizeGradPrimitiveCreator);
RegistryMSOps g_reverseV2PrimitiveCreatorRegistry("ReverseV2", ReverseV2PrimitiveCreator);
RegistryMSOps g_reverseSequencePrimitiveCreatorRegistry("ReverseSequence", ReverseSequencePrimitiveCreator);
RegistryMSOps g_rfftPrimitiveCreatorRegistry("Rfft", RfftPrimitiveCreator);
RegistryMSOps g_rOIPoolingPrimitiveCreatorRegistry("ROIPooling", ROIPoolingPrimitiveCreator);
RegistryMSOps g_roundPrimitiveCreatorRegistry("Round", RoundPrimitiveCreator);
RegistryMSOps g_rsqrtPrimitiveCreatorRegistry("Rsqrt", RsqrtPrimitiveCreator);
RegistryMSOps g_rsqrtGradPrimitiveCreatorRegistry("RsqrtGrad", RsqrtGradPrimitiveCreator);
RegistryMSOps g_quantDTypeCastPrimitiveCreatorRegistry("QuantDTypeCast", QuantDTypeCastPrimitiveCreator);
RegistryMSOps g_scalePrimitiveCreatorRegistry("Scale", ScaleFusionPrimitiveCreator);
RegistryMSOps g_scaleFusionPrimitiveCreatorRegistry("ScaleFusion", ScaleFusionPrimitiveCreator);
RegistryMSOps g_scatterNdPrimitiveCreatorRegistry("ScatterNd", ScatterNdPrimitiveCreator);
RegistryMSOps g_selectPrimitiveCreatorRegistry("Select", SelectPrimitiveCreator);
RegistryMSOps g_SGDPrimitiveCreatorRegistry("SGD", SGDPrimitiveCreator);
RegistryMSOps g_shapePrimitiveCreatorRegistry("Shape", ShapePrimitiveCreator);
RegistryMSOps g_sigmoidCrossEntropyWithLogitsPrimitiveCreatorRegistry("SigmoidCrossEntropyWithLogits",
                                                                      SigmoidCrossEntropyWithLogitsPrimitiveCreator);
RegistryMSOps g_sigmoidCrossEntropyWithLogitsGradPrimitiveCreatorRegistry(
  "SigmoidCrossEntropyWithLogitsGrad", SigmoidCrossEntropyWithLogitsGradPrimitiveCreator);
RegistryMSOps g_sinPrimitiveCreatorRegistry("Sin", SinPrimitiveCreator);
RegistryMSOps g_sizePrimitiveCreatorRegistry("Size", SizePrimitiveCreator);
RegistryMSOps g_skipGramPrimitiveCreatorRegistry("SkipGram", SkipGramPrimitiveCreator);
RegistryMSOps g_sliceFusionPrimitiveCreatorRegistry("SliceFusion", SliceFusionPrimitiveCreator);
RegistryMSOps g_smoothL1LossPrimitiveCreatorRegistry("SmoothL1Loss", SmoothL1LossPrimitiveCreator);
RegistryMSOps g_smoothL1LossGradPrimitiveCreatorRegistry("SmoothL1LossGrad", SmoothL1LossGradPrimitiveCreator);
RegistryMSOps g_softmaxPrimitiveCreatorRegistry("Softmax", SoftmaxPrimitiveCreator);
RegistryMSOps g_softmaxCrossEntropyWithLogitsPrimitiveCreatorRegistry("SoftmaxCrossEntropyWithLogits",
                                                                      SoftmaxCrossEntropyWithLogitsPrimitiveCreator);
RegistryMSOps g_spaceToBatchPrimitiveCreatorRegistry("SpaceToBatch", SpaceToBatchPrimitiveCreator);
RegistryMSOps g_spaceToBatchNDPrimitiveCreatorRegistry("SpaceToBatchND", SpaceToBatchNDPrimitiveCreator);
RegistryMSOps g_spaceToDepthPrimitiveCreatorRegistry("SpaceToDepth", SpaceToDepthPrimitiveCreator);
RegistryMSOps g_sparseSoftmaxCrossEntropyWithLogitsPrimitiveCreatorRegistry(
  "SparseSoftmaxCrossEntropyWithLogits", SparseSoftmaxCrossEntropyWithLogitsPrimitiveCreator);
RegistryMSOps g_sparseToDensePrimitiveCreatorRegistry("SparseToDense", SparseToDensePrimitiveCreator);
RegistryMSOps g_splitPrimitiveCreatorRegistry("Split", SplitPrimitiveCreator);
RegistryMSOps g_sqrtPrimitiveCreatorRegistry("Sqrt", SqrtPrimitiveCreator);
RegistryMSOps g_sqrtGradPrimitiveCreatorRegistry("SqrtGrad", SqrtGradPrimitiveCreator);
RegistryMSOps g_squeezePrimitiveCreatorRegistry("Squeeze", SqueezePrimitiveCreator);
RegistryMSOps g_squarePrimitiveCreatorRegistry("Square", SquarePrimitiveCreator);
RegistryMSOps g_squaredDifferencePrimitiveCreatorRegistry("SquaredDifference", SquaredDifferencePrimitiveCreator);
RegistryMSOps g_stackPrimitiveCreatorRegistry("Stack", StackPrimitiveCreator);
RegistryMSOps g_stridedSlicePrimitiveCreatorRegistry("StridedSlice", StridedSlicePrimitiveCreator);
RegistryMSOps g_stridedSliceGradPrimitiveCreatorRegistry("StridedSliceGrad", StridedSliceGradPrimitiveCreator);
RegistryMSOps g_subPrimitiveCreatorRegistry("Sub", SubFusionPrimitiveCreator);
RegistryMSOps g_subFusionPrimitiveCreatorRegistry("SubFusion", SubFusionPrimitiveCreator);
RegistryMSOps g_subGradPrimitiveCreatorRegistry("SubGrad", SubGradPrimitiveCreator);
RegistryMSOps g_switchPrimitiveCreatorRegistry("Switch", SwitchPrimitiveCreator);
RegistryMSOps g_tensorListFromTensorPrimitiveCreatorRegistry("TensorListFromTensor",
                                                             TensorListFromTensorPrimitiveCreator);
RegistryMSOps g_tensorListGetItemPrimitiveCreatorRegistry("TensorListGetItem", TensorListGetItemPrimitiveCreator);
RegistryMSOps g_tensorListReservePrimitiveCreatorRegistry("TensorListReserve", TensorListReservePrimitiveCreator);
RegistryMSOps g_tensorListSetItemPrimitiveCreatorRegistry("TensorListSetItem", TensorListSetItemPrimitiveCreator);
RegistryMSOps g_tensorListStackPrimitiveCreatorRegistry("TensorListStack", TensorListStackPrimitiveCreator);
RegistryMSOps g_tileFusionPrimitiveCreatorRegistry("TileFusion", TileFusionPrimitiveCreator);
RegistryMSOps g_topKPrimitiveCreatorRegistry("TopK", TopKFusionPrimitiveCreator);
RegistryMSOps g_topKFusionPrimitiveCreatorRegistry("TopKFusion", TopKFusionPrimitiveCreator);
RegistryMSOps g_transposePrimitiveCreatorxRegistry("Transpose", TransposePrimitiveCreator);
RegistryMSOps g_uniquePrimitiveCreatorRegistry("Unique", UniquePrimitiveCreator);
RegistryMSOps g_unstackPrimitiveCreatorRegistry("Unstack", UnstackPrimitiveCreator);
RegistryMSOps g_unsortedSegmentSumPrimitiveCreatorRegistry("UnsortedSegmentSum", UnsortedSegmentSumPrimitiveCreator);
RegistryMSOps g_unsqueezePrimitiveCreatorRegistry("Unsqueeze", UnsqueezePrimitiveCreator);
RegistryMSOps g_wherePrimitiveCreatorRegistry("Where", WherePrimitiveCreator);
RegistryMSOps g_zerosLikePrimitiveCreatorRegistry("ZerosLike", ZerosLikePrimitiveCreator);
RegistryMSOps g_erfPrimitiveCreatorRegistry("Erf", ErfPrimitiveCreator);
RegistryMSOps g_SplicePrimitiveCreatorRegistry("Splice", SplicePrimitiveCreator);
RegistryMSOps g_LogSoftmaxPrimitiveCreatorRegistry("LogSoftmax", LogSoftmaxPrimitiveCreator);
RegistryMSOps g_CallPrimitiveCreatorRegistry("call", CallPrimitiveCreator);
RegistryMSOps g_CumSumPrimitiveCreatorRegistry("CumSum", CumSumPrimitiveCreator);
RegistryMSOps g_SplitWithOverlapCreatorRegistry("SplitWithOverlap", SplitWithOverlapPrimitiveCreator);
RegistryMSOps g_GluCreatorRegistry("GLU", GluPrimitiveCreator);
RegistryMSOps g_TensorArrayCreatorRegistry("TensorArray", TensorArrayPrimitiveCreator);
RegistryMSOps g_TensorArrayReadCreatorRegistry("TensorArrayRead", TensorArrayReadPrimitiveCreator);
RegistryMSOps g_TensorArrayWriteCreatorRegistry("TensorArrayWrite", TensorArrayWritePrimitiveCreator);
RegistryMSOps g_AffineCreatorRegistry("Affine", AffinePrimitiveCreator);
RegistryMSOps g_AttentionCreatorRegistry("Attention", AttentionPrimitiveCreator);
RegistryMSOps g_ScatterNdUpdateCreatorRegistry("ScatterNdUpdate", ScatterNdUpdatePrimitiveCreator);
RegistryMSOps g_AllGatherCreatorRegistry("AllGather", AllGatherPrimitiveCreator);
RegistryMSOps g_ReduceScatterCreatorRegistry("ReduceScatter", ReduceScatterPrimitiveCreator);
RegistryMSOps g_DynamicQuantCreatorRegistry("DynamicQuant", DynamicQuantPrimitiveCreator);
RegistryMSOps g_RandomNormalCreatorRegistry("RandomNormal", RandomNormalPrimitiveCreator);
RegistryMSOps g_NLLLossCreatorRegistry("NLLLoss", NLLLossPrimitiveCreator);
RegistryMSOps g_NLLLossGradCreatorRegistry("NLLLossGrad", NLLLossGradPrimitiveCreator);
RegistryMSOps g_CustomPrimitiveCreatorRegistry("Custom", CustomPrimitiveCreator);
}  // namespace lite
}  // namespace mindspore

#endif
