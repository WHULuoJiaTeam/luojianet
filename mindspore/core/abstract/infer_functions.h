/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
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

#ifndef MINDSPORE_CORE_ABSTRACT_INFER_FUNCTIONS_H_
#define MINDSPORE_CORE_ABSTRACT_INFER_FUNCTIONS_H_
#include <string>
#include <memory>
#include "abstract/abstract_value.h"
#include "abstract/param_validator.h"
#include "base/core_ops.h"
namespace mindspore {
namespace abstract {
AbstractBasePtr InferImplReturn(const AnalysisEnginePtr &, const PrimitivePtr &,
                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSwitch(const AnalysisEnginePtr &, const PrimitivePtr &,
                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSwitchLayer(const AnalysisEnginePtr &, const PrimitivePtr &,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplIs_(const AnalysisEnginePtr &, const PrimitivePtr &,
                             const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplIsNot(const AnalysisEnginePtr &, const PrimitivePtr &,
                               const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplInDict(const AnalysisEnginePtr &, const PrimitivePtr &,
                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplNotInDict(const AnalysisEnginePtr &, const PrimitivePtr &,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplIsConstant(const AnalysisEnginePtr &, const PrimitivePtr &,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplPooling(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplPoolingGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplFusedSparseAdam(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplBatchNorm(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplBiasAddGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplHSigmoid(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplHSigmoidGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplBpropCut(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDropout(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list);

AbstractBasePtr InferImplMinOrMaxGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSqrt(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSqrtGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);

AbstractBasePtr InferImplScalarToArray(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplArrayToScalar(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplBroadCastShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplStack(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list);

AbstractBasePtr InferImplMakeTuple(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeList(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeDict(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplUnsortedSegmentSum(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplUnsortedSegmentMax(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplUnsortedSegmentMin(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeKwarg(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplExtractKwarg(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeRecord(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplTupleGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplListGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplTupleSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplListSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDictGetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDictSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDictGetKeys(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDictGetValues(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDictItems(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplListAppend(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplTupleLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplListLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplArrayLen(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplIdentity(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);

AbstractBasePtr InferImplEnvironCreate(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplEnvironGet(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplEnvironSet(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplEnvironAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplEnvironDestroyAll(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                           const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeRefKey(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeRef(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplGetRefKey(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplGetRefValue(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplStateSetItem(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDepend(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplUpdateState(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDebug(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list);
template <typename T>
std::shared_ptr<T> InferSparseAttr(const PrimitivePtr &primitive, const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeCOOTensor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCOOTensorGetValues(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCOOTensorGetIndices(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCOOTensorGetDenseShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCSRElementWise(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCSRMV(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCSRReduceSum(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCSRGather(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCSR2COO(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCOO2CSR(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplIsCSRFunc(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMakeRowTensor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplRowTensorGetValues(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplRowTensorGetIndices(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplRowTensorGetDenseShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplRowTensorAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);

AbstractBasePtr InferImplMakeCSRTensor(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCSRTensorGetValues(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCSRTensorGetIndices(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCSRTensorGetIndptr(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCSRTensorGetDenseShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                const AbstractBasePtrList &args_spec_list);

AbstractBasePtr InferImplUniqueGrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplUnique(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplNonZero(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCTCGreedyDecoder(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDynamicStitch(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplScatterAdd(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplScatterSub(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplScatterUpdate(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplScatterElements(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDiv(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list);

AbstractBasePtr InferImplSubAndFilter(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMapCacheIdx(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCacheSwapTable(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplUpdateCache(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplComputeAccidentalHits(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                               const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplPadAndShift(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDynamicAssign(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplGatherV2(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSparseApplyFtrl(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSparseApplyProximalAdagrad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplAllSwap(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplAllReduce(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplBroadcast(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplAllGather(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplReduceScatter(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                       const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSGD(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplTranspose(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplReshape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMemCpyAsync(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplEmbeddingLookup(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplReduceFunc(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplCast(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMinimum(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplDivNoNan(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplLinSpace(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplExpandDims(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplGpuConvertToDynamicShape(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                  const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplPad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                             const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMapUniform(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSplit(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSequenceMask(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplConcat(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplConcatOffset(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplRange(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMatMul(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplBatchMatMul(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplLess(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplArgMaxWithValue(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplLoad(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplSort(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplMaskedSelect(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplTransData(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplTensorCopySlices(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplReal(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                              const AbstractBasePtrList &args_spec_list);
AbstractBasePtr InferImplTensorArrayStack(const AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const AbstractBasePtrList &args_spec_list);
template <typename T>
AbstractBasePtr InferTupleOrListOrDictLen(const std::string &op_name, const AbstractBasePtrList &args_spec_list) {
  // Inputs: a tuple or list or dict.
  CheckArgsSize(op_name, args_spec_list, 1);
  auto arg = CheckArg<T>(op_name, args_spec_list, 0);
  return std::make_shared<AbstractScalar>(SizeToLong(arg->size()));
}
}  // namespace abstract
}  // namespace mindspore
#endif  // MINDSPORE_CORE_ABSTRACT_INFER_FUNCTIONS_H_
