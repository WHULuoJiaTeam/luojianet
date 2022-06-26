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
#include "nnacl/kernel.h"
static KernelCreator g_kernelCreatorRegistry[PrimType_MAX][16];

void RegKernelCreator(int opType, int dataType, KernelCreator creator) {
  g_kernelCreatorRegistry[opType][dataType - kNumberTypeBegin - 1] = creator;
}

KernelBase *CreateKernel(OpParameter *param, TensorC *in[], size_t insize, TensorC *out[], size_t outsize) {
  int dtype = in[kInputIndex]->data_type_;
  return g_kernelCreatorRegistry[param->type_][dtype - kNumberTypeBegin - 1](param, in, insize, out, outsize);
}
