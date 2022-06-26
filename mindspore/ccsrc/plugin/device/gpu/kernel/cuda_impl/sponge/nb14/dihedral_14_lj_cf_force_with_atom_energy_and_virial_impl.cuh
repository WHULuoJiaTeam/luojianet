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
#ifndef MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_NB14_DIHEDRAL_14_LJ_CF_FORCE_WITH_ATOM_ENERGY_IMPL_H
#define MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_NB14_DIHEDRAL_14_LJ_CF_FORCE_WITH_ATOM_ENERGY_IMPL_H

#include <curand_kernel.h>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/cuda_common.h"

void Dihedral14LJCFForceWithAtomEnergyAndVirial(const int dihedral_14_numbers, const int atom_numbers,
                                                const int *uint_crd_f, const int *LJtype, const float *charge,
                                                float *uint_crd_with_LJ_f, const float *boxlength_f, const int *a_14,
                                                const int *b_14, const float *lj_scale_factor,
                                                const float *cf_scale_factor, const float *LJ_type_A,
                                                const float *LJ_type_B, float *frc_f, float *atom_energy,
                                                float *atom_virial, cudaStream_t stream);
#endif  // MINDSPORE_CCSRC_KERNEL_GPU_CUDA_IMPL_SPONGE_NB14_DIHEDRAL_14_LJ_CF_FORCE_WITH_ATOM_ENERGY_IMPL_H
