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
#include <x86intrin.h>

// nnacl gemm in x86 fma asm code
void nnacl_gemm_fma_7x8_kernel_nc8hw8_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                           const size_t act_flag, const size_t row_block, const size_t col_block,
                                           const size_t deep, const size_t src_stride, const size_t dst_stride,
                                           const size_t inc_flag) {
  size_t deep_t = deep >> 3;
  size_t dst_stride_t = dst_stride << 2;
  size_t src_stride_t = src_stride << 2;
  asm volatile(
    // inc in deep
    "and $0x1, %[inc_flag]\n"
    "je 0f\n"
    "vmovups 0(%[dst]), %%ymm0\n"
    "vmovups 32(%[dst]), %%ymm1\n"
    "vmovups 64(%[dst]), %%ymm2\n"
    "vmovups 96(%[dst]), %%ymm3\n"
    "vmovups 128(%[dst]), %%ymm4\n"
    "vmovups 160(%[dst]), %%ymm5\n"
    "vmovups 192(%[dst]), %%ymm6\n"
    "jmp 2f\n"
    "0:\n"
    "cmpq $0, %[bias]\n"
    "je 1f\n"
    "vmovaps 0(%[bias]), %%ymm0\n"
    "vmovaps 0(%[bias]), %%ymm1\n"
    "vmovaps 0(%[bias]), %%ymm2\n"
    "vmovaps 0(%[bias]), %%ymm3\n"
    "vmovaps 0(%[bias]), %%ymm4\n"
    "vmovaps 0(%[bias]), %%ymm5\n"
    "vmovaps 0(%[bias]), %%ymm6\n"
    "jmp 2f\n"
    "1:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"
    "vxorps %%ymm3, %%ymm3, %%ymm3\n"
    "vxorps %%ymm4, %%ymm4, %%ymm4\n"
    "vxorps %%ymm5, %%ymm5, %%ymm5\n"
    "vxorps %%ymm6, %%ymm6, %%ymm6\n"
    "2:\n"
    :
    : [ dst ] "r"(dst), [ bias ] "r"(bias), [ dst_stride ] "r"(dst_stride_t), [ inc_flag ] "r"(inc_flag)
    : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6");
  asm volatile(
    "0:\n"
    // block 0
    "vmovaps 0(%[weight]), %%ymm15\n"
    "vbroadcastss 0(%[src]), %%ymm14\n"
    "vbroadcastss 32(%[src]), %%ymm13\n"
    "vbroadcastss 64(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm15\n"
    "vbroadcastss 96(%[src]), %%ymm14\n"
    "vbroadcastss 128(%[src]), %%ymm13\n"
    "vbroadcastss 160(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm15\n"
    "vbroadcastss 192(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm6, %%ymm14, %%ymm15\n"
    // block 1
    "vmovaps 32(%[weight]), %%ymm15\n"
    "vbroadcastss 1(%[src]), %%ymm14\n"
    "vbroadcastss 33(%[src]), %%ymm13\n"
    "vbroadcastss 65(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm15\n"
    "vbroadcastss 97(%[src]), %%ymm14\n"
    "vbroadcastss 129(%[src]), %%ymm13\n"
    "vbroadcastss 161(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm15\n"
    "vbroadcastss 193(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm6, %%ymm14, %%ymm15\n"
    // block 2
    "vmovaps 64(%[weight]), %%ymm15\n"
    "vbroadcastss 2(%[src]), %%ymm14\n"
    "vbroadcastss 34(%[src]), %%ymm13\n"
    "vbroadcastss 66(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm15\n"
    "vbroadcastss 98(%[src]), %%ymm14\n"
    "vbroadcastss 130(%[src]), %%ymm13\n"
    "vbroadcastss 162(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm15\n"
    "vbroadcastss 194(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm6, %%ymm14, %%ymm15\n"
    // block 3
    "vmovaps 96(%[weight]), %%ymm15\n"
    "vbroadcastss 3(%[src]), %%ymm14\n"
    "vbroadcastss 35(%[src]), %%ymm13\n"
    "vbroadcastss 67(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm15\n"
    "vbroadcastss 99(%[src]), %%ymm14\n"
    "vbroadcastss 131(%[src]), %%ymm13\n"
    "vbroadcastss 163(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm15\n"
    "vbroadcastss 195(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm6, %%ymm14, %%ymm15\n"
    // block 4
    "vmovaps 128(%[weight]), %%ymm15\n"
    "vbroadcastss 4(%[src]), %%ymm14\n"
    "vbroadcastss 36(%[src]), %%ymm13\n"
    "vbroadcastss 68(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm15\n"
    "vbroadcastss 100(%[src]), %%ymm14\n"
    "vbroadcastss 132(%[src]), %%ymm13\n"
    "vbroadcastss 164(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm15\n"
    "vbroadcastss 196(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm6, %%ymm14, %%ymm15\n"
    // block 5
    "vmovaps 160(%[weight]), %%ymm15\n"
    "vbroadcastss 5(%[src]), %%ymm14\n"
    "vbroadcastss 37(%[src]), %%ymm13\n"
    "vbroadcastss 69(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm15\n"
    "vbroadcastss 101(%[src]), %%ymm14\n"
    "vbroadcastss 133(%[src]), %%ymm13\n"
    "vbroadcastss 165(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm15\n"
    "vbroadcastss 197(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm6, %%ymm14, %%ymm15\n"
    // block 6
    "vmovaps 192(%[weight]), %%ymm15\n"
    "vbroadcastss 6(%[src]), %%ymm14\n"
    "vbroadcastss 38(%[src]), %%ymm13\n"
    "vbroadcastss 70(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm15\n"
    "vbroadcastss 102(%[src]), %%ymm14\n"
    "vbroadcastss 134(%[src]), %%ymm13\n"
    "vbroadcastss 166(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm15\n"
    "vbroadcastss 198(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm6, %%ymm14, %%ymm15\n"
    // block 7
    "vmovaps 224(%[weight]), %%ymm15\n"
    "vbroadcastss 7(%[src]), %%ymm14\n"
    "vbroadcastss 39(%[src]), %%ymm13\n"
    "vbroadcastss 71(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm15\n"
    "vbroadcastss 103(%[src]), %%ymm14\n"
    "vbroadcastss 135(%[src]), %%ymm13\n"
    "vbroadcastss 167(%[src]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm14, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm13, %%ymm15\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm15\n"
    "vbroadcastss 199(%[src]), %%ymm14\n"
    "vfmadd231ps %%ymm6, %%ymm14, %%ymm15\n"
    "dec %[deep]\n"
    "add 256, %[weight]\n"
    "add %[src_stride], %[src]\n"
    "jg 0b\n"

    "movq %[inc_flag], %%rax\n"
    "and $0x2, %%eax\n"
    "je 3f\n"
    "movq %[act_flag], %%rax\n"
    "and $0x3, %%eax\n"
    "je 3f\n"
    // relu
    "vxorps %%ymm15, %%ymm15, %%ymm15\n"
    "vmaxps %%ymm0, %%ymm15, %%ymm0\n"
    "vmaxps %%ymm1, %%ymm15, %%ymm1\n"
    "vmaxps %%ymm2, %%ymm15, %%ymm2\n"
    "vmaxps %%ymm3, %%ymm15, %%ymm3\n"
    "vmaxps %%ymm4, %%ymm15, %%ymm4\n"
    "vmaxps %%ymm5, %%ymm15, %%ymm5\n"
    "vmaxps %%ymm6, %%ymm15, %%ymm6\n"
    "and $0x1, %%eax\n"
    "je 3f\n"
    // relu6
    "mov $0x40C00000, %%eax\n"
    "vmovd %%eax, %%xmm14\n"
    "vpermps %%ymm14, %%ymm15, %%ymm14\n"
    "vminps %%ymm0, %%ymm14, %%ymm0\n"
    "vminps %%ymm1, %%ymm14, %%ymm1\n"
    "vminps %%ymm2, %%ymm14, %%ymm2\n"
    "vminps %%ymm3, %%ymm14, %%ymm3\n"
    "vminps %%ymm4, %%ymm14, %%ymm4\n"
    "vminps %%ymm5, %%ymm14, %%ymm5\n"
    "vminps %%ymm6, %%ymm14, %%ymm6\n"
    "3:\n"
    "vmovups %%ymm0, 0(%[dst])\n"
    "vmovups %%ymm1, 32(%[dst])\n"
    "vmovups %%ymm2, 64(%[dst])\n"
    "vmovups %%ymm3, 96(%[dst])\n"
    "vmovups %%ymm4, 128(%[dst])\n"
    "vmovups %%ymm5, 160(%[dst])\n"
    "vmovups %%ymm6, 192(%[dst])\n"
    :
    : [ src ] "r"(src), [ src_stride ] "r"(src_stride_t), [ weight ] "r"(weight), [ deep ] "r"(deep_t),
      [ inc_flag ] "r"(inc_flag), [ act_flag ] "r"(act_flag), [ dst ] "r"(dst), [ dst_stride ] "r"(dst_stride_t)
    : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
      "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}
