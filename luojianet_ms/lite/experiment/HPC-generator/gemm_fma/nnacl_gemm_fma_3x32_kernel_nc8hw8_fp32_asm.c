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
void nnacl_gemm_fma_3x32_kernel_nc8hw8_fp32(float *dst, const float *src, const float *weight, const float *bias,
                                            const size_t act_flag, const size_t row_block, const size_t col_block,
                                            const size_t deep, const size_t src_stride, const size_t dst_stride,
                                            const size_t inc_flag) {
  const float *dst_4 = dst + 3 * dst_stride;
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
    "vmovups 0(%[dst], %[dst_stride], 1), %%ymm3\n"
    "vmovups 32(%[dst], %[dst_stride], 1), %%ymm4\n"
    "vmovups 64(%[dst], %[dst_stride], 1), %%ymm5\n"
    "vmovups 0(%[dst], %[dst_stride], 2), %%ymm6\n"
    "vmovups 32(%[dst], %[dst_stride], 2), %%ymm7\n"
    "vmovups 64(%[dst], %[dst_stride], 2), %%ymm8\n"
    "vmovups 0(%[dst_4]), %%ymm9\n"
    "vmovups 32(%[dst_4]), %%ymm10\n"
    "vmovups 64(%[dst_4]), %%ymm11\n"
    "jmp 2f\n"
    "0:\n"
    "cmpq $0, %[bias]\n"
    "je 1f\n"
    "vmovaps 0(%[bias]), %%ymm0\n"
    "vmovaps 0(%[bias]), %%ymm1\n"
    "vmovaps 0(%[bias]), %%ymm2\n"
    "vmovaps 32(%[bias]), %%ymm3\n"
    "vmovaps 32(%[bias]), %%ymm4\n"
    "vmovaps 32(%[bias]), %%ymm5\n"
    "vmovaps 64(%[bias]), %%ymm6\n"
    "vmovaps 64(%[bias]), %%ymm7\n"
    "vmovaps 64(%[bias]), %%ymm8\n"
    "vmovaps 96(%[bias]), %%ymm9\n"
    "vmovaps 96(%[bias]), %%ymm10\n"
    "vmovaps 96(%[bias]), %%ymm11\n"
    "jmp 2f\n"
    "1:\n"
    "vxorps %%ymm0, %%ymm0, %%ymm0\n"
    "vxorps %%ymm1, %%ymm1, %%ymm1\n"
    "vxorps %%ymm2, %%ymm2, %%ymm2\n"
    "vxorps %%ymm3, %%ymm3, %%ymm3\n"
    "vxorps %%ymm4, %%ymm4, %%ymm4\n"
    "vxorps %%ymm5, %%ymm5, %%ymm5\n"
    "vxorps %%ymm6, %%ymm6, %%ymm6\n"
    "vxorps %%ymm7, %%ymm7, %%ymm7\n"
    "vxorps %%ymm8, %%ymm8, %%ymm8\n"
    "vxorps %%ymm9, %%ymm9, %%ymm9\n"
    "vxorps %%ymm10, %%ymm10, %%ymm10\n"
    "vxorps %%ymm11, %%ymm11, %%ymm11\n"
    "2:\n"
    :
    : [ dst ] "r"(dst), [ bias ] "r"(bias), [ dst_stride ] "r"(dst_stride_t), [ inc_flag ] "r"(inc_flag),
      [ dst_4 ] "r"(dst_4)
    : "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10", "%ymm11");
  asm volatile(
    "0:\n"
    // block 0
    "vbroadcastss 0(%[src]), %%ymm15\n"
    "vbroadcastss 32(%[src]), %%ymm14\n"
    "vbroadcastss 64(%[src]), %%ymm13\n"
    "vmovaps 0(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm13\n"
    "vmovaps 32(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm13\n"
    "vmovaps 64(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm6, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm7, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm8, %%ymm12, %%ymm13\n"
    "vmovaps 96(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm9, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm10, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm11, %%ymm12, %%ymm13\n"
    // block 1
    "vbroadcastss 1(%[src]), %%ymm15\n"
    "vbroadcastss 33(%[src]), %%ymm14\n"
    "vbroadcastss 65(%[src]), %%ymm13\n"
    "vmovaps 128(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm13\n"
    "vmovaps 160(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm13\n"
    "vmovaps 192(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm6, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm7, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm8, %%ymm12, %%ymm13\n"
    "vmovaps 224(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm9, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm10, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm11, %%ymm12, %%ymm13\n"
    // block 2
    "vbroadcastss 2(%[src]), %%ymm15\n"
    "vbroadcastss 34(%[src]), %%ymm14\n"
    "vbroadcastss 66(%[src]), %%ymm13\n"
    "vmovaps 256(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm13\n"
    "vmovaps 288(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm13\n"
    "vmovaps 320(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm6, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm7, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm8, %%ymm12, %%ymm13\n"
    "vmovaps 352(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm9, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm10, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm11, %%ymm12, %%ymm13\n"
    // block 3
    "vbroadcastss 3(%[src]), %%ymm15\n"
    "vbroadcastss 35(%[src]), %%ymm14\n"
    "vbroadcastss 67(%[src]), %%ymm13\n"
    "vmovaps 384(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm13\n"
    "vmovaps 416(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm13\n"
    "vmovaps 448(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm6, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm7, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm8, %%ymm12, %%ymm13\n"
    "vmovaps 480(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm9, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm10, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm11, %%ymm12, %%ymm13\n"
    // block 4
    "vbroadcastss 4(%[src]), %%ymm15\n"
    "vbroadcastss 36(%[src]), %%ymm14\n"
    "vbroadcastss 68(%[src]), %%ymm13\n"
    "vmovaps 512(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm13\n"
    "vmovaps 544(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm13\n"
    "vmovaps 576(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm6, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm7, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm8, %%ymm12, %%ymm13\n"
    "vmovaps 608(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm9, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm10, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm11, %%ymm12, %%ymm13\n"
    // block 5
    "vbroadcastss 5(%[src]), %%ymm15\n"
    "vbroadcastss 37(%[src]), %%ymm14\n"
    "vbroadcastss 69(%[src]), %%ymm13\n"
    "vmovaps 640(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm13\n"
    "vmovaps 672(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm13\n"
    "vmovaps 704(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm6, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm7, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm8, %%ymm12, %%ymm13\n"
    "vmovaps 736(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm9, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm10, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm11, %%ymm12, %%ymm13\n"
    // block 6
    "vbroadcastss 6(%[src]), %%ymm15\n"
    "vbroadcastss 38(%[src]), %%ymm14\n"
    "vbroadcastss 70(%[src]), %%ymm13\n"
    "vmovaps 768(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm13\n"
    "vmovaps 800(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm13\n"
    "vmovaps 832(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm6, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm7, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm8, %%ymm12, %%ymm13\n"
    "vmovaps 864(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm9, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm10, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm11, %%ymm12, %%ymm13\n"
    // block 7
    "vbroadcastss 7(%[src]), %%ymm15\n"
    "vbroadcastss 39(%[src]), %%ymm14\n"
    "vbroadcastss 71(%[src]), %%ymm13\n"
    "vmovaps 896(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm0, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm1, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm2, %%ymm12, %%ymm13\n"
    "vmovaps 928(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm3, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm4, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm5, %%ymm12, %%ymm13\n"
    "vmovaps 960(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm6, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm7, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm8, %%ymm12, %%ymm13\n"
    "vmovaps 992(%[weight]), %%ymm12\n"
    "vfmadd231ps %%ymm9, %%ymm12, %%ymm15\n"
    "vfmadd231ps %%ymm10, %%ymm12, %%ymm14\n"
    "vfmadd231ps %%ymm11, %%ymm12, %%ymm13\n"
    "dec %[deep]\n"
    "add 1024, %[weight]\n"
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
    "vmaxps %%ymm7, %%ymm15, %%ymm7\n"
    "vmaxps %%ymm8, %%ymm15, %%ymm8\n"
    "vmaxps %%ymm9, %%ymm15, %%ymm9\n"
    "vmaxps %%ymm10, %%ymm15, %%ymm10\n"
    "vmaxps %%ymm11, %%ymm15, %%ymm11\n"
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
    "vminps %%ymm7, %%ymm14, %%ymm7\n"
    "vminps %%ymm8, %%ymm14, %%ymm8\n"
    "vminps %%ymm9, %%ymm14, %%ymm9\n"
    "vminps %%ymm10, %%ymm14, %%ymm10\n"
    "vminps %%ymm11, %%ymm14, %%ymm11\n"
    "3:\n"
    "vmovups %%ymm0, 0(%[dst])\n"
    "vmovups %%ymm1, 32(%[dst])\n"
    "vmovups %%ymm2, 64(%[dst])\n"
    "vmovups %%ymm3, 0(%[dst], %[dst_stride], 1)\n"
    "vmovups %%ymm4, 32(%[dst], %[dst_stride], 1)\n"
    "vmovups %%ymm5, 64(%[dst], %[dst_stride], 1)\n"
    "vmovups %%ymm6, 0(%[dst], %[dst_stride], 2)\n"
    "vmovups %%ymm7, 32(%[dst], %[dst_stride], 2)\n"
    "vmovups %%ymm8, 64(%[dst], %[dst_stride], 2)\n"
    "vmovups %%ymm9, 0(%[dst_4])\n"
    "vmovups %%ymm10, 32(%[dst_4])\n"
    "vmovups %%ymm11, 64(%[dst_4])\n"
    :
    : [ src ] "r"(src), [ src_stride ] "r"(src_stride_t), [ weight ] "r"(weight), [ deep ] "r"(deep_t),
      [ inc_flag ] "r"(inc_flag), [ act_flag ] "r"(act_flag), [ dst ] "r"(dst), [ dst_stride ] "r"(dst_stride_t),
      [ dst_4 ] "r"(dst_4)
    : "%rax", "%ymm0", "%ymm1", "%ymm2", "%ymm3", "%ymm4", "%ymm5", "%ymm6", "%ymm7", "%ymm8", "%ymm9", "%ymm10",
      "%ymm11", "%ymm12", "%ymm13", "%ymm14", "%ymm15");
}
