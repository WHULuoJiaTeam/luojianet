/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "nnacl/fp16/deconv_winograd_fp16.h"
#include "nnacl/base/minimal_filtering_generator.h"

void DeConvWgInputPackFp16(const float16_t *src_ptr, float16_t *dst_ptr, int channel, int stride) {
  int ic4div = channel / C4NUM;
  int ic4mod = channel % C4NUM;
  const float16_t *src = src_ptr;
  float16_t *dst = dst_ptr;

  for (int ic = 0; ic < ic4div; ic++) {
    vst1_f16(dst, vld1_f16(src));
    dst += stride;
    src += C4NUM;
  }

  if (ic4mod != 0) {
    int ic_res = 0;
    for (; ic_res < ic4mod; ic_res++) {
      dst[ic_res] = src[ic_res];
    }
    for (; ic_res < C4NUM; ic_res++) {
      dst[ic_res] = 0;
    }
  }
  return;
}

#ifdef ENABLE_ARM82_A32
void DeconvWgMergeFp16A32Fun(const float16_t *src_ptr, float16_t *dst_ptr, size_t src_step, size_t dst_step) {
  asm volatile(
    "mov r0, %[src_ptr]\n"
    "mov r1, %[dst_ptr]\n"
    "mov r2, r1\n"

    "vld1.16 {d0}, [r0], %[src_step]\n"
    "vld1.16 {d2}, [r1], %[dst_step]\n"
    "vld1.16 {d4}, [r0], %[src_step]\n"
    "vld1.16 {d6}, [r1], %[dst_step]\n"
    "vadd.f16 d0, d0, d2\n"
    "vld1.16 {d8}, [r0], %[src_step]\n"
    "vadd.f16 d4, d4, d6\n"
    "vst1.16 {d0}, [r2], %[dst_step]\n"
    "vst1.16 {d4}, [r2], %[dst_step]\n"

    "vld1.16 {d10}, [r1], %[dst_step]\n"
    "vld1.16 {d12}, [r0], %[src_step]\n"
    "vadd.f16 d8, d8, d10\n"
    "vld1.16 {d14}, [r1], %[dst_step]\n"
    "vadd.f16 d12, d12, d14\n"
    "vld1.16 {d0}, [r0], %[src_step]\n"
    "vst1.16 {d8}, [r2], %[dst_step]\n"
    "vst1.16 {d12}, [r2], %[dst_step]\n"

    "vld1.16 {d2}, [r1], %[dst_step]\n"
    "vld1.16 {d4}, [r0], %[src_step]\n"
    "vld1.16 {d6}, [r1], %[dst_step]\n"
    "vadd.f16 d0, d0, d2\n"
    "vadd.f16 d4, d4, d6\n"
    "vst1.16 {d0}, [r2], %[dst_step]\n"
    "vst1.16 {d4}, [r2], %[dst_step]\n"

    "vld1.16 {d8}, [r0], %[src_step]\n"
    "vld1.16 {d10}, [r1], %[dst_step]\n"
    "vld1.16 {d12}, [r0], %[src_step]\n"
    "vld1.16 {d14}, [r1], %[dst_step]\n"
    "vadd.f16 d8, d8, d10\n"
    "vadd.f16 d12, d12, d14\n"
    "vst1.16 {d8}, [r2], %[dst_step]\n"
    "vst1.16 {d12}, [r2], %[dst_step]\n"

    :
    : [ src_ptr ] "r"(src_ptr), [ dst_ptr ] "r"(dst_ptr), [ src_step ] "r"(src_step), [ dst_step ] "r"(dst_step)
    : "r0", "r1", "r2", "q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7");
}
#endif

void DeConvWgMergeFp16(const float16_t *src, float16_t *dst, size_t src_stride, size_t dst_stride, size_t count) {
  const float16_t *src_ptr = src;
  float16_t *dst_ptr = dst;
  size_t cuont8 = count / C8NUM * C8NUM;
  int i = 0;
  for (; i < cuont8; i += C8NUM) {
#ifdef ENABLE_ARM64
    size_t src_step = src_stride * sizeof(float16_t);
    size_t dst_step = dst_stride * sizeof(float16_t);
    asm volatile(
      "mov x7, %[src_ptr]\n"
      "mov x8, %[dst_ptr]\n"
      "mov x10, x8\n"

      "ld1 {v0.4h}, [x7], %[src_step]\n"
      "ld1 {v1.4h}, [x8], %[dst_step]\n"
      "ld1 {v2.4h}, [x7], %[src_step]\n"
      "ld1 {v3.4h}, [x8], %[dst_step]\n"
      "fadd v0.4h, v0.4h, v1.4h\n"
      "ld1 {v4.4h}, [x7], %[src_step]\n"
      "fadd v2.4h, v2.4h, v3.4h\n"
      "st1 {v0.4h}, [x10], %[dst_step]\n"
      "st1 {v2.4h}, [x10], %[dst_step]\n"

      "ld1 {v5.4h}, [x8], %[dst_step]\n"
      "ld1 {v6.4h}, [x7], %[src_step]\n"
      "fadd v4.4h, v4.4h, v5.4h\n"
      "ld1 {v7.4h}, [x8], %[dst_step]\n"
      "fadd v6.4h, v6.4h, v7.4h\n"
      "ld1 {v0.4h}, [x7], %[src_step]\n"
      "st1 {v4.4h}, [x10], %[dst_step]\n"
      "st1 {v6.4h}, [x10], %[dst_step]\n"

      "ld1 {v1.4h}, [x8], %[dst_step]\n"
      "ld1 {v2.4h}, [x7], %[src_step]\n"
      "ld1 {v3.4h}, [x8], %[dst_step]\n"
      "fadd v0.4h, v0.4h, v1.4h\n"
      "fadd v2.4h, v2.4h, v3.4h\n"
      "st1 {v0.4h}, [x10], %[dst_step]\n"
      "st1 {v2.4h}, [x10], %[dst_step]\n"

      "ld1 {v4.4h}, [x7], %[src_step]\n"
      "ld1 {v5.4h}, [x8], %[dst_step]\n"
      "ld1 {v6.4h}, [x7], %[src_step]\n"
      "ld1 {v7.4h}, [x8], %[dst_step]\n"
      "fadd v4.4h, v4.4h, v5.4h\n"
      "fadd v6.4h, v6.4h, v7.4h\n"
      "st1 {v4.4h}, [x10], %[dst_step]\n"
      "st1 {v6.4h}, [x10], %[dst_step]\n"

      :
      : [ src_ptr ] "r"(src_ptr), [ dst_ptr ] "r"(dst_ptr), [ src_step ] "r"(src_step), [ dst_step ] "r"(dst_step)
      : "x7", "x8", "x10", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7");
#elif defined(ENABLE_ARM82_A32)
    size_t src_step = src_stride * sizeof(float16_t);
    size_t dst_step = dst_stride * sizeof(float16_t);
    DeconvWgMergeFp16A32Fun(src_ptr, dst_ptr, src_step, dst_step);
#else
    for (int j = 0; j < C8NUM; j++) {
      const float16_t *s = src_ptr + j * src_stride;
      float16_t *d = dst_ptr + j * dst_stride;
      for (int k = 0; k < C4NUM; k++) {
        d[k] += s[k];
      }
    }
#endif
    src_ptr += C8NUM * src_stride;
    dst_ptr += C8NUM * dst_stride;
  }

  for (; i < count; i++) {
    float16x4_t src_data = vld1_f16(src_ptr);
    float16x4_t dst_data = vld1_f16(dst_ptr);
    dst_data = vadd_f16(src_data, dst_data);
    vst1_f16(dst_ptr, dst_data);

    src_ptr += src_stride;
    dst_ptr += dst_stride;
  }
  return;
}

void DeConvWgCalWgFp16(const float16_t *tile_in, float16_t *tile_out, const float16_t *weight_buf, float16_t *tmp_buf,
                       const float16_t *at_buf, float16_t *a_mid_buf, float16_t *trans_a_buf, bool *transferred,
                       const float16_t *bt_buf, float16_t *b_tmp_buf, int unit_size, int w_start, int h_start,
                       const ConvParameter *conv_param, const DeConvParam *deconv_param) {
  int winograd_plane = unit_size * unit_size;
  if (!transferred[unit_size]) {
    WinogradTransLeftFp16(tile_in, at_buf, a_mid_buf, DECONV_WINOGRAD_DEFAULT_UNIT, unit_size,
                          DECONV_WINOGRAD_DEFAULT_UNIT, deconv_param->ic_div_ * DECONV_WINOGRAD_DEFAULT_TILE);
    WinogradTransRightFp16(a_mid_buf, at_buf, trans_a_buf, unit_size, unit_size, DECONV_WINOGRAD_DEFAULT_UNIT,
                           deconv_param->ic_div_ * DECONV_WINOGRAD_DEFAULT_TILE);
    transferred[unit_size] = true;
  }

  for (int index = 0; index < winograd_plane; index++) {
    float16_t *src = trans_a_buf + index * DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->ic_up_;
    float16_t *dst = tmp_buf + index * deconv_param->oc_up_ * DECONV_WINOGRAD_DEFAULT_TILE;
    const float16_t *weight = weight_buf + index * deconv_param->ic_up_ * deconv_param->oc_up_;
    TiledC4MatmulFp16(dst, src, weight, DECONV_WINOGRAD_DEFAULT_TILE * C4NUM, deconv_param->ic_div_,
                      deconv_param->oc_div_);
  }

  WinogradTransLeftFp16(tmp_buf, bt_buf, b_tmp_buf, unit_size, unit_size, unit_size,
                        deconv_param->oc_div_ * DECONV_WINOGRAD_DEFAULT_TILE);
  WinogradTransRightFp16(b_tmp_buf, bt_buf, tmp_buf, unit_size, unit_size, unit_size,
                         deconv_param->oc_div_ * DECONV_WINOGRAD_DEFAULT_TILE);

  // Add to dest
  for (int uhi = 0; uhi < unit_size; uhi++) {
    int h_index = uhi * conv_param->stride_h_ + h_start;
    for (int uwi = 0; uwi < unit_size; uwi++) {
      int w_index = uwi * conv_param->stride_w_ + w_start;

      float16_t *dst = tile_out + w_index * DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->oc_up_ +
                       h_index * deconv_param->out_tile_w_ * DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->oc_up_;
      float16_t *src = tmp_buf + (uwi + uhi * unit_size) * DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->oc_up_;
      DeConvWgMergeFp16(src, dst, C4NUM, C4NUM, DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->oc_div_);
    }
  }
  return;
}

void DeConvWgCalCommFp16(const float16_t *tile_in, float16_t *tile_out, const float16_t *weight, float16_t *tmp_buf,
                         int h_start, int w_start, int h_size, int w_size, const ConvParameter *conv_param,
                         const DeConvParam *deconv_param) {
  int count = deconv_param->oc_div_ * w_size * h_size;
  int in_stride = DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->ic_up_;
  int out_stride = DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->oc_up_;

  for (int hi = 0; hi < DECONV_WINOGRAD_DEFAULT_UNIT; hi++) {
    for (int wi = 0; wi < DECONV_WINOGRAD_DEFAULT_UNIT; wi++) {
      const float16_t *src_in = tile_in + (wi + hi * DECONV_WINOGRAD_DEFAULT_UNIT) * in_stride;
      TiledC4MatmulFp16(tmp_buf, src_in, weight, DECONV_WINOGRAD_DEFAULT_TILE * C4NUM, deconv_param->ic_div_, count);

      for (int uhi = 0; uhi < h_size; uhi++) {
        for (int uwi = 0; uwi < w_size; uwi++) {
          int w_index = (wi + uwi) * conv_param->stride_w_ + w_start;
          int h_index = (hi + uhi) * conv_param->stride_h_ + h_start;
          float16_t *dst = tile_out + h_index * out_stride * deconv_param->out_tile_w_ + w_index * out_stride;
          float16_t *src = tmp_buf + (uwi + uhi * w_size) * out_stride;
          DeConvWgMergeFp16(src, dst, C4NUM, C4NUM, DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->oc_div_);
        }
      }
    }
  }
  return;
}

int PackDeConvWgDataFp16(const float16_t *nhwc_weight, DeConvComputeUnit *unit, const ConvParameter *conv_param,
                         const DeConvParam *deconv_param) {
  int tmp_kernel_plane = unit->w_size_ * unit->h_size_;
  int output_channel = conv_param->output_channel_;
  int size = conv_param->input_channel_ * output_channel * tmp_kernel_plane;
  float16_t *current_unit_weight = (float16_t *)malloc(size * sizeof(float16_t));
  if (current_unit_weight == NULL) {
    return NNACL_NULL_PTR;
  }
  for (int ic = 0; ic < conv_param->input_channel_; ic++) {
    const float16_t *src_ic = nhwc_weight + deconv_param->kernel_plane_ * output_channel * ic;
    float16_t *dst_ic = current_unit_weight + tmp_kernel_plane * output_channel * ic;
    for (int uhi = 0; uhi < unit->h_size_; uhi++) {
      for (int uwi = 0; uwi < unit->w_size_; uwi++) {
        int src_h_offset = unit->h_start_ + uhi * conv_param->stride_h_;
        int src_w_offset = unit->w_start_ + uwi * conv_param->stride_w_;
        const float16_t *src_hw = src_ic + (src_h_offset * conv_param->kernel_w_ + src_w_offset) * output_channel;
        float16_t *dst_hw = dst_ic + (uhi * unit->w_size_ + uwi) * output_channel;
        memcpy(dst_hw, src_hw, output_channel * sizeof(float16_t));
      }
    }
  }

  if (unit->use_winograd_) {
    /* Generate winograd  */
    float matrix_g[64];
    float matrix_gt[64];
    float matrix_a[64];
    float matrix_at[64];
    float matrix_b[64];
    float matrix_bt[64];
    int ret = CookToomFilter(matrix_a, matrix_at, matrix_b, matrix_bt, matrix_g, matrix_gt, 0.5f,
                             DECONV_WINOGRAD_DEFAULT_UNIT, unit->h_size_);
    if (ret != NNACL_OK) {
      free(current_unit_weight);
      current_unit_weight = NULL;
      return NNACL_ERRCODE_WINOGRAD_GENERATOR_ERROR;
    }

    /* winograd AT */
    unit->winograd_.AT_ = malloc(unit->winograd_.i_ * unit->winograd_.o_ * sizeof(float16_t));
    if (unit->winograd_.AT_ == NULL) {
      free(current_unit_weight);
      current_unit_weight = NULL;
      return NNACL_NULL_PTR;
    }
    Float32ToFloat16(matrix_at, unit->winograd_.AT_, unit->winograd_.i_ * unit->winograd_.o_);

    /* winograd BT */
    unit->winograd_.BT_ = malloc(unit->winograd_.o_ * unit->winograd_.o_ * sizeof(float16_t));
    if (unit->winograd_.BT_ == NULL) {
      free(current_unit_weight);
      free(unit->winograd_.AT_);
      current_unit_weight = NULL;
      unit->winograd_.AT_ = NULL;
      return NNACL_NULL_PTR;
    }
    Float32ToFloat16(matrix_bt, unit->winograd_.BT_, unit->winograd_.o_ * unit->winograd_.o_);

    /* winograd Weight */
    size = conv_param->input_channel_ * output_channel * unit->winograd_.kh_ * unit->winograd_.kw_;
    float16_t *winograd_unit_weight = (float16_t *)malloc(size * sizeof(float16_t));
    if (winograd_unit_weight == NULL) {
      free(current_unit_weight);
      free(unit->winograd_.AT_);
      free(unit->winograd_.BT_);
      current_unit_weight = NULL;
      unit->winograd_.AT_ = NULL;
      unit->winograd_.BT_ = NULL;
      return NNACL_NULL_PTR;
    }

    WinogradWeightTransformFp16(current_unit_weight, winograd_unit_weight, matrix_g, matrix_gt, C4NUM,
                                unit->winograd_.kh_, unit->h_size_, output_channel, conv_param->input_channel_, false);

    /* reset weight data & info */
    tmp_kernel_plane = unit->winograd_.kh_ * unit->winograd_.kw_;
    free(current_unit_weight);
    current_unit_weight = winograd_unit_weight;
    winograd_unit_weight = NULL;
  }

  /* trans mhwc -> hw1:k1-knc0-c4:k1-knc5-c8:hw2:k1-knc0-c4:k1 */
  float16_t *dst_weight = (float16_t *)unit->weight_;
  size = deconv_param->ic_up_ * deconv_param->oc_up_ * tmp_kernel_plane;
  memset(dst_weight, 0, size * sizeof(float16_t));
  for (int ic = 0; ic < conv_param->input_channel_; ic++) {
    for (int oc = 0; oc < output_channel; oc++) {
      int oc4div = oc / C4NUM, oc4mod = oc % C4NUM;
      for (int upi = 0; upi < tmp_kernel_plane; upi++) {
        int src_index = ic * output_channel * tmp_kernel_plane + upi * output_channel + oc;
        int dst_index = upi * deconv_param->oc_up_ * deconv_param->ic_up_ + oc4div * C4NUM * deconv_param->ic_up_ +
                        ic * C4NUM + oc4mod;
        dst_weight[dst_index] = current_unit_weight[src_index];
      }
    }
  }

  free(current_unit_weight);
  current_unit_weight = NULL;
  return NNACL_OK;
}

void DeconvWgFp16(const float16_t *nhwc_input_, float16_t *tile_in, float16_t *tile_out, int start_index,
                  int calculate_count, const ConvParameter *conv_param, DeConvParam *deconv_param, int task_id) {
  NNACL_CHECK_ZERO_RETURN(deconv_param->in_tile_w_count_);
  /* pack tile input */
  int tile_in_unit_stride = deconv_param->ic_up_ * DECONV_WINOGRAD_DEFAULT_TILE;
  float16x4_t zero = vdup_n_f16(0.0f);

  for (int unit_index = 0; unit_index < calculate_count; unit_index++) {
    int plane_index = start_index + unit_index;
    int w_unit_index = plane_index % deconv_param->in_tile_w_count_;
    int h_unit_index = plane_index / deconv_param->in_tile_w_count_;
    int w_start = w_unit_index * DECONV_WINOGRAD_DEFAULT_UNIT;
    int h_start = h_unit_index * DECONV_WINOGRAD_DEFAULT_UNIT;

    float16_t *dst_unit = tile_in + unit_index * C4NUM;
    for (int hi = 0; hi < DECONV_WINOGRAD_DEFAULT_UNIT; hi++) {
      for (int wi = 0; wi < DECONV_WINOGRAD_DEFAULT_UNIT; wi++) {
        float16_t *dst = dst_unit + (wi + hi * DECONV_WINOGRAD_DEFAULT_UNIT) * tile_in_unit_stride;
        int w_index = w_start + wi;
        int h_index = h_start + hi;
        if (w_index >= conv_param->input_w_ || h_index >= conv_param->input_h_) {
          for (int ic4_index = 0; ic4_index < deconv_param->ic_div_; ic4_index++) {
            vst1_f16(dst + ic4_index * DECONV_WINOGRAD_DEFAULT_TILE * C4NUM, zero);
          }
          continue;
        }

        const float16_t *src = nhwc_input_ + (w_index + h_index * conv_param->input_w_) * conv_param->input_channel_;
        DeConvWgInputPackFp16(src, dst, conv_param->input_channel_, DECONV_WINOGRAD_DEFAULT_TILE * C4NUM);
      }
    }
  }

  /* compute */
  bool transferred[DECONV_WINOGRAD_BUFFER_COUNT] = {false};
  for (int i = 0; i < deconv_param->compute_size_; i++) {
    DeConvComputeUnit *unit = &deconv_param->compute_units_[i];
    if (unit->use_winograd_) {
      float16_t *tmp_buf = (float16_t *)unit->tmp_buffer_ + task_id * unit->winograd_.kh_ * unit->winograd_.kw_ *
                                                              deconv_param->oc_up_ * DECONV_WINOGRAD_DEFAULT_TILE;

      /* winograd a buffer */
      if (unit->winograd_.kh_ >= DECONV_WINOGRAD_BUFFER_COUNT) {
        return;
      }
      DeConvWgABuffer *tmp_a = &deconv_param->a_buffer_[unit->winograd_.kh_];
      float16_t *mid_a = (float16_t *)tmp_a->middle_buffer_ + task_id * unit->winograd_.kw_ * unit->winograd_.kh_ *
                                                                DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->ic_up_;
      float16_t *dst_a = (float16_t *)tmp_a->dest_buffer_ + task_id * unit->winograd_.kw_ * unit->winograd_.kh_ *
                                                              DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->ic_up_;
      float16_t *tmp_b = (float16_t *)unit->winograd_.b_buffer_ + task_id * unit->winograd_.kh_ * unit->winograd_.kw_ *
                                                                    DECONV_WINOGRAD_DEFAULT_TILE * deconv_param->oc_up_;
      DeConvWgCalWgFp16(tile_in, tile_out, (float16_t *)unit->weight_, tmp_buf, unit->winograd_.AT_, mid_a, dst_a,
                        transferred, unit->winograd_.BT_, tmp_b, unit->winograd_.kh_, unit->w_start_, unit->h_start_,
                        conv_param, deconv_param);
    } else {
      float16_t *tmp_buf = (float16_t *)unit->tmp_buffer_ + task_id * deconv_param->oc_div_ * unit->w_size_ *
                                                              unit->h_size_ * DECONV_WINOGRAD_DEFAULT_TILE * C4NUM;
      DeConvWgCalCommFp16(tile_in, tile_out, (float16_t *)unit->weight_, tmp_buf, unit->h_start_, unit->w_start_,
                          unit->h_size_, unit->w_size_, conv_param, deconv_param);
    }
  }
  return;
}

void DeconvWgPostFp16(const float16_t *tile_out, float16_t *nc4hw4_output, const ConvParameter *conv_param,
                      const DeConvParam *deconv_param, int calculate_count, int tile_index) {
  NNACL_CHECK_ZERO_RETURN(deconv_param->in_tile_w_count_);
  /* merge */
  int src_unit_stride = deconv_param->oc_up_ * DECONV_WINOGRAD_DEFAULT_TILE;

  int src_stride = DECONV_WINOGRAD_DEFAULT_TILE * C4NUM;
  int dst_stride = conv_param->output_w_ * conv_param->output_h_ * C4NUM;

  for (int index = 0; index < calculate_count; ++index) {
    const float16_t *src_start = tile_out + index * C4NUM;

    int plane_index = tile_index * DECONV_WINOGRAD_DEFAULT_TILE + index;
    int w_unit_index = plane_index % deconv_param->in_tile_w_count_;
    int h_unit_index = plane_index / deconv_param->in_tile_w_count_;
    int w_start = w_unit_index * DECONV_WINOGRAD_DEFAULT_UNIT * conv_param->stride_w_ - conv_param->pad_l_;
    int h_start = h_unit_index * DECONV_WINOGRAD_DEFAULT_UNIT * conv_param->stride_h_ - conv_param->pad_u_;
    float16_t *dst_start = nc4hw4_output + h_start * conv_param->output_w_ * C4NUM + w_start * C4NUM;

    int merge_w_start = MSMAX(-w_start, 0);
    int merge_h_start = MSMAX(-h_start, 0);
    int merge_h_end = MSMIN(deconv_param->out_tile_h_, conv_param->output_h_ - h_start);
    int merge_w_end = MSMIN(deconv_param->out_tile_w_, conv_param->output_w_ - w_start);

    for (int hi = merge_h_start; hi < merge_h_end; hi++) {
      for (int wi = merge_w_start; wi < merge_w_end; wi++) {
        const float16_t *src = src_start + (hi * deconv_param->out_tile_w_ + wi) * src_unit_stride;
        float16_t *dst = dst_start + (hi * conv_param->output_w_ + wi) * C4NUM;
        DeConvWgMergeFp16(src, dst, src_stride, dst_stride, deconv_param->oc_div_);
      }
    }
  }
  return;
}

#ifndef ENABLE_ARM
void WinogradTransLeftFp16(const float16_t *S, const float16_t *B, float16_t *M, size_t w, size_t h, size_t k,
                           size_t length) {
  const int unitStep = C4NUM * length;
  for (int y = 0; y < h; ++y) {
    float16_t *dstY = M + y * w * unitStep;
    for (int x = 0; x < w; ++x) {
      float16_t *dstX = dstY + x * unitStep;
      const float16_t *srcX = S + x * unitStep;
      memset(dstX, 0, unitStep * sizeof(float16_t));
      for (int i = 0; i < k; ++i) {
        float16_t b = B[i * h + y];
        const float16_t *srcY = srcX + i * w * unitStep;
        if (0.0f == b) {
          continue;
        }
        for (int j = 0; j < unitStep; ++j) {
          dstX[j] += srcY[j] * b;
        }
      }
    }
  }
}

void WinogradTransRightFp16(const float16_t *S, const float16_t *B, float16_t *M, size_t w, size_t h, size_t k,
                            size_t length) {
  const int unitStep = C4NUM * length;
  for (int y = 0; y < h; ++y) {
    float16_t *dstY = M + y * w * unitStep;
    const float16_t *srcY = S + y * k * unitStep;

    for (int x = 0; x < w; ++x) {
      float16_t *dstX = dstY + x * unitStep;
      memset(dstX, 0, unitStep * sizeof(float16_t));
      for (int i = 0; i < k; ++i) {
        const float16_t *srcX = srcY + i * unitStep;
        float16_t b = B[i * h + x];
        if (0.0f == b) {
          continue;
        }
        for (int j = 0; j < unitStep; ++j) {
          dstX[j] += srcX[j] * b;
        }
      }
    }
  }
}

void TiledC4MatmulFp16(float16_t *dst, const float16_t *src, const float16_t *weight, size_t cal_num, size_t ic4,
                       size_t oc4) {
  int dx, sz, dz;
  int src_depth_step = C4NUM * DECONV_WINOGRAD_DEFAULT_TILE;
  for (dz = 0; dz < oc4; ++dz) {
    float16_t *dst_z = dst + dz * cal_num;
    const float16_t *weight_dz = weight + dz * ic4 * 16;
    for (dx = 0; dx < DECONV_WINOGRAD_DEFAULT_TILE; ++dx) {
      float16_t *dst_x = dst_z + dx * C4NUM;
      dst_x[0] = 0.0f;
      dst_x[1] = 0.0f;
      dst_x[2] = 0.0f;
      dst_x[3] = 0.0f;
      const float16_t *src_dx = src + C4NUM * dx;
      for (sz = 0; sz < ic4; ++sz) {
        const float16_t *src_z = src_dx + sz * src_depth_step;
        const float16_t *weight_z = weight_dz + sz * 16;
        for (int i = 0; i < C4NUM; ++i) {
          for (int j = 0; j < C4NUM; ++j) {
            dst_x[j] += src_z[i] * weight_z[C4NUM * i + j];
          }
        }
      }
    }
  }
}
#endif
