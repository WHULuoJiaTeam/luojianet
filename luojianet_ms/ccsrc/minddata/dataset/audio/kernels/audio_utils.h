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
#ifndef LUOJIANET_MS_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_AUDIO_UTILS_H_
#define LUOJIANET_MS_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_AUDIO_UTILS_H_

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/util/validators.h"

constexpr double PI = 3.141592653589793;
constexpr int kMinAudioDim = 1;
constexpr int kDefaultAudioDim = 2;
constexpr int TWO = 2;

namespace luojianet_ms {
namespace dataset {
/// \brief Turn a tensor from the power/amplitude scale to the decibel scale.
/// \param input/output: Tensor of shape <..., freq, time>.
/// \param multiplier: power - 10, amplitude - 20.
/// \param amin: lower bound.
/// \param db_multiplier: multiplier for decibels.
/// \param top_db: the lower bound for decibels cut-off.
/// \return Status code.
template <typename T>
Status AmplitudeToDB(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, T multiplier, T amin,
                     T db_multiplier, T top_db) {
  TensorShape input_shape = input->shape();
  TensorShape to_shape = input_shape.Rank() == 2
                           ? TensorShape({1, 1, input_shape[-2], input_shape[-1]})
                           : TensorShape({input->Size() / (input_shape[-3] * input_shape[-2] * input_shape[-1]),
                                          input_shape[-3], input_shape[-2], input_shape[-1]});
  RETURN_IF_NOT_OK(input->Reshape(to_shape));

  std::vector<T> max_val;
  uint64_t step = to_shape[-3] * input_shape[-2] * input_shape[-1];
  uint64_t cnt = 0;
  T temp_max = std::numeric_limits<T>::lowest();
  for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++) {
    // do clamp
    *itr = *itr < amin ? log10(amin) * multiplier : log10(*itr) * multiplier;
    *itr -= multiplier * db_multiplier;
    // calculate max by axis
    cnt++;
    if ((*itr) > temp_max) temp_max = *itr;
    if (cnt % step == 0) {
      max_val.push_back(temp_max);
      temp_max = std::numeric_limits<T>::lowest();
    }
  }

  if (!std::isnan(top_db)) {
    uint64_t ind = 0;
    for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++, ind++) {
      T lower_bound = max_val[ind / step] - top_db;
      *itr = std::max((*itr), lower_bound);
    }
  }
  RETURN_IF_NOT_OK(input->Reshape(input_shape));
  *output = input;
  return Status::OK();
}

/// \brief Calculate the angles of the complex numbers.
/// \param input/output: Tensor of shape <..., time>.
template <typename T>
Status Angle(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  TensorShape shape = input->shape();
  std::vector output_shape = shape.AsVector();
  output_shape.pop_back();
  std::shared_ptr<Tensor> output_tensor;
  std::vector<T> out;
  T o;
  T x;
  T y;
  for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++) {
    x = static_cast<T>(*itr);
    itr++;
    y = static_cast<T>(*itr);
    o = std::atan2(y, x);
    out.emplace_back(o);
  }
  // Generate multidimensional results corresponding to input
  Tensor::CreateFromVector(out, TensorShape{output_shape}, &output_tensor);
  *output = output_tensor;
  return Status::OK();
}

Status Bartlett(std::shared_ptr<Tensor> *output, int len);

/// \brief Perform a biquad filter of input tensor.
/// \param input/output: Tensor of shape <..., time>.
/// \param a0: denominator coefficient of current output y[n], typically 1.
/// \param a1: denominator coefficient of current output y[n-1].
/// \param a2: denominator coefficient of current output y[n-2].
/// \param b0: numerator coefficient of current input, x[n].
/// \param b1: numerator coefficient of input one time step ago x[n-1].
/// \param b2: numerator coefficient of input two time steps ago x[n-2].
/// \return Status code.
template <typename T>
Status Biquad(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, T b0, T b1, T b2, T a0, T a1,
              T a2) {
  std::vector<T> a_coeffs;
  std::vector<T> b_coeffs;
  a_coeffs.push_back(a0);
  a_coeffs.push_back(a1);
  a_coeffs.push_back(a2);
  b_coeffs.push_back(b0);
  b_coeffs.push_back(b1);
  b_coeffs.push_back(b2);
  return LFilter(input, output, a_coeffs, b_coeffs, true);
}

/// \brief Apply contrast effect.
/// \param input/output: Tensor of shape <..., time>.
/// \param enhancement_amount: controls the amount of the enhancement.
/// \return Status code.
template <typename T>
Status Contrast(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, T enhancement_amount) {
  const float enhancement_zoom = 750.0;
  T enhancement_amount_value = enhancement_amount / enhancement_zoom;
  TensorShape output_shape{input->shape()};
  std::shared_ptr<Tensor> out;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(output_shape, input->type(), &out));
  auto itr_out = out->begin<T>();
  for (auto itr_in = input->begin<T>(); itr_in != input->end<T>(); itr_in++) {
    // PI / 2 is half of the constant PI
    T temp1 = static_cast<T>(*itr_in) * (PI / TWO);
    T temp2 = enhancement_amount_value * std::sin(temp1 * 4);
    *itr_out = std::sin(temp1 + temp2);
    itr_out++;
  }
  *output = out;
  return Status::OK();
}

/// \brief Apply DBToAmplitude effect.
/// \param input/output: Tensor of shape <...,time>
/// \param ref: Reference which the output will be scaled by.
/// \param power: If power equals 1, will compute DB to power. If 0.5, will compute DB to amplitude.
/// \return Status code
template <typename T>
Status DBToAmplitude(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, T ref, T power) {
  std::shared_ptr<Tensor> out;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(input->shape(), input->type(), &out));
  auto itr_out = out->begin<T>();
  for (auto itr_in = input->begin<T>(); itr_in != input->end<T>(); itr_in++) {
    *itr_out = ref * pow(pow(10, (*itr_in) * 0.1), power);
    itr_out++;
  }
  *output = out;
  return Status::OK();
}

/// \brief Apply a DC shift to the audio.
/// \param input/output: Tensor of shape <...,time>.
/// \param shift: the amount to shift the audio.
/// \param limiter_gain: used only on peaks to prevent clipping.
/// \return Status code.
template <typename T>
Status DCShift(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float shift, float limiter_gain) {
  float limiter_threshold = 0.0;
  if (shift != limiter_gain && shift != 0) {
    limiter_threshold = 1.0 - (std::abs(shift) - limiter_gain);
    for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++) {
      if (*itr > limiter_threshold && shift > 0) {
        T peak = (*itr - limiter_threshold) * limiter_gain / (1 - limiter_threshold);
        T sample = (peak + limiter_threshold + shift);
        *itr = sample > limiter_threshold ? limiter_threshold : sample;
      } else if (*itr < -limiter_threshold && shift < 0) {
        T peak = (*itr + limiter_threshold) * limiter_gain / (1 - limiter_threshold);
        T sample = (peak + limiter_threshold + shift);
        *itr = sample < -limiter_threshold ? -limiter_threshold : sample;
      } else {
        T sample = (*itr + shift);
        *itr = (sample > 1 || sample < -1) ? (sample > 1 ? 1 : -1) : sample;
      }
    }
  } else {
    for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++) {
      T sample = (*itr + shift);
      *itr = sample > 1 || sample < -1 ? (sample > 1 ? 1 : -1) : sample;
    }
  }
  *output = input;
  return Status::OK();
}

/// \brief Apply amplification or attenuation to the whole waveform.
/// \param input/output: Tensor of shape <..., time>.
/// \param gain_db: Gain adjustment in decibels (dB).
/// \return Status code.
template <typename T>
Status Gain(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, T gain_db) {
  if (gain_db == 0) {
    *output = input;
    return Status::OK();
  }

  T radio = pow(10, gain_db / 20);
  for (auto itr = input->begin<T>(); itr != input->end<T>(); ++itr) {
    *itr = (*itr) * radio;
  }
  *output = input;
  return Status::OK();
}

/// \brief Perform an IIR filter by evaluating difference equation.
/// \param input/output: Tensor of shape <..., time>
/// \param a_coeffs: denominator coefficients of difference equation of dimension of (n_order + 1).
/// \param b_coeffs: numerator coefficients of difference equation of dimension of (n_order + 1).
/// \param clamp: If True, clamp the output signal to be in the range [-1, 1] (Default: True).
/// \return Status code
template <typename T>
Status LFilter(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, std::vector<T> a_coeffs,
               std::vector<T> b_coeffs, bool clamp) {
  //  pack batch
  TensorShape input_shape = input->shape();
  TensorShape toShape({input->Size() / input_shape[-1], input_shape[-1]});
  input->Reshape(toShape);
  auto shape_0 = static_cast<size_t>(input->shape()[0]);
  auto shape_1 = static_cast<size_t>(input->shape()[1]);
  std::vector<T> signal;
  std::shared_ptr<Tensor> out;
  std::vector<T> out_vect(shape_0 * shape_1);
  size_t x_idx = 0;
  size_t channel_idx = 1;
  size_t m_num_order = b_coeffs.size() - 1;
  size_t m_den_order = a_coeffs.size() - 1;
  CHECK_FAIL_RETURN_UNEXPECTED(a_coeffs[0] != static_cast<T>(0),
                               "Invalid data, the first value of 'a_coeffs' should not be 0, but got 0.");
  // init A_coeffs and B_coeffs by div(a0)
  for (size_t i = 1; i < a_coeffs.size(); i++) {
    a_coeffs[i] /= a_coeffs[0];
  }
  for (size_t i = 0; i < b_coeffs.size(); i++) {
    b_coeffs[i] /= a_coeffs[0];
  }
  // Sliding window
  T *m_px = new T[m_num_order + 1];
  T *m_py = new T[m_den_order + 1];

  // Tensor -> vector
  for (auto itr = input->begin<T>(); itr != input->end<T>();) {
    while (x_idx < shape_1 * channel_idx) {
      signal.push_back(*itr);
      itr++;
      x_idx++;
    }
    // Sliding window
    for (size_t j = 0; j < m_den_order; j++) {
      m_px[j] = static_cast<T>(0);
    }
    for (size_t j = 0; j <= m_den_order; j++) {
      m_py[j] = static_cast<T>(0);
    }
    // Each channel is processed with the sliding window
    for (size_t i = x_idx - shape_1; i < x_idx; i++) {
      m_px[m_num_order] = signal[i];
      for (size_t j = 0; j < m_num_order + 1; j++) {
        m_py[m_num_order] += b_coeffs[j] * m_px[m_num_order - j];
      }
      for (size_t j = 1; j < m_den_order + 1; j++) {
        m_py[m_num_order] -= a_coeffs[j] * m_py[m_num_order - j];
      }
      if (clamp) {
        if (m_py[m_num_order] > static_cast<T>(1))
          out_vect[i] = static_cast<T>(1);
        else if (m_py[m_num_order] < static_cast<T>(-1))
          out_vect[i] = static_cast<T>(-1);
        else
          out_vect[i] = m_py[m_num_order];
      } else {
        out_vect[i] = m_py[m_num_order];
      }
      if (i + 1 == x_idx) continue;
      for (size_t j = 0; j < m_num_order; j++) {
        m_px[j] = m_px[j + 1];
      }
      for (size_t j = 0; j < m_num_order; j++) {
        m_py[j] = m_py[j + 1];
      }
      m_py[m_num_order] = static_cast<T>(0);
    }
    if (x_idx % shape_1 == 0) {
      ++channel_idx;
    }
  }
  // unpack batch
  Tensor::CreateFromVector(out_vect, input_shape, &out);
  *output = out;
  delete[] m_px;
  delete[] m_py;
  return Status::OK();
}

/// \brief Generate linearly spaced vector.
/// \param[in] start Value of the startpoint.
/// \param[in] end Value of the endpoint.
/// \param[in] n N points in the output tensor.
/// \param[out] output Tensor has n points with linearly space.
///     The spacing between the points is (end - start) / (n - 1).
/// \return Status return code.
template <typename T>
Status Linspace(std::shared_ptr<Tensor> *output, T start, T end, int32_t n) {
  RETURN_IF_NOT_OK(ValidateNoGreaterThan("Linspace", "start", start, "end", end));
  int hundred = 100;
  n = std::isnan(n) ? hundred : n;
  CHECK_FAIL_RETURN_UNEXPECTED(n >= 0, "Linspace: input param n must be non-negative.");

  TensorShape out_shape({n});
  std::vector<T> linear_vect(n);
  T interval = (n == 1) ? 0 : ((end - start) / (n - 1));
  for (auto i = 0; i < linear_vect.size(); ++i) {
    linear_vect[i] = start + i * interval;
  }
  std::shared_ptr<Tensor> out_t;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(linear_vect, out_shape, &out_t));
  linear_vect.clear();
  linear_vect.shrink_to_fit();
  *output = out_t;
  return Status::OK();
}

/// \brief Convert normal STFT to STFT at the Mel scale.
/// \param input: Input audio tensor.
/// \param output: Mel scale audio tensor.
/// \param n_mels: Number of mel filter.
/// \param sample_rate: Sample rate of the signal.
/// \param f_min: Minimum frequency.
/// \param f_max: Maximum frequency.
/// \param n_stft: Number of bins in STFT.
/// \param norm: Enum, NormType::kSlaney or NormType::kNone. If norm is NormType::kSlaney, divide the triangle mel
///     weight by the width of the mel band.
/// \param mel_type: Type of calculate mel type, value should be MelType::kHtk or MelType::kSlaney.
/// \return Status code.
template <typename T>
Status MelScale(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t n_mels,
                int32_t sample_rate, T f_min, T f_max, int32_t n_stft, NormType norm, MelType mel_type) {
  // pack
  TensorShape input_shape = input->shape();
  TensorShape input_reshape({input->Size() / input_shape[-1] / input_shape[-2], input_shape[-2], input_shape[-1]});
  RETURN_IF_NOT_OK(input->Reshape(input_reshape));
  // gen freq bin mat
  std::shared_ptr<Tensor> freq_bin_mat;
  RETURN_IF_NOT_OK(CreateFbanks(&freq_bin_mat, n_stft, f_min, f_max, n_mels, sample_rate, norm, mel_type));
  auto data_ptr = &*freq_bin_mat->begin<T>();
  Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> matrix_fb(data_ptr, n_mels, n_stft);

  // input vector
  std::vector<T> in_vect(input->Size());
  size_t ind = 0;
  for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++, ind++) {
    in_vect[ind] = (*itr);
  }
  int rows = input_reshape[1];
  int cols = input_reshape[2];

  std::vector<T> mel_specgram;

  for (int c = 0; c < input_reshape[0]; c++) {
    std::vector<T> mat_c = std::vector<T>(in_vect.begin() + rows * cols * c, in_vect.begin() + rows * cols * (c + 1));
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> matrix_c(mat_c.data(), cols, rows);
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> mat_res = (matrix_c * matrix_fb.transpose());
    std::vector<T> vec_c(mat_res.data(), mat_res.data() + mat_res.size());
    mel_specgram.insert(mel_specgram.end(), vec_c.begin(), vec_c.end());
  }

  // unpack
  std::vector<int64_t> out_shape_vec = input_shape.AsVector();
  out_shape_vec[input_shape.Size() - 1] = cols;
  out_shape_vec[input_shape.Size() - TWO] = n_mels;
  TensorShape output_shape(out_shape_vec);
  std::shared_ptr<Tensor> out;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(mel_specgram, output_shape, &out));
  *output = out;
  return Status::OK();
}

/// \brief Transform audio signal into spectrogram.
/// \param[in] n_fft Size of FFT, creates n_fft / 2 + 1 bins.
/// \param[in] win_length Window size.
/// \param[in] hop_length Length of hop between STFT windows.
/// \param[in] pad Two sided padding of signal.
/// \param[in] window A function to create a window tensor
///     that is applied/multiplied to each frame/window.
/// \param[in] power Exponent for the magnitude spectrogram.
/// \param[in] normalized Whether to normalize by magnitude after stft.
/// \param[in] center Whether to pad waveform on both sides.
/// \param[in] pad_mode Controls the padding method used when center is true.
/// \param[in] onesided Controls whether to return half of results to avoid redundancy.
/// \return Status code.
Status Spectrogram(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int pad, WindowType window,
                   int n_fft, int hop_length, int win_length, float power, bool normalized, bool center,
                   BorderType pad_mode, bool onesided);

/// \brief Transform audio signal into spectrogram.
/// \param[in] input Tensor of shape <..., time>.
/// \param[out] output Tensor of shape <..., time>.
/// \param[in] sample_rate The sample rate of input tensor.
/// \param[in] n_fft Size of FFT, creates n_fft / 2 + 1 bins.
/// \param[in] win_length Window size.
/// \param[in] hop_length Length of hop between STFT windows.
/// \param[in] pad Two sided padding of signal.
/// \param[in] window A function to create a window tensor that is applied/multiplied to each frame/window.
/// \return Status code.
Status SpectralCentroid(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int sample_rate,
                        int n_fft, int win_length, int hop_length, int pad, WindowType window);

/// \brief Stretch STFT in time at a given rate, without changing the pitch.
/// \param input: Tensor of shape <..., freq, time>.
/// \param rate: Stretch factor.
/// \param phase_advance: Expected phase advance in each bin.
/// \param output: Tensor after stretch in time domain.
/// \return Status code.
Status TimeStretch(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float rate, float hop_length,
                   int32_t n_freq);

/// \brief Stretch STFT in time at a given rate, without changing the pitch.
/// \param[in] input Tensor of shape <..., freq, time, 2>.
/// \param[in] output Tensor of shape <..., freq, ceil(time/rate), 2>.
/// \param[in] rate Speed-up factor.
/// \param[in] phase_advance Expected phase advance in each bin in shape of (freq, 1).
/// \return Status code.
Status PhaseVocoder(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float rate,
                    const std::shared_ptr<Tensor> &phase_advance);

/// \brief Apply a mask along axis.
/// \param input: Tensor of shape <..., freq, time>.
/// \param output: Tensor of shape <..., freq, time>.
/// \param mask_param: Number of columns to be masked will be uniformly sampled from [0, mask_param].
/// \param mask_value: Value to assign to the masked columns.
/// \param axis: Axis to apply masking on (1 -> frequency, 2 -> time).
/// \param rnd: Number generator.
/// \return Status code.
Status RandomMaskAlongAxis(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t mask_param,
                           float mask_value, int axis, std::mt19937 rnd);

/// \brief Apply a mask along axis. All examples will have the same mask interval.
/// \param input: Tensor of shape <..., freq, time>.
/// \param output: Tensor of shape <..., freq, time>.
/// \param mask_width: The width of the mask.
/// \param mask_start: Starting position of the mask.
///     Mask will be applied from indices [mask_start, mask_start + mask_width).
/// \param mask_value: Value to assign to the masked columns.
/// \param axis: Axis to apply masking on (1 -> frequency, 2 -> time).
/// \return Status code.
Status MaskAlongAxis(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t mask_width,
                     int32_t mask_start, float mask_value, int32_t axis);

/// \brief Create a frequency transformation matrix with shape (n_freqs, n_mels).
/// \param output Tensor of the frequency transformation matrix.
/// \param n_freqs: Number of frequency.
/// \param f_min: Minimum of frequency in Hz.
/// \param f_max: Maximum of frequency in Hz.
/// \param n_mels: Number of mel filterbanks.
/// \param sample_rate: Sample rate.
/// \param norm: Norm to use, can be NormTyppe::kSlaney or NormTyppe::kNone.
/// \param mel_type: Scale to use, can be MelTyppe::kSlaney or MelTyppe::kHtk.
/// \return Status code.
Status CreateFbanks(std::shared_ptr<Tensor> *output, int32_t n_freqs, float f_min, float f_max, int32_t n_mels,
                    int32_t sample_rate, NormType norm, MelType mel_type);

/// \brief Create a DCT transformation matrix with shape (n_mels, n_mfcc), normalized depending on norm.
/// \param n_mfcc: Number of mfc coefficients to retain, the value must be greater than 0.
/// \param n_mels: Number of mel filterbanks, the value must be greater than 0.
/// \param norm: Norm to use, can be NormMode::kNone or NormMode::kOrtho.
/// \return Status code.
Status Dct(std::shared_ptr<Tensor> *output, int32_t n_mfcc, int32_t n_mels, NormMode norm);

/// \brief Compute the norm of complex tensor input.
/// \param power Power of the norm description (optional).
/// \param input Tensor shape of <..., complex=2>.
/// \param output Tensor shape of <..., >.
/// \return Status code.
Status ComplexNorm(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float power);

/// \brief Decode mu-law encoded signal.
/// \param input Tensor of shape <..., time>.
/// \param output Tensor of shape <..., time>.
/// \param quantization_channels Number of channels.
/// \return Status code.
Status MuLawDecoding(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                     int32_t quantization_channels);

/// \brief Encode signal based on mu-law companding.
/// \param input Tensor of shape <..., time>.
/// \param output Tensor of shape <..., time>.
/// \param quantization_channels Number of channels.
/// \return Status code.
Status MuLawEncoding(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                     int32_t quantization_channels);

/// \brief Apply a overdrive effect to the audio.
/// \param input Tensor of shape <..., time>.
/// \param output Tensor of shape <..., time>.
/// \param gain Coefficient of overload in dB.
/// \param color Coefficient of translation.
/// \return Status code.
template <typename T>
Status Overdrive(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, float gain, float color) {
  TensorShape input_shape = input->shape();
  // input->2D.
  auto rows = input->Size() / input_shape[-1];
  auto cols = input_shape[-1];
  TensorShape to_shape({rows, cols});
  RETURN_IF_NOT_OK(input->Reshape(to_shape));
  // apply dB2Linear on gain, 20dB is expect to gain.
  float gain_ex = exp(gain * log(10) / 20.0);
  color = color / 200;
  // declare the array used to store the input.
  std::vector<T> input_vec;
  // out_vec is used to save the result of applying overdrive.
  std::vector<T> out_vec;
  // store intermediate results of input.
  std::vector<T> temp;
  // scale and pan the input two-dimensional sound wave array to a certain extent.
  for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++) {
    // store the value of traverse the input.
    T temp_fp = *itr;
    input_vec.push_back(temp_fp);
    // use 0 to initialize out_vec.
    out_vec.push_back(0);
    T temp_fp2 = temp_fp * gain_ex + color;
    // 0.5 + 2/3 * 0.75 = 1, zoom and shift the sound.
    if (temp_fp2 < -1) {
      // -2.0 / 3.0 is -2/3 in the formula.
      temp.push_back(-2.0 / 3.0);
    } else if (temp_fp2 > 1) {
      // 2.0 / 3.0 is 2/3 in the formula.
      temp.push_back(2.0 / 3.0);
    } else {
      temp.push_back(temp_fp2 - temp_fp2 * temp_fp2 * temp_fp2 / 3.0);
    }
  }
  // last_in and last_out are the intermediate values for processing each moment.
  std::vector<T> last_in;
  std::vector<T> last_out;
  for (size_t i = 0; i < cols; i++) {
    last_in.push_back(0.0);
    last_out.push_back(0.0);
  }
  // overdrive core loop.
  for (size_t i = 0; i < cols; i++) {
    size_t index = 0;
    // calculate the value of each moment according to the rules of overdrive.
    for (size_t j = i; j < rows * cols; j += cols, index++) {
      // 0.995 is the preservation ratio of sound waves.
      last_out[index] = temp[j] - last_in[index] + last_out[index] * 0.995;
      last_in[index] = temp[j];
      // 0.5 + 2/3 * 0.75 = 1, zoom and shift the sound.
      T temp_fp = input_vec[j] * 0.5 + last_out[index] * 0.75;
      // clamp min=-1, max=1.
      if (temp_fp < -1) {
        out_vec[j] = -1.0;
      } else if (temp_fp > 1) {
        out_vec[j] = 1.0;
      } else {
        out_vec[j] = temp_fp;
      }
    }
  }
  // move data to output tensor.
  std::shared_ptr<Tensor> out;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(out_vec, input_shape, &out));
  *output = out;
  return Status::OK();
}

/// \brief Add a fade in and/or fade out to an input.
/// \param[in] input: The input tensor.
/// \param[out] output: Added fade in and/or fade out audio with the same shape.
/// \param[in] fade_in_len: Length of fade-in (time frames).
/// \param[in] fade_out_len: Length of fade-out (time frames).
/// \param[in] fade_shape: Shape of fade.
Status Fade(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t fade_in_len,
            int32_t fade_out_len, FadeShape fade_shape);

/// \brief Add a volume to an waveform.
/// \param input/output: Tensor of shape <..., time>.
/// \param gain: Gain value, varies according to the value of gain_type.
/// \param gain_type: Type of gain, should be one of [GainType::kAmplitude, GainType::kDb, GainType::kPower].
/// \return Status code.
template <typename T>
Status Vol(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, T gain, GainType gain_type) {
  const T lower_bound = -1;
  const T upper_bound = 1;

  // DB is a unit which converts a numeric value into decibel scale and for conversion, we have to use log10
  // A(in dB) = 20log10(A in amplitude)
  // When referring to measurements of power quantities, a ratio can be expressed as a level in decibels by evaluating
  // ten times the base-10 logarithm of the ratio of the measured quantity to reference value
  // A(in dB) = 10log10(A in power)
  const int power_factor_div = 20;
  const int power_factor_mul = 10;
  const int base = 10;

  if (gain_type == GainType::kDb) {
    if (gain != 0) {
      gain = std::pow(base, (gain / power_factor_div));
    }
  } else if (gain_type == GainType::kPower) {
    gain = power_factor_mul * std::log10(gain);
    gain = std::pow(base, (gain / power_factor_div));
  }

  for (auto itr = input->begin<T>(); itr != input->end<T>(); itr++) {
    if (gain != 0 || gain_type == GainType::kAmplitude) {
      *itr = (*itr) * gain;
    }
    *itr = std::min(std::max((*itr), lower_bound), upper_bound);
  }

  *output = input;

  return Status::OK();
}

/// \brief Separate a complex-valued spectrogram with shape (…, 2) into its magnitude and phase.
/// \param input: Complex tensor.
/// \param output: The magnitude and phase of the complex tensor.
/// \param power: Power of the norm.
Status Magphase(const TensorRow &input, TensorRow *output, float power);

/// \brief Compute Normalized Cross-Correlation Function (NCCF).
/// \param input: Tensor of shape <channel,waveform_length>.
/// \param output: Tensor of shape <channel, num_of_frames, lags>.
/// \param sample_rate: The sample rate of the waveform (Hz).
/// \param frame_time: Duration of a frame.
/// \param freq_low: Lowest frequency that can be detected (Hz).
/// \return Status code.
template <typename T>
Status ComputeNccf(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t sample_rate,
                   float frame_time, int32_t freq_low) {
  auto channel = input->shape()[0];
  auto waveform_length = input->shape()[1];
  size_t idx = 0;
  size_t channel_idx = 1;
  int32_t lags = static_cast<int32_t>(ceil(static_cast<float>(sample_rate) / freq_low));
  int32_t frame_size = static_cast<int32_t>(ceil(sample_rate * frame_time));
  int32_t num_of_frames = static_cast<int32_t>(ceil(static_cast<float>(waveform_length) / frame_size));
  int32_t p = lags + num_of_frames * frame_size - waveform_length;
  TensorShape output_shape({channel, num_of_frames, lags});
  DataType intput_type = input->type();
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(output_shape, intput_type, output));
  // pad p 0 in -1 dimension
  std::vector<T> signal;
  // Tensor -> vector
  for (auto itr = input->begin<T>(); itr != input->end<T>();) {
    while (idx < waveform_length * channel_idx) {
      signal.push_back(*itr);
      ++itr;
      ++idx;
    }
    // Each channel is processed with the sliding window
    // waveform：[channel, time] -->  waveform：[channel, time+p]
    for (size_t i = 0; i < p; ++i) {
      signal.push_back(static_cast<T>(0.0));
    }
    if (idx % waveform_length == 0) {
      ++channel_idx;
    }
  }
  // compute ncc
  for (dsize_t lag = 1; lag <= lags; ++lag) {
    // compute one ncc
    // one ncc out
    std::vector<T> out;
    channel_idx = 1;
    idx = 0;
    size_t win_idx = 0;
    size_t waveform_length_p = waveform_length + p;
    // Traversal signal
    for (auto itr = signal.begin(); itr != signal.end();) {
      // Each channel is processed with the sliding window
      size_t s1 = idx;
      size_t s2 = idx + lag;
      size_t frame_count = 0;
      T s1_norm = static_cast<T>(0);
      T s2_norm = static_cast<T>(0);
      T ncc_umerator = static_cast<T>(0);
      T ncc = static_cast<T>(0);
      while (idx < waveform_length_p * channel_idx) {
        // Sliding window
        if (frame_count == num_of_frames) {
          ++itr;
          ++idx;
          continue;
        }
        if (win_idx < frame_size) {
          ncc_umerator += signal[s1] * signal[s2];
          s1_norm += signal[s1] * signal[s1];
          s2_norm += signal[s2] * signal[s2];
          ++win_idx;
          ++s1;
          ++s2;
        }
        if (win_idx == frame_size) {
          if (s1_norm != static_cast<T>(0.0) && s2_norm != static_cast<T>(0.0)) {
            ncc = ncc_umerator / s1_norm / s2_norm;
          } else {
            ncc = static_cast<T>(0.0);
          }
          out.push_back(ncc);
          ncc_umerator = static_cast<T>(0.0);
          s1_norm = static_cast<T>(0.0);
          s2_norm = static_cast<T>(0.0);
          ++frame_count;
          win_idx = 0;
        }
        ++itr;
        ++idx;
      }
      if (idx % waveform_length_p == 0) {
        ++channel_idx;
      }
    }  // compute one ncc
    // cat tensor
    auto itr_out = out.begin();
    for (dsize_t row_idx = 0; row_idx < channel; ++row_idx) {
      for (dsize_t frame_idx = 0; frame_idx < num_of_frames; ++frame_idx) {
        RETURN_IF_NOT_OK((*output)->SetItemAt({row_idx, frame_idx, lag - 1}, *itr_out));
        ++itr_out;
      }
    }
  }  // compute ncc
  return Status::OK();
}

/// \brief For each frame, take the highest value of NCCF.
/// \param input: Tensor of shape <channel, num_of_frames, lags>.
/// \param output: Tensor of shape <channel, num_of_frames>.
/// \param sample_rate: The sample rate of the waveform (Hz).
/// \param freq_high: Highest frequency that can be detected (Hz).
/// \return Status code.
template <typename T>
Status FindMaxPerFrame(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t sample_rate,
                       int32_t freq_high) {
  std::vector<T> signal;
  std::vector<int> out;
  auto channel = input->shape()[0];
  auto num_of_frames = input->shape()[1];
  auto lags = input->shape()[2];
  CHECK_FAIL_RETURN_UNEXPECTED(freq_high != 0, "DetectPitchFrequency: freq_high can not be zero.");
  auto lag_min = static_cast<int32_t>(ceil(static_cast<float>(sample_rate) / freq_high));
  TensorShape out_shape({channel, num_of_frames});
  // pack batch
  for (auto itr = input->begin<T>(); itr != input->end<T>(); ++itr) {
    signal.push_back(*itr);
  }
  // find the best nccf
  T best_max_value = static_cast<T>(0.0);
  T half_max_value = static_cast<T>(0.0);
  int32_t best_max_indices = 0;
  int32_t half_max_indices = 0;
  auto thresh = static_cast<T>(0.99);
  auto lags_half = lags / 2;
  for (dsize_t channel_idx = 0; channel_idx < channel; ++channel_idx) {
    for (dsize_t frame_idx = 0; frame_idx < num_of_frames; ++frame_idx) {
      auto index_01 = channel_idx * num_of_frames * lags + frame_idx * lags + lag_min;
      best_max_value = signal[index_01];
      half_max_value = signal[index_01];
      best_max_indices = lag_min;
      half_max_indices = lag_min;
      for (dsize_t lag_idx = 0; lag_idx < lags; ++lag_idx) {
        if (lag_idx > lag_min) {
          auto index_02 = channel_idx * num_of_frames * lags + frame_idx * lags + lag_idx;
          if (signal[index_02] > best_max_value) {
            best_max_value = signal[index_02];
            best_max_indices = lag_idx;
            if (lag_idx < lags_half) {
              half_max_value = signal[index_02];
              half_max_indices = lag_idx;
            }
          }
        }
      }
      // Add back minimal lag
      // Add 1 empirical calibration offset
      if (half_max_value > best_max_value * thresh) {
        out.push_back(half_max_indices + 1);
      } else {
        out.push_back(best_max_indices + 1);
      }
    }
  }
  // unpack batch
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(out, out_shape, output));
  return Status::OK();
}

/// \brief Apply median smoothing to the 1D tensor over the given window.
/// \param input: Tensor of shape<channel, num_of_frames>.
/// \param output: Tensor of shape <channel, num_of_window>.
/// \param win_length: The window length for median smoothing (in number of frames).
/// \return Status code.
Status MedianSmoothing(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t win_length);

/// \brief Detect pitch frequency.
/// \param input: Tensor of shape <channel,waveform_length>.
/// \param output: Tensor of shape <channel, num_of_frames, lags>.
/// \param sample_rate: The sample rate of the waveform (Hz).
/// \param frame_time: Duration of a frame.
/// \param win_length: The window length for median smoothing (in number of frames).
/// \param freq_low: Lowest frequency that can be detected (Hz).
/// \param freq_high: Highest frequency that can be detected (Hz).
/// \return Status code.
Status DetectPitchFrequency(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t sample_rate,
                            float frame_time, int32_t win_length, int32_t freq_low, int32_t freq_high);

/// \brief A helper function for phaser, generates a table with given parameters.
/// \param output: Tensor of shape <time>.
/// \param type: can choose DataType::DE_FLOAT32 or DataType::DE_INT32.
/// \param modulation: Modulation of the input tensor.
///     It can be one of Modulation.kSinusoidal or Modulation.kTriangular.
/// \param table_size: The length of table.
/// \param min: Calculate the sampling rate within the delay time.
/// \param max: Calculate the sampling rate within the delay and delay depth time.
/// \param phase: Phase offset of function.
/// \return Status code.
Status GenerateWaveTable(std::shared_ptr<Tensor> *output, const DataType &type, Modulation modulation,
                         int32_t table_size, float min, float max, float phase);

/// \brief Apply a phaser effect to the audio.
/// \param input Tensor of shape <..., time>.
/// \param output Tensor of shape <..., time>.
/// \param sample_rate Sampling rate of the waveform.
/// \param gain_in Desired input gain at the boost (or attenuation) in dB.
/// \param gain_out Desired output gain at the boost (or attenuation) in dB.
/// \param delay_ms Desired delay in milli seconds.
/// \param decay Desired decay relative to gain-in.
/// \param mod_speed Modulation speed in Hz.
/// \param sinusoidal If true, use sinusoidal modulation. If false, use triangular modulation.
/// \return Status code.
template <typename T>
Status Phaser(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t sample_rate, float gain_in,
              float gain_out, float delay_ms, float decay, float mod_speed, bool sinusoidal) {
  TensorShape input_shape = input->shape();
  // input convert to 2D (channels,time)
  auto channels = input->Size() / input_shape[-1];
  auto time = input_shape[-1];
  TensorShape to_shape({channels, time});
  RETURN_IF_NOT_OK(input->Reshape(to_shape));
  // input vector
  std::vector<std::vector<T>> input_vec(channels, std::vector<T>(time, 0));
  // output vector
  std::vector<std::vector<T>> out_vec(channels, std::vector<T>(time, 0));
  // input convert to vector
  auto input_itr = input->begin<T>();
  for (size_t i = 0; i < channels; i++) {
    for (size_t j = 0; j < time; j++) {
      input_vec[i][j] = *input_itr * gain_in;
      input_itr++;
    }
  }
  // compute
  // create delay buffer
  int delay_buf_nrow = channels;
  // calculate the length of the delay
  int delay_buf_len = static_cast<int>((delay_ms * 0.001 * sample_rate) + 0.5);
  std::vector<std::vector<T>> delay_buf(delay_buf_nrow, std::vector<T>(delay_buf_len, 0 * decay));
  // calculate the length after the momentum
  int mod_buf_len = static_cast<int>(sample_rate / mod_speed + 0.5);
  Modulation modulation = sinusoidal ? Modulation::kSinusoidal : Modulation::kTriangular;
  // create and compute mod buffer
  std::shared_ptr<Tensor> mod_buf_tensor;
  RETURN_IF_NOT_OK(GenerateWaveTable(&mod_buf_tensor, DataType(DataType::DE_INT32), modulation, mod_buf_len,
                                     static_cast<float>(1.0f), static_cast<float>(delay_buf_len),
                                     static_cast<float>(PI / 2)));
  // tensor mod_buf convert to vector
  std::vector<int> mod_buf;
  for (auto itr = mod_buf_tensor->begin<int>(); itr != mod_buf_tensor->end<int>(); itr++) {
    mod_buf.push_back(*itr);
  }
  dsize_t delay_pos = 0;
  dsize_t mod_pos = 0;
  // for every channal at the current time
  for (size_t i = 0; i < time; i++) {
    // calculate the delay data that should be added to each channal at this time
    int idx = static_cast<int>((delay_pos + mod_buf[mod_pos]) % delay_buf_len);
    mod_pos = (mod_pos + 1) % mod_buf_len;
    delay_pos = (delay_pos + 1) % delay_buf_len;
    // update the next delay data with the current result * decay
    for (size_t j = 0; j < channels; j++) {
      out_vec[j][i] = input_vec[j][i] + delay_buf[j][idx];
      delay_buf[j][delay_pos] = (input_vec[j][i] + delay_buf[j][idx]) * decay;
    }
  }
  std::vector<T> out_vec_one_d;
  for (size_t i = 0; i < channels; i++) {
    for (size_t j = 0; j < time; j++) {
      // gain_out on the output
      out_vec[i][j] *= gain_out;
      // clamp
      out_vec[i][j] = std::max<float>(-1.0f, std::min<float>(1.0f, out_vec[i][j]));
      // output vector is transformed from 2d to 1d
      out_vec_one_d.push_back(out_vec[i][j]);
    }
  }
  // move data to output tensor
  std::shared_ptr<Tensor> out;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(out_vec_one_d, input_shape, &out));
  *output = out;
  return Status::OK();
}

/// \brief Flanger about interpolation effect.
/// \param input: Tensor of shape <batch, channel, time>.
/// \param int_delay: A dimensional vector about integer delay, subscript representing delay.
/// \param frac_delay: A dimensional vector about delay obtained by using the frac function.
/// \param interpolation: Interpolation of the input tensor.
///     It can be one of Interpolation::kLinear or Interpolation::kQuadratic.
/// \param delay_buf_pos: Minimum dimension length about delay_bufs.
/// \Returns Flanger about interpolation effect.
template <typename T>
std::vector<std::vector<T>> FlangerInterpolation(const std::shared_ptr<Tensor> &input, std::vector<int> int_delay,
                                                 const std::vector<T> &frac_delay, Interpolation interpolation,
                                                 int delay_buf_pos) {
  int n_batch = input->shape()[0];
  int n_channels = input->shape()[-2];
  int delay_buf_length = input->shape()[-1];

  std::vector<std::vector<T>> delayed_value_a(n_batch, std::vector<T>(n_channels, 0));
  std::vector<std::vector<T>> delayed_value_b(n_batch, std::vector<T>(n_channels, 0));
  for (int j = 0; j < n_batch; j++) {
    for (int k = 0; k < n_channels; k++) {
      // delay after obtaining the current number of channels
      auto iter_input = input->begin<T>();
      int it = j * n_channels * delay_buf_length + k * delay_buf_length;
      iter_input += it + (delay_buf_pos + int_delay[k]) % delay_buf_length;
      delayed_value_a[j][k] = *(iter_input);
      iter_input = input->begin<T>();
      iter_input += it + (delay_buf_pos + int_delay[k] + 1) % delay_buf_length;
      delayed_value_b[j][k] = *(iter_input);
    }
  }
  // delay subscript backward
  for (int j = 0; j < n_channels; j++) {
    int_delay[j] = int_delay[j] + 2;
  }
  std::vector<std::vector<T>> delayed(n_batch, std::vector<T>(n_channels, 0));
  std::vector<std::vector<T>> delayed_value_c(n_batch, std::vector<T>(n_channels, 0));
  if (interpolation == Interpolation::kLinear) {
    for (int j = 0; j < n_batch; j++) {
      for (int k = 0; k < n_channels; k++) {
        delayed[j][k] = delayed_value_a[j][k] + (delayed_value_b[j][k] - delayed_value_a[j][k]) * frac_delay[k];
      }
    }
  } else {
    for (int j = 0; j < n_batch; j++) {
      for (int k = 0; k < n_channels; k++) {
        auto iter_input = input->begin<T>();
        int it = j * n_channels * delay_buf_length + k * delay_buf_length;
        iter_input += it + (delay_buf_pos + int_delay[k]) % delay_buf_length;
        delayed_value_c[j][k] = *(iter_input);
      }
    }
    // delay subscript backward
    for (int j = 0; j < n_channels; j++) {
      int_delay[j] = int_delay[j] + 1;
    }
    std::vector<std::vector<T>> frac_delay_coefficient(n_batch, std::vector<T>(n_channels, 0));
    std::vector<std::vector<T>> frac_delay_value(n_batch, std::vector<T>(n_channels, 0));
    for (int j = 0; j < n_batch; j++) {
      for (int k = 0; k < n_channels; k++) {
        delayed_value_c[j][k] = delayed_value_c[j][k] - delayed_value_a[j][k];
        delayed_value_b[j][k] = delayed_value_b[j][k] - delayed_value_a[j][k];
        // delayed_value_c[j][k] * 0.5 is half of the delayed_value_c[j][k]
        frac_delay_coefficient[j][k] = delayed_value_c[j][k] * 0.5 - delayed_value_b[j][k];
        frac_delay_value[j][k] = delayed_value_b[j][k] * 2 - delayed_value_c[j][k] * 0.5;
        // the next delay is obtained by delaying the data in the buffer
        delayed[j][k] = delayed_value_a[j][k] +
                        (frac_delay_coefficient[j][k] * frac_delay[k] + frac_delay_value[j][k]) * frac_delay[k];
      }
    }
  }
  return delayed;
}

/// \brief Interval limiting function.
/// \param output_waveform: Tensor of shape <..., time>.
/// \param min: If value is less than min, min is returned.
/// \param max: If value is greater than max, max is returned.
/// \Returns Tensor at the same latitude.
template <typename T>
std::shared_ptr<Tensor> Clamp(const std::shared_ptr<Tensor> &tensor, T min, T max) {
  for (auto itr = tensor->begin<T>(); itr != tensor->end<T>(); itr++) {
    if (*itr > max) {
      *itr = max;
    } else if (*itr < min) {
      *itr = min;
    }
  }
  return tensor;
}

/// \brief Apply flanger effect.
/// \param input/output: Tensor of shape <..., channel, time>.
/// \param sample_rate: Sampling rate of the waveform, e.g. 44100 (Hz), the value can't be zero.
/// \param delay: Desired delay in milliseconds (ms), range: [0, 30].
/// \param depth: Desired delay depth in milliseconds (ms), range: [0, 10].
/// \param regen: Desired regen (feedback gain) in dB., range: [-95, 95].
/// \param width: Desired width (delay gain) in dB, range: [0, 100].
/// \param speed: Modulation speed in Hz, range: [0.1, 10].
/// \param phase: Percentage phase-shift for multi-channel, range: [0, 100].
/// \param modulation: Modulation of the input tensor.
///     It can be one of Modulation::kSinusoidal or Modulation::kTriangular.
/// \param interpolation: Interpolation of the input tensor.
///     It can be one of Interpolation::kLinear or Interpolation::kQuadratic.
/// \return Status code.
template <typename T>
Status Flanger(const std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, int32_t sample_rate, float delay,
               float depth, float regen, float width, float speed, float phase, Modulation modulation,
               Interpolation interpolation) {
  std::shared_ptr<Tensor> waveform;
  if (input->type() == DataType::DE_FLOAT64) {
    waveform = input;
  } else {
    RETURN_IF_NOT_OK(TypeCast(input, &waveform, DataType(DataType::DE_FLOAT32)));
  }
  // convert to 3D (batch, channels, time)
  TensorShape actual_shape = waveform->shape();
  TensorShape toShape({waveform->Size() / actual_shape[-2] / actual_shape[-1], actual_shape[-2], actual_shape[-1]});
  RETURN_IF_NOT_OK(waveform->Reshape(toShape));

  // scaling
  T feedback_gain = static_cast<T>(regen) / 100;
  T delay_gain = static_cast<T>(width) / 100;
  T channel_phase = static_cast<T>(phase) / 100;
  T delay_min = static_cast<T>(delay) / 1000;
  T delay_depth = static_cast<T>(depth) / 1000;

  // balance output:
  T in_gain = 1.0 / (1 + delay_gain);
  delay_gain = delay_gain / (1 + delay_gain);
  // balance feedback loop:
  delay_gain = delay_gain * (1 - abs(feedback_gain));

  int delay_buf_length = static_cast<int>((delay_min + delay_depth) * sample_rate + 0.5);
  delay_buf_length = delay_buf_length + 2;

  int lfo_length = static_cast<int>(sample_rate / speed);

  T table_min = floor(delay_min * sample_rate + 0.5);
  T table_max = delay_buf_length - 2.0;
  // generate wave table
  T lfo_phase = 3 * PI / 2;
  std::shared_ptr<Tensor> lfo;
  RETURN_IF_NOT_OK(GenerateWaveTable(&lfo, DataType(DataType::DE_FLOAT32), modulation, lfo_length,
                                     static_cast<float>(table_min), static_cast<float>(table_max),
                                     static_cast<float>(lfo_phase)));
  int n_batch = waveform->shape()[0];
  int n_channels = waveform->shape()[-2];
  int time = waveform->shape()[-1];
  std::vector<T> delay_tensor(n_channels, 0.0), frac_delay(n_channels, 0.0);
  std::vector<int> cur_channel_phase(n_channels, 0), int_delay(n_channels, 0);
  // next delay
  std::vector<std::vector<T>> delay_last(n_batch, std::vector<T>(n_channels, 0));

  // initialization of delay_bufs
  TensorShape delay_bufs_shape({n_batch, n_channels, delay_buf_length});
  std::shared_ptr<Tensor> delay_bufs, output_waveform;
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(delay_bufs_shape, waveform->type(), &delay_bufs));
  RETURN_IF_NOT_OK(delay_bufs->Zero());
  // initialization of output_waveform
  TensorShape output_waveform_shape({n_batch, n_channels, actual_shape[-1]});
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(output_waveform_shape, waveform->type(), &output_waveform));

  int delay_buf_pos = 0, lfo_pos = 0;
  for (int i = 0; i < time; i++) {
    delay_buf_pos = (delay_buf_pos + delay_buf_length - 1) % delay_buf_length;
    for (int j = 0; j < n_channels; j++) {
      // get current channel phase
      cur_channel_phase[j] = static_cast<int>(j * lfo_length * channel_phase + 0.5);
      // through the current channel phase and lfo arrays to get the delay
      auto iter_lfo = lfo->begin<float>();
      delay_tensor[j] = *(iter_lfo + static_cast<ptrdiff_t>((lfo_pos + cur_channel_phase[j]) % lfo_length));
      // the frac delay is obtained by using the frac function
      frac_delay[j] = delay_tensor[j] - static_cast<int>(delay_tensor[j]);
      delay_tensor[j] = floor(delay_tensor[j]);
      int_delay[j] = static_cast<int>(delay_tensor[j]);
    }
    // get the waveform of [:, :, i]
    std::shared_ptr<Tensor> temp;
    TensorShape temp_shape({n_batch, n_channels});
    RETURN_IF_NOT_OK(Tensor::CreateEmpty(temp_shape, waveform->type(), &temp));
    Slice ss1(0, n_batch), ss2(0, n_channels), ss3(i, i + 1);
    SliceOption sp1(ss1), sp2(ss2), sp3(ss3);
    std::vector<SliceOption> slice_option;
    slice_option.push_back(sp1), slice_option.push_back(sp2), slice_option.push_back(sp3);
    RETURN_IF_NOT_OK(waveform->Slice(&temp, slice_option));

    auto iter_temp = temp->begin<T>();
    auto iter_delay_bufs = delay_bufs->begin<T>();
    for (int j = 0; j < n_batch; j++) {
      for (int k = 0; k < n_channels; k++) {
        iter_delay_bufs += delay_buf_pos;
        // the value of delay_bufs is processed by next delay
        *(iter_delay_bufs) = *iter_temp + delay_last[j][k] * feedback_gain;
        iter_delay_bufs -= (delay_buf_pos - delay_buf_length);
        iter_temp++;
      }
    }
    // different delayed values can be obtained by judging the type of interpolation
    std::vector<std::vector<T>> delayed(n_batch, std::vector<T>(n_channels, 0));
    delayed = FlangerInterpolation<T>(delay_bufs, int_delay, frac_delay, interpolation, delay_buf_pos);

    for (int j = 0; j < n_channels; j++) {
      int_delay[j] = int_delay[j] + 1;
    }
    iter_temp = temp->begin<T>();
    for (int j = 0; j < n_batch; j++) {
      for (int k = 0; k < n_channels; k++) {
        auto iter_output_waveform = output_waveform->begin<T>();
        // update the next delay
        delay_last[j][k] = delayed[j][k];
        int it = j * n_channels * actual_shape[-1] + k * actual_shape[-1];
        iter_output_waveform += it + i;
        // the results are obtained by balancing the output and balancing the feedback loop
        *(iter_output_waveform) = *(iter_temp)*in_gain + delayed[j][k] * delay_gain;
        iter_temp++;
      }
    }
    // update lfo location
    lfo_pos = (lfo_pos + 1) % lfo_length;
  }
  // the output value is limited by the interval limit function
  output_waveform = Clamp<T>(output_waveform, -1, 1);
  // convert dimension to waveform dimension
  RETURN_IF_NOT_OK(output_waveform->Reshape(actual_shape));
  RETURN_IF_NOT_OK(TypeCast(output_waveform, output, input->type()));
  return Status::OK();
}

// A brief structure of wave file header.
struct WavHeader {
  int8_t chunk_id[4] = {0};
  int32_t chunk_size = 0;
  int8_t format[4] = {0};
  int8_t sub_chunk1_id[4] = {0};
  int32_t sub_chunk1_size = 0;
  int16_t audio_format = 0;
  int16_t num_channels = 0;
  int32_t sample_rate = 0;
  int32_t byte_rate = 0;
  int16_t byte_align = 0;
  int16_t bits_per_sample = 0;
  int8_t sub_chunk2_id[4] = {0};
  int32_t sub_chunk2_size = 0;
  WavHeader() {}
};

/// \brief Get an audio data from a wav file and store into a vector.
/// \param wav_file_dir: wave file dir.
/// \param waveform_vec: vector of waveform.
/// \param sample_rate: sample rate.
/// \return Status code.
Status ReadWaveFile(const std::string &wav_file_dir, std::vector<float> *waveform_vec, int32_t *sample_rate);

/// \brief Apply sliding-window cepstral mean and variance (optional) normalization per utterance.
/// \param input: Tensor of shape <..., freq, time>.
/// \param output: Tensor of shape <..., frame>.
/// \param cmn_window: Window in frames for running average CMN computation.
/// \param min_cmn_window: Minimum CMN window used at start of decoding.
/// \param center: If true, use a window centered on the current frame. If false, window is to the left.
/// \param norm_vars: If true, normalize variance to one.
/// \return Status code.
Status SlidingWindowCmn(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t cmn_window,
                        int32_t min_cmn_window, bool center, bool norm_vars);
/// \brief Compute delta coefficients of a tensor, usually a spectrogram.
/// \param input: Tensor of shape <...,freq,time>.
/// \param output: Tensor of shape <...,freq,time>.
/// \param win_length: The window length used for computing delta.
/// \param mode: Padding mode.
/// \return Status code.
Status ComputeDeltas(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int32_t win_length,
                     const BorderType &mode);

template <typename T>
Status Mul(const std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, T value) {
  RETURN_UNEXPECTED_IF_NULL(output);
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(input->shape(), input->type(), output));
  auto iter_in = input->begin<T>();
  auto iter_out = (*output)->begin<T>();
  for (; iter_in != input->end<T>(); ++iter_in, ++iter_out) {
    *iter_out = (*iter_in) * value;
  }
  return Status::OK();
}

template <typename T>
Status Div(const std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, T value) {
  RETURN_UNEXPECTED_IF_NULL(output);
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(input->shape(), input->type(), output));
  CHECK_FAIL_RETURN_UNEXPECTED(value != 0, "Div: invalid parameter, 'value' can not be zero.");
  auto iter_in = input->begin<T>();
  auto iter_out = (*output)->begin<T>();
  for (; iter_in != input->end<T>(); ++iter_in, ++iter_out) {
    *iter_out = (*iter_in) / value;
  }
  return Status::OK();
}

template <typename T>
Status Add(const std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> *output, T value) {
  RETURN_UNEXPECTED_IF_NULL(output);
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(input->shape(), input->type(), output));
  auto iter_in = input->begin<T>();
  auto iter_out = (*output)->begin<T>();
  for (; iter_in != input->end<T>(); ++iter_in, ++iter_out) {
    *iter_out = (*iter_in) + value;
  }
  return Status::OK();
}

template <typename T>
Status SubTensor(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int len) {
  RETURN_UNEXPECTED_IF_NULL(output);
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({len}), input->type(), output));
  RETURN_IF_NOT_OK(
    ValidateNoGreaterThan("SubTensor", "len", len, "size of input tensor", static_cast<int>(input->Size())));
  auto iter_in = input->begin<T>();
  auto iter_out = (*output)->begin<T>();
  for (; iter_out != (*output)->end<T>(); ++iter_in, ++iter_out) {
    *iter_out = *iter_in;
  }
  return Status::OK();
}

template <typename T>
Status TensorAdd(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
                 std::shared_ptr<Tensor> *output) {
  RETURN_UNEXPECTED_IF_NULL(output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape() == other->shape(), "TensorAdd: input tensor shape must be the same.");
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == other->type(), "TensorAdd: input tensor type must be the same.");

  RETURN_IF_NOT_OK(Tensor::CreateEmpty(input->shape(), input->type(), output));
  auto iter_in1 = input->begin<T>();
  auto iter_in2 = other->begin<T>();
  auto iter_out = (*output)->begin<T>();
  for (; iter_out != (*output)->end<T>(); ++iter_in1, ++iter_in2, ++iter_out) {
    *iter_out = (*iter_in1) + (*iter_in2);
  }
  return Status::OK();
}

template <typename T>
Status TensorSub(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
                 std::shared_ptr<Tensor> *output) {
  RETURN_UNEXPECTED_IF_NULL(output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape() == other->shape(), "TensorSub: input tensor shape must be the same.");
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == other->type(), "TensorSub: input tensor type must be the same.");

  RETURN_IF_NOT_OK(Tensor::CreateEmpty(input->shape(), input->type(), output));
  auto iter_in1 = input->begin<T>();
  auto iter_in2 = other->begin<T>();
  auto iter_out = (*output)->begin<T>();
  for (; iter_out != (*output)->end<T>(); ++iter_in1, ++iter_in2, ++iter_out) {
    *iter_out = (*iter_in1) - (*iter_in2);
  }
  return Status::OK();
}

template <typename T>
Status TensorCat(const std::shared_ptr<Tensor> &input, const std::shared_ptr<Tensor> &other,
                 std::shared_ptr<Tensor> *output) {
  RETURN_UNEXPECTED_IF_NULL(output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == other->type(), "TensorCat: input tensor type must be the same.");
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({input->shape()[-1] + other->shape()[-1]}), input->type(), output));
  auto iter_in1 = input->begin<T>();
  auto iter_in2 = other->begin<T>();
  auto iter_out = (*output)->begin<T>();
  for (; iter_in1 != input->end<T>(); ++iter_in1, ++iter_out) {
    *iter_out = *iter_in1;
  }
  for (; iter_in2 != other->end<T>(); ++iter_in2, ++iter_out) {
    *iter_out = *iter_in2;
  }
  return Status::OK();
}

template <typename T>
Status TensorRepeat(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int rank_repeat) {
  RETURN_UNEXPECTED_IF_NULL(output);

  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({rank_repeat, (input->shape()[-1])}), input->type(), output));
  auto iter_in = input->begin<T>();
  auto iter_out = (*output)->begin<T>();
  for (int i = 0; i < rank_repeat; i++) {
    auto iter_in = input->begin<T>();
    for (; iter_in != input->end<T>(); ++iter_in, ++iter_out) {
      *iter_out = *iter_in;
    }
  }
  return Status::OK();
}

template <typename T>
Status TensorRowReplace(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int row) {
  RETURN_UNEXPECTED_IF_NULL(output);
  auto iter_in = input->begin<T>();
  auto iter_out = (*output)->begin<T>() + static_cast<ptrdiff_t>((*output)->shape()[-1] * row);
  CHECK_FAIL_RETURN_UNEXPECTED(iter_out <= (*output)->end<T>(), "TensorRowReplace: pointer out of bounds");
  CHECK_FAIL_RETURN_UNEXPECTED(input->Size() <= (*output)->shape()[-1], "TensorRowReplace: pointer out of bounds");
  for (; iter_in != input->end<T>(); ++iter_in, ++iter_out) {
    *iter_out = *iter_in;
  }
  return Status::OK();
}

template <typename T>
Status TensorRowAt(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, int rank_index) {
  RETURN_UNEXPECTED_IF_NULL(output);
  RETURN_IF_NOT_OK(Tensor::CreateEmpty(TensorShape({input->shape()[-1]}), input->type(), output));
  auto iter_in = input->begin<T>() + static_cast<ptrdiff_t>(input->shape()[-1] * rank_index);
  auto iter_out = (*output)->begin<T>();
  CHECK_FAIL_RETURN_UNEXPECTED(iter_in <= input->end<T>(), "TensorRowAt: pointer out of bounds");
  for (; iter_out != (*output)->end<T>(); ++iter_in, ++iter_out) {
    *iter_out = *iter_in;
  }
  return Status::OK();
}

template <typename T>
Status TensorRound(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  RETURN_UNEXPECTED_IF_NULL(output);

  RETURN_IF_NOT_OK(Tensor::CreateEmpty(input->shape(), input->type(), output));
  auto iter_in = input->begin<T>();
  auto iter_out = (*output)->begin<T>();
  for (; iter_in != input->end<T>(); ++iter_in, ++iter_out) {
    *iter_out = round(*iter_in);
  }
  return Status::OK();
}

template <typename T>
Status ApplyProbabilityDistribution(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output,
                                    DensityFunction density_function, std::mt19937 rnd) {
  int channel_size = input->shape()[0] - 1;
  int time_size = input->shape()[-1] - 1;
  std::uniform_int_distribution<> dis_channel(0, channel_size);
  int random_channel = channel_size > 0 ? dis_channel(rnd) : 0;
  std::uniform_int_distribution<> dis_time(0, time_size);
  int random_time = time_size > 0 ? dis_time(rnd) : 0;
  int number_of_bits = 16;
  int up_scaling = static_cast<int>(pow(2, number_of_bits - 1) - 2);
  int down_scaling = static_cast<int>(pow(2, number_of_bits - 1));

  std::shared_ptr<Tensor> signal_scaled;
  RETURN_IF_NOT_OK(Mul<T>(input, &signal_scaled, up_scaling));

  std::shared_ptr<Tensor> signal_scaled_dis;
  RETURN_IF_NOT_OK(Tensor::CreateFromTensor(input, &signal_scaled_dis));

  if (density_function == DensityFunction::kRPDF) {
    auto iter_in = input->begin<T>();
    iter_in += (time_size + 1) * random_channel + random_time;
    auto RPDF = *(iter_in);
    RETURN_IF_NOT_OK(Add<T>(signal_scaled, &signal_scaled_dis, RPDF));
  } else if (density_function == DensityFunction::kGPDF) {
    int num_rand_variables = 6;
    auto iter_in = input->begin<T>();
    iter_in += (time_size + 1) * random_channel + random_time;
    auto gaussian = *(iter_in);
    for (int i = 0; i < num_rand_variables; i++) {
      int rand_channel = channel_size > 0 ? dis_channel(rnd) : 0;
      int rand_time = time_size > 0 ? dis_time(rnd) : 0;

      auto iter_in_rand = input->begin<T>();
      iter_in_rand += (time_size + 1) * rand_channel + rand_time;
      gaussian += *(iter_in_rand);
      *(iter_in_rand) = gaussian;
    }
    RETURN_IF_NOT_OK(Add<T>(signal_scaled, &signal_scaled_dis, gaussian));
  } else {
    int window_length = time_size + 1;
    std::shared_ptr<Tensor> float_bartlett;
    RETURN_IF_NOT_OK(Bartlett(&float_bartlett, window_length));
    std::shared_ptr<Tensor> type_convert_bartlett;
    RETURN_IF_NOT_OK(TypeCast(float_bartlett, &type_convert_bartlett, input->type()));

    int rank_repeat = channel_size + 1;
    std::shared_ptr<Tensor> TPDF;
    RETURN_IF_NOT_OK(TensorRepeat<T>(type_convert_bartlett, &TPDF, rank_repeat));
    RETURN_IF_NOT_OK(TensorAdd<T>(signal_scaled, TPDF, &signal_scaled_dis));
  }
  std::shared_ptr<Tensor> quantised_signal_scaled;
  RETURN_IF_NOT_OK(TensorRound<T>(signal_scaled_dis, &quantised_signal_scaled));

  std::shared_ptr<Tensor> quantised_signal;
  RETURN_IF_NOT_OK(Div<T>(quantised_signal_scaled, &quantised_signal, down_scaling));
  *output = quantised_signal;
  return Status::OK();
}

template <typename T>
Status AddNoiseShaping(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  std::shared_ptr<Tensor> dithered_waveform;
  RETURN_IF_NOT_OK(Tensor::CreateFromTensor(*output, &dithered_waveform));
  std::shared_ptr<Tensor> waveform;
  RETURN_IF_NOT_OK(Tensor::CreateFromTensor(input, &waveform));

  std::shared_ptr<Tensor> error;
  RETURN_IF_NOT_OK(TensorSub<T>(dithered_waveform, waveform, &error));
  for (int i = 0; i < error->shape()[0]; i++) {
    std::shared_ptr<Tensor> err;
    RETURN_IF_NOT_OK(TensorRowAt<T>(error, &err, i));
    std::shared_ptr<Tensor> tensor_zero;
    std::vector<T> vector_zero(1, 0);
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(vector_zero, TensorShape({1}), &tensor_zero));
    std::shared_ptr<Tensor> error_offset;
    RETURN_IF_NOT_OK(TensorCat<T>(tensor_zero, err, &error_offset));
    int k = error->shape()[-1];
    std::shared_ptr<Tensor> fresh_error_offset;
    RETURN_IF_NOT_OK(SubTensor<T>(error_offset, &fresh_error_offset, k));
    RETURN_IF_NOT_OK(TensorRowReplace<T>(fresh_error_offset, &error, i));
  }
  std::shared_ptr<Tensor> noise_shaped;
  RETURN_IF_NOT_OK(TensorAdd<T>(dithered_waveform, error, &noise_shaped));
  *output = noise_shaped;
  return Status::OK();
}

/// \brief Apply dither effect.
/// \param input/output: Tensor of shape <..., time>.
/// \param density_function: The density function of a continuous random variable.
/// \param noise_shaing: A filtering process that shapes the spectral energy of quantisation error.
/// \return Status code.
template <typename T>
Status Dither(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output, DensityFunction density_function,
              bool noise_shaping, std::mt19937 rnd) {
  TensorShape shape = input->shape();
  TensorShape new_shape({input->Size() / shape[-1], shape[-1]});
  RETURN_IF_NOT_OK(input->Reshape(new_shape));

  RETURN_IF_NOT_OK(ApplyProbabilityDistribution<T>(input, output, density_function, rnd));
  if (noise_shaping) {
    RETURN_IF_NOT_OK(AddNoiseShaping<T>(input, output));
  }

  RETURN_IF_NOT_OK((*output)->Reshape(shape));
  RETURN_IF_NOT_OK(input->Reshape(shape));
  return Status::OK();
}

}  // namespace dataset
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_CCSRC_MINDDATA_DATASET_AUDIO_KERNELS_AUDIO_UTILS_H_
