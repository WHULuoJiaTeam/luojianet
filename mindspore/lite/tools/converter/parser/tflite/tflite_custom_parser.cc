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

#include "tools/converter/parser/tflite/tflite_custom_parser.h"
#include <map>
#include <memory>
#include <vector>
#include "flatbuffers/flexbuffers.h"

#include "ops/audio_spectrogram.h"
#include "ops/custom_extract_features.h"
#include "ops/custom_normalize.h"
#include "ops/custom_predict.h"
#include "ops/detection_post_process.h"
#include "ops/identity.h"
#include "ops/fft_real.h"
#include "ops/fft_imag.h"
#include "ops/mfcc.h"
#include "ops/rfft.h"
#include "nnacl/op_base.h"

namespace mindspore {
namespace lite {
PrimitiveCPtr TfliteCustomParser::Parse(const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                        const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                        const std::unique_ptr<tflite::ModelT> &tflite_model) {
  const auto &custom_attr = tflite_op->custom_options;
  const auto &opnode = tflite_model->operator_codes.at(tflite_op->opcode_index);
  if (opnode == nullptr) {
    MS_LOG(ERROR) << "opnode is null";
    return nullptr;
  }
  const auto &custom_type = opnode->custom_code;
  if (custom_type == "TFLite_Detection_PostProcess") {
    return DetectPostProcess(custom_attr, tflite_op);
  } else if (custom_type == "Predict") {
    return Predict(custom_attr);
  } else if (custom_type == "Normalize") {
    return Normalize();
  } else if (custom_type == "ExtractFeatures") {
    return ExtractFeatures();
  } else if (custom_type == "AudioSpectrogram") {
    return AudioSpectrogram(custom_attr);
  } else if (custom_type == "Mfcc") {
    return Mfcc(custom_attr);
  } else if (custom_type == "FlexRFFT") {
    return Rfft(custom_attr, tflite_op, tflite_subgraph, tflite_model);
  } else if (custom_type == "FlexReal") {
    return FftReal();
  } else if (custom_type == "FlexImag") {
    return FftImag();
  } else {
    MS_LOG(ERROR) << "custom type : " << custom_type << " is not supported";
    return nullptr;
  }
}

PrimitiveCPtr TfliteCustomParser::DetectPostProcess(const std::vector<uint8_t> &custom_attr,
                                                    const std::unique_ptr<tflite::OperatorT> &tflite_op) {
  MS_CHECK_TRUE_RET(tflite_op != nullptr, nullptr);
  auto prim = std::make_unique<ops::DetectionPostProcess>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  prim->set_format(mindspore::Format::NHWC);
  prim->set_input_size(tflite_op->inputs.size());

  auto attr_map = flexbuffers::GetRoot(custom_attr).AsMap();
  prim->set_scale({attr_map["h_scale"].AsFloat(), attr_map["w_scale"].AsFloat(), attr_map["x_scale"].AsFloat(),
                   attr_map["y_scale"].AsFloat()});
  prim->set_nms_iou_threshold(attr_map["nms_iou_threshold"].AsFloat());
  prim->set_nms_score_threshold(attr_map["nms_score_threshold"].AsFloat());
  prim->set_max_detections(attr_map["max_detections"].AsInt64());
  if (attr_map["detections_per_class"].IsNull()) {
    prim->set_detections_per_class(100);
  } else {
    prim->set_detections_per_class(attr_map["detections_per_class"].AsInt64());
  }
  prim->set_max_classes_per_detection(attr_map["max_classes_per_detection"].AsInt64());
  prim->set_num_classes(attr_map["num_classes"].AsInt64());
  if (attr_map["use_regular_nms"].IsNull()) {
    prim->set_use_regular_nms(false);
  } else {
    prim->set_use_regular_nms(attr_map["use_regular_nms"].AsBool());
  }
  if (attr_map["_output_quantized"].IsNull()) {
    prim->set_out_quantized(false);
  } else {
    prim->set_out_quantized(attr_map["_output_quantized"].AsBool());
  }

  return prim->GetPrim();
}

PrimitiveCPtr TfliteCustomParser::AudioSpectrogram(const std::vector<uint8_t> &custom_attr) {
  auto prim = std::make_unique<ops::AudioSpectrogram>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  auto attr_map = flexbuffers::GetRoot(custom_attr).AsMap();
  prim->set_window_size(attr_map["window_size"].AsInt64());
  prim->set_stride(attr_map["stride"].AsInt64());
  prim->set_mag_square(attr_map["magnitude_squared"].AsBool());

  return prim->GetPrim();
}

PrimitiveCPtr TfliteCustomParser::Mfcc(const std::vector<uint8_t> &custom_attr) {
  auto prim = std::make_unique<ops::Mfcc>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  auto attr_map = flexbuffers::GetRoot(custom_attr).AsMap();
  prim->set_freq_upper_limit(attr_map["upper_frequency_limit"].AsFloat());
  prim->set_freq_lower_limit(attr_map["lower_frequency_limit"].AsFloat());
  prim->set_filter_bank_channel_num(attr_map["filterbank_channel_count"].AsInt64());
  prim->set_dct_coeff_num(attr_map["dct_coefficient_count"].AsInt64());

  return prim->GetPrim();
}

PrimitiveCPtr TfliteCustomParser::Predict(const std::vector<uint8_t> &custom_attr) {
  auto prim = std::make_unique<ops::CustomPredict>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  MS_CHECK_TRUE_RET(custom_attr.data() != nullptr, nullptr);
  MS_CHECK_GE(custom_attr.size(), kInputSize1, nullptr);
  auto out_num = reinterpret_cast<const int *>(custom_attr.data())[0];
  auto weight_thres = reinterpret_cast<const float *>(custom_attr.data())[1];
  prim->set_output_num(static_cast<int64_t>(out_num));
  prim->set_weight_threshold(weight_thres);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteCustomParser::Normalize() {
  auto prim = std::make_unique<ops::CustomNormalize>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteCustomParser::ExtractFeatures() {
  auto prim = std::make_unique<ops::CustomExtractFeatures>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteCustomParser::Rfft(const std::vector<uint8_t> &custom_attr,
                                       const std::unique_ptr<tflite::OperatorT> &tflite_op,
                                       const std::unique_ptr<tflite::SubGraphT> &tflite_subgraph,
                                       const std::unique_ptr<tflite::ModelT> &tflite_model) {
  MS_CHECK_TRUE_RET(tflite_op != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_subgraph != nullptr, nullptr);
  MS_CHECK_TRUE_RET(tflite_model != nullptr, nullptr);
  MS_CHECK_GE(tflite_op->inputs.size(), kInputSize1, nullptr);
  auto prim = std::make_unique<ops::Rfft>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);

  std::vector<int64_t> fft_length;
  if (GetTfliteData(tflite_op->inputs[SECOND_INPUT], tflite_subgraph->tensors, tflite_model->buffers, &fft_length)) {
    MS_LOG(ERROR) << "rfft -> fftLength get failed";
    return nullptr;
  }
  prim->set_fft_length(fft_length[0]);

  return prim->GetPrim();
}

PrimitiveCPtr TfliteCustomParser::FftReal() {
  auto prim = std::make_unique<ops::FftReal>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteCustomParser::FftImag() {
  auto prim = std::make_unique<ops::FftImag>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

PrimitiveCPtr TfliteCustomParser::Identity() {
  auto prim = std::make_unique<ops::Identity>();
  MS_CHECK_TRUE_RET(prim != nullptr, nullptr);
  return prim->GetPrim();
}

TfliteNodeRegister g_tfliteCustomParser(tflite::BuiltinOperator_CUSTOM, new TfliteCustomParser());
}  // namespace lite
}  // namespace mindspore
