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

#include "graph/utils/type_utils.h"
#include <algorithm>
#include "graph/buffer.h"
#include "graph/debug/ge_util.h"
#include "external/graph/types.h"

namespace ge {
namespace{
const std::map<Format, std::string> kFormatToStringMap = {
    {FORMAT_NCHW, "NCHW"},
    {FORMAT_NHWC, "NHWC"},
    {FORMAT_ND, "ND"},
    {FORMAT_NC1HWC0, "NC1HWC0"},
    {FORMAT_FRACTAL_Z, "FRACTAL_Z"},
    {FORMAT_NC1C0HWPAD, "NC1C0HWPAD"},
    {FORMAT_NHWC1C0, "NHWC1C0"},
    {FORMAT_FSR_NCHW, "FSR_NCHW"},
    {FORMAT_FRACTAL_DECONV, "FRACTAL_DECONV"},
    {FORMAT_C1HWNC0, "C1HWNC0"},
    {FORMAT_FRACTAL_DECONV_TRANSPOSE, "FRACTAL_DECONV_TRANSPOSE"},
    {FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS, "FRACTAL_DECONV_SP_STRIDE_TRANS"},
    {FORMAT_NC1HWC0_C04, "NC1HWC0_C04"},
    {FORMAT_FRACTAL_Z_C04, "FRACTAL_Z_C04"},
    {FORMAT_CHWN, "CHWN"},
    {FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS, "DECONV_SP_STRIDE8_TRANS"},
    {FORMAT_NC1KHKWHWC0, "NC1KHKWHWC0"},
    {FORMAT_BN_WEIGHT, "BN_WEIGHT"},
    {FORMAT_FILTER_HWCK, "FILTER_HWCK"},
    {FORMAT_HWCN, "HWCN"},
    {FORMAT_HASHTABLE_LOOKUP_LOOKUPS, "LOOKUP_LOOKUPS"},
    {FORMAT_HASHTABLE_LOOKUP_KEYS, "LOOKUP_KEYS"},
    {FORMAT_HASHTABLE_LOOKUP_VALUE, "LOOKUP_VALUE"},
    {FORMAT_HASHTABLE_LOOKUP_OUTPUT, "LOOKUP_OUTPUT"},
    {FORMAT_HASHTABLE_LOOKUP_HITS, "LOOKUP_HITS"},
    {FORMAT_MD, "MD"},
    {FORMAT_NDHWC, "NDHWC"},
    {FORMAT_NCDHW, "NCDHW"},
    {FORMAT_DHWCN, "DHWCN"},
    {FORMAT_DHWNC, "DHWNC"},
    {FORMAT_NDC1HWC0, "NDC1HWC0"},
    {FORMAT_FRACTAL_Z_3D, "FRACTAL_Z_3D"},
    {FORMAT_FRACTAL_Z_3D_TRANSPOSE, "FRACTAL_Z_3D_TRANSPOSE"},
    {FORMAT_C1HWNCoC0, "C1HWNCoC0"},
    {FORMAT_FRACTAL_NZ, "FRACTAL_NZ"},
    {FORMAT_CN, "CN"},
    {FORMAT_NC, "NC"},
    {FORMAT_FRACTAL_ZN_LSTM, "FRACTAL_ZN_LSTM"},
    {FORMAT_FRACTAL_Z_G, "FRACTAL_Z_G"},
    {FORMAT_ND_RNN_BIAS, "ND_RNN_BIAS"},
    {FORMAT_FRACTAL_ZN_RNN, "FRACTAL_ZN_RNN"},
    {FORMAT_RESERVED, "FORMAT_RESERVED"},
    {FORMAT_ALL, "ALL"},
    {FORMAT_NULL, "NULL"},
    {FORMAT_END, "END"},
    {FORMAT_MAX, "MAX"}};

const std::map<domi::domiTensorFormat_t, Format> kDomiFormatToGeFormat = {
    {domi::DOMI_TENSOR_NCHW, FORMAT_NCHW},
    {domi::DOMI_TENSOR_NHWC, FORMAT_NHWC},
    {domi::DOMI_TENSOR_ND, FORMAT_ND},
    {domi::DOMI_TENSOR_NC1HWC0, FORMAT_NC1HWC0},
    {domi::DOMI_TENSOR_FRACTAL_Z, FORMAT_FRACTAL_Z},
    {domi::DOMI_TENSOR_NC1C0HWPAD, FORMAT_NC1C0HWPAD},
    {domi::DOMI_TENSOR_NHWC1C0, FORMAT_NHWC1C0},
    {domi::DOMI_TENSOR_FSR_NCHW, FORMAT_FSR_NCHW},
    {domi::DOMI_TENSOR_FRACTAL_DECONV, FORMAT_FRACTAL_DECONV},
    {domi::DOMI_TENSOR_BN_WEIGHT, FORMAT_BN_WEIGHT},
    {domi::DOMI_TENSOR_CHWN, FORMAT_CHWN},
    {domi::DOMI_TENSOR_FILTER_HWCK, FORMAT_FILTER_HWCK},
    {domi::DOMI_TENSOR_NDHWC, FORMAT_NDHWC},
    {domi::DOMI_TENSOR_NCDHW, FORMAT_NCDHW},
    {domi::DOMI_TENSOR_DHWCN, FORMAT_DHWCN},
    {domi::DOMI_TENSOR_DHWNC, FORMAT_DHWNC},
    {domi::DOMI_TENSOR_RESERVED, FORMAT_RESERVED}
};

const std::set<std::string> kInternalFormat = {
    "NC1HWC0",
    "FRACTAL_Z",
    "NC1C0HWPAD",
    "NHWC1C0",
    "FRACTAL_DECONV",
    "C1HWNC0",
    "FRACTAL_DECONV_TRANSPOSE",
    "FRACTAL_DECONV_SP_STRIDE_TRANS",
    "NC1HWC0_C04",
    "FRACTAL_Z_C04",
    "FRACTAL_DECONV_SP_STRIDE8_TRANS",
    "NC1KHKWHWC0",
    "C1HWNCoC0",
    "FRACTAL_ZZ",
    "FRACTAL_NZ",
    "NDC1HWC0",
    "FRACTAL_Z_3D",
    "FRACTAL_Z_3D_TRANSPOSE",
    "FRACTAL_ZN_LSTM",
    "FRACTAL_Z_G",
    "ND_RNN_BIAS",
    "FRACTAL_ZN_RNN"
};

const std::map<std::string, Format> kDataFormatMap = {
    {"NCHW", FORMAT_NCHW},
    {"NHWC", FORMAT_NHWC},
    {"NDHWC", FORMAT_NDHWC},
    {"NCDHW", FORMAT_NCDHW},
    {"ND",   FORMAT_ND}};

const std::map<std::string, Format> kStringToFormatMap = {
    {"NCHW", FORMAT_NCHW},
    {"NHWC", FORMAT_NHWC},
    {"ND", FORMAT_ND},
    {"NC1HWC0", FORMAT_NC1HWC0},
    {"FRACTAL_Z", FORMAT_FRACTAL_Z},
    {"NC1C0HWPAD", FORMAT_NC1C0HWPAD},
    {"NHWC1C0", FORMAT_NHWC1C0},
    {"FSR_NCHW", FORMAT_FSR_NCHW},
    {"FRACTAL_DECONV", FORMAT_FRACTAL_DECONV},
    {"C1HWNC0", FORMAT_C1HWNC0},
    {"FRACTAL_DECONV_TRANSPOSE", FORMAT_FRACTAL_DECONV_TRANSPOSE},
    {"FRACTAL_DECONV_SP_STRIDE_TRANS", FORMAT_FRACTAL_DECONV_SP_STRIDE_TRANS},
    {"NC1HWC0_C04", FORMAT_NC1HWC0_C04},
    {"FRACTAL_Z_C04", FORMAT_FRACTAL_Z_C04},
    {"CHWN", FORMAT_CHWN},
    {"DECONV_SP_STRIDE8_TRANS", FORMAT_FRACTAL_DECONV_SP_STRIDE8_TRANS},
    {"NC1KHKWHWC0", FORMAT_NC1KHKWHWC0},
    {"BN_WEIGHT", FORMAT_BN_WEIGHT},
    {"FILTER_HWCK", FORMAT_FILTER_HWCK},
    {"HWCN", FORMAT_HWCN},
    {"LOOKUP_LOOKUPS", FORMAT_HASHTABLE_LOOKUP_LOOKUPS},
    {"LOOKUP_KEYS", FORMAT_HASHTABLE_LOOKUP_KEYS},
    {"LOOKUP_VALUE", FORMAT_HASHTABLE_LOOKUP_VALUE},
    {"LOOKUP_OUTPUT", FORMAT_HASHTABLE_LOOKUP_OUTPUT},
    {"LOOKUP_HITS", FORMAT_HASHTABLE_LOOKUP_HITS},
    {"MD", FORMAT_MD},
    {"C1HWNCoC0", FORMAT_C1HWNCoC0},
    {"FRACTAL_NZ", FORMAT_FRACTAL_NZ},
    {"NDHWC", FORMAT_NDHWC},
    {"NCDHW", FORMAT_NCDHW},
    {"DHWCN", FORMAT_DHWCN},
    {"DHWNC", FORMAT_DHWNC},
    {"NDC1HWC0", FORMAT_NDC1HWC0},
    {"FRACTAL_Z_3D", FORMAT_FRACTAL_Z_3D},
    {"FRACTAL_Z_3D_TRANSPOSE", FORMAT_FRACTAL_Z_3D_TRANSPOSE},
    {"CN", FORMAT_CN},
    {"NC", FORMAT_NC},
    {"FRACTAL_ZN_LSTM", FORMAT_FRACTAL_ZN_LSTM},
    {"FRACTAL_Z_G", FORMAT_FRACTAL_Z_G},
    {"FORMAT_RESERVED", FORMAT_RESERVED},
    {"ALL", FORMAT_ALL},
    {"NULL", FORMAT_NULL},
    // add for json input
    {"ND_RNN_BIAS", FORMAT_ND_RNN_BIAS},
    {"FRACTAL_ZN_RNN", FORMAT_FRACTAL_ZN_RNN},
    {"RESERVED", FORMAT_RESERVED},
    {"UNDEFINED", FORMAT_RESERVED}
  };

const std::map<DataType, std::string> kDataTypeToStringMap = {
    {DT_UNDEFINED, "DT_UNDEFINED"},            // Used to indicate a DataType field has not been set.
    {DT_FLOAT, "DT_FLOAT"},                    // float type
    {DT_FLOAT16, "DT_FLOAT16"},                // fp16 type
    {DT_INT8, "DT_INT8"},                      // int8 type
    {DT_INT16, "DT_INT16"},                    // int16 type
    {DT_UINT16, "DT_UINT16"},                  // uint16 type
    {DT_UINT8, "DT_UINT8"},                    // uint8 type
    {DT_INT32, "DT_INT32"},                    // uint32 type
    {DT_INT64, "DT_INT64"},                    // int64 type
    {DT_UINT32, "DT_UINT32"},                  // unsigned int32
    {DT_UINT64, "DT_UINT64"},                  // unsigned int64
    {DT_BOOL, "DT_BOOL"},                      // bool type
    {DT_DOUBLE, "DT_DOUBLE"},                  // double type
    {DT_DUAL, "DT_DUAL"},                      // dual output type
    {DT_DUAL_SUB_INT8, "DT_DUAL_SUB_INT8"},    // dual output int8 type
    {DT_DUAL_SUB_UINT8, "DT_DUAL_SUB_UINT8"},  // dual output uint8 type
    {DT_COMPLEX64, "DT_COMPLEX64"},            // complex64 type
    {DT_COMPLEX128, "DT_COMPLEX128"},          // complex128 type
    {DT_QINT8, "DT_QINT8"},                    // qint8 type
    {DT_QINT16, "DT_QINT16"},                  // qint16 type
    {DT_QINT32, "DT_QINT32"},                  // qint32 type
    {DT_QUINT8, "DT_QUINT8"},                  // quint8 type
    {DT_QUINT16, "DT_QUINT16"},                // quint16 type
    {DT_RESOURCE, "DT_RESOURCE"},              // resource type
    {DT_STRING_REF, "DT_STRING_REF"},          // string ref type
    {DT_STRING, "DT_STRING"},                  // string type
    {DT_VARIANT, "DT_VARIANT"},                // dt_variant type
    {DT_BF16, "DT_BFLOAT16"},                  // dt_bfloat16 type
    {DT_INT4, "DT_INT4"},                      // dt_variant type
    {DT_UINT1, "DT_UINT1"},                    // dt_variant type
    {DT_INT2, "DT_INT2"},                      // dt_variant type
    {DT_UINT2, "DT_UINT2"}                     // dt_variant type
};

const std::map<std::string, DataType> kStringTodataTypeMap = {
    {"DT_UNDEFINED", DT_UNDEFINED},            // Used to indicate a DataType field has not been set.
    {"DT_FLOAT", DT_FLOAT},                    // float type
    {"DT_FLOAT16", DT_FLOAT16},                // fp16 type
    {"DT_INT8", DT_INT8},                      // int8 type
    {"DT_INT16", DT_INT16},                    // int16 type
    {"DT_UINT16", DT_UINT16},                  // uint16 type
    {"DT_UINT8", DT_UINT8},                    // uint8 type
    {"DT_INT32", DT_INT32},                    // uint32 type
    {"DT_INT64", DT_INT64},                    // int64 type
    {"DT_UINT32", DT_UINT32},                  // unsigned int32
    {"DT_UINT64", DT_UINT64},                  // unsigned int64
    {"DT_BOOL", DT_BOOL},                      // bool type
    {"DT_DOUBLE", DT_DOUBLE},                  // double type
    {"DT_DUAL", DT_DUAL},                      // dual output type
    {"DT_DUAL_SUB_INT8", DT_DUAL_SUB_INT8},    // dual output int8 type
    {"DT_DUAL_SUB_UINT8", DT_DUAL_SUB_UINT8},  // dual output uint8 type
    {"DT_COMPLEX64", DT_COMPLEX64},            // complex64 type
    {"DT_COMPLEX128", DT_COMPLEX128},          // complex128 type
    {"DT_QINT8", DT_QINT8},                    // qint8 type
    {"DT_QINT16", DT_QINT16},                  // qint16 type
    {"DT_QINT32", DT_QINT32},                  // qint32 type
    {"DT_QUINT8", DT_QUINT8},                  // quint8 type
    {"DT_QUINT16", DT_QUINT16},                // quint16 type
    {"DT_RESOURCE", DT_RESOURCE},              // resource type
    {"DT_STRING_REF", DT_STRING_REF},          // string ref type
    {"DT_STRING", DT_STRING},                  // string type
    // add for json input
    {"DT_FLOAT32", DT_FLOAT},
    {"DT_VARIANT", DT_VARIANT},                // dt_variant type
    {"DT_BFLOAT16", DT_BF16},                  // dt_bf16 type
    {"DT_INT4", DT_INT4},                      // dt_int4 type
    {"DT_UINT1", DT_UINT1},                    // dt_uint1 type
    {"DT_INT2", DT_INT2},                      // dt_int2 type
    {"DT_UINT2", DT_UINT2},                    // dt_uint2 type
    {"RESERVED", DT_UNDEFINED},                      // RESERVED will be deserialized to DT_UNDEFINED
};

const std::map<ge::DataType, uint32_t> kDataTypeToLength = {
    {DT_STRING_REF, sizeof(uint64_t) * 2U},
    {DT_STRING, sizeof(uint64_t) * 2U},
};

const std::map<domi::FrameworkType, std::string> kFmkTypeToString = {
    {domi::CAFFE, "caffe"},
    {domi::LUOJIANET, "luojianet_ms"},
    {domi::TENSORFLOW, "tensorflow"},
    {domi::ANDROID_NN, "android_nn"},
    {domi::ONNX, "onnx"},
    {domi::FRAMEWORK_RESERVED, "framework_reserved"},
};

const std::map<domi::ImplyType, std::string> kImplyTypeToString = {
    {domi::ImplyType::BUILDIN, "buildin"},
    {domi::ImplyType::TVM, "tvm"},
    {domi::ImplyType::CUSTOM, "custom"},
    {domi::ImplyType::AI_CPU, "ai_cpu"},
    {domi::ImplyType::CCE, "cce"},
    {domi::ImplyType::GELOCAL, "gelocal"},
    {domi::ImplyType::HCCL, "hccl"},
    {domi::ImplyType::INVALID, "invalid"}
};
}


std::string TypeUtils::ImplyTypeToSerialString(const domi::ImplyType imply_type) {
  const auto it = kImplyTypeToString.find(imply_type);
  if (it != kImplyTypeToString.end()) {
    return it->second;
  } else {
    REPORT_INNER_ERROR("E19999", "ImplyTypeToSerialString: imply_type not support %u",
                       static_cast<uint32_t>(imply_type));
    GELOGE(GRAPH_FAILED, "[Check][Param] ImplyTypeToSerialString: imply_type not support %u",
           static_cast<uint32_t>(imply_type));
    return "UNDEFINED";
  }
}

bool TypeUtils::IsDataTypeValid(const DataType dt) {
  const uint32_t num = static_cast<uint32_t>(dt);
  GE_CHK_BOOL_EXEC((num < DT_MAX),
                   REPORT_INNER_ERROR("E19999", "param dt:%d >= DT_MAX:%d, check invalid", num, DT_MAX);
                   return false, "[Check][Param] The DataType is invalid, dt:%u >= DT_MAX:%d", num, DT_MAX);
  return true;
}

std::string TypeUtils::DataTypeToSerialString(const DataType data_type) {
  const auto it = kDataTypeToStringMap.find(data_type);
  if (it != kDataTypeToStringMap.end()) {
    return it->second;
  } else {
    REPORT_INNER_ERROR("E19999", "DataTypeToSerialString: datatype not support %u", data_type);
    return "UNDEFINED";
  }
}

DataType TypeUtils::SerialStringToDataType(const std::string &str) {
  const auto it = kStringTodataTypeMap.find(str);
  if (it != kStringTodataTypeMap.end()) {
    return it->second;
  } else {
    REPORT_INNER_ERROR("E19999", "SerialStringToDataType: datatype not support %s", str.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] SerialStringToDataType: datatype not support %s", str.c_str());
    return DT_UNDEFINED;
  }
}

bool TypeUtils::IsFormatValid(const Format format) {
  const uint32_t num = static_cast<uint32_t>(GetPrimaryFormat(format));
  GE_CHK_BOOL_EXEC((num <= FORMAT_RESERVED),
                   REPORT_INNER_ERROR("E19999", "The Format is invalid, num:%u > FORMAT_RESERVED:%d",
                                      num, FORMAT_RESERVED);
                   return false,
                   "[Check][Param] The Format is invalid, num:%u > FORMAT_RESERVED:%d", num, FORMAT_RESERVED);
  return true;
}

bool TypeUtils::IsDataTypeValid(std::string dt) {
  (void)transform(dt.begin(), dt.end(), dt.begin(), &::toupper);
  const std::string key = "DT_" + dt;
  const auto it = kStringTodataTypeMap.find(key);
  if (it == kStringTodataTypeMap.end()) {
    return false;
  }
  return true;
}

bool TypeUtils::IsFormatValid(std::string format) {
  (void)transform(format.begin(), format.end(), format.begin(), &::toupper);
  const auto it = kStringToFormatMap.find(format);
  if (it == kStringToFormatMap.end()) {
    return false;
  }
  return true;
}

bool TypeUtils::IsInternalFormat(const Format format) {
  const std::string serial_format = FormatToSerialString(static_cast<Format>(GetPrimaryFormat(format)));
  const auto iter = kInternalFormat.find(serial_format);
  const bool result = (iter == kInternalFormat.end()) ? false : true;
  return result;
}

std::string TypeUtils::FormatToSerialString(const Format format) {
  const auto it = kFormatToStringMap.find(static_cast<Format>(GetPrimaryFormat(format)));
  if (it != kFormatToStringMap.end()) {
    if (HasSubFormat(format)) {
      return it->second + ":" + std::to_string(GetSubFormat(format));
    }
    return it->second;
  } else {
    REPORT_INNER_ERROR("E19999", "Format not support %u", format);
    GELOGE(GRAPH_FAILED, "[Check][Param] Format not support %u", format);
    return "RESERVED";
  }
}

Format TypeUtils::SerialStringToFormat(const std::string &str) {
  std::string primary_format_str = str;
  int32_t sub_format = 0;
  if (SplitFormatFromStr(str, primary_format_str, sub_format) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Split][Format] from %s failed", str.c_str());
    return FORMAT_RESERVED;
  }
  int32_t primary_format;
  const auto it = kStringToFormatMap.find(primary_format_str);
  if (it != kStringToFormatMap.end()) {
    primary_format = it->second;
  } else {
    REPORT_INNER_ERROR("E19999", "Format not support %s", str.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Format not support %s", str.c_str());
    return FORMAT_RESERVED;
  }
  return static_cast<Format>(GetFormatFromSub(primary_format, sub_format));
}

Format TypeUtils::DataFormatToFormat(const std::string &str) {
  std::string primary_format_str = str;
  int32_t sub_format = 0;
  if (SplitFormatFromStr(str, primary_format_str, sub_format) != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Split][Format] from %s failed", str.c_str());
    return FORMAT_RESERVED;
  }
  int32_t primary_format;
  const auto it = kDataFormatMap.find(primary_format_str);
  if (it != kDataFormatMap.end()) {
    primary_format = it->second;
  } else {
    REPORT_INNER_ERROR("E19999", "Format not support %s", str.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] Format not support %s", str.c_str());
    return FORMAT_RESERVED;
  }
  return static_cast<Format>(GetFormatFromSub(primary_format, sub_format));
}

graphStatus TypeUtils::SplitFormatFromStr(const std::string &str,
                                          std::string &primary_format_str, int32_t &sub_format) {
  const size_t split_pos = str.find_first_of(':');
  if (split_pos != std::string::npos) {
    const std::string sub_format_str = str.substr(split_pos + 1U);
    try {
      primary_format_str = str.substr(0U, split_pos);
      if (std::any_of(sub_format_str.cbegin(), sub_format_str.cend(),
                      [](const char_t c) { return !static_cast<bool>(isdigit(c)); })) {
        REPORT_CALL_ERROR("E19999", "sub_format: %s is not digital.", sub_format_str.c_str());
        GELOGE(GRAPH_FAILED, "[Check][Param] sub_format: %s is not digital.", sub_format_str.c_str());
        return GRAPH_FAILED;
      }
      sub_format = std::stoi(sub_format_str);
    } catch (std::invalid_argument &) {
      REPORT_INNER_ERROR("E19999", "sub_format: %s is invalid.", sub_format_str.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] sub_format: %s is invalid.", sub_format_str.c_str());
      return GRAPH_FAILED;
    } catch (std::out_of_range &) {
      REPORT_INNER_ERROR("E19999", "sub_format: %s is out of range.", sub_format_str.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] sub_format: %s is out of range.", sub_format_str.c_str());
      return GRAPH_FAILED;
    } catch (...) {
      REPORT_INNER_ERROR("E19999", "sub_format: %s cannot change to int.", sub_format_str.c_str());
      GELOGE(GRAPH_FAILED, "[Check][Param] sub_format: %s cannot change to int.", sub_format_str.c_str());
      return GRAPH_FAILED;
    }
    if (sub_format > 0xffff) {
      REPORT_INNER_ERROR("E19999", "sub_format: %u is out of range [0, 0xffff].", sub_format);
      GELOGE(GRAPH_FAILED, "[Check][Param] sub_format: %u is out of range [0, 0xffff].", sub_format);
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

Format TypeUtils::DomiFormatToFormat(const domi::domiTensorFormat_t domi_format) {
  const auto it = kDomiFormatToGeFormat.find(domi_format);
  if (it != kDomiFormatToGeFormat.end()) {
    return it->second;
  }
  REPORT_INNER_ERROR("E19999", "do not find domi Format %d from map", domi_format);
  GELOGE(GRAPH_FAILED, "[Check][Param] do not find domi Format %d from map", domi_format);
  return FORMAT_RESERVED;
}

std::string TypeUtils::FmkTypeToSerialString(const domi::FrameworkType fmk_type) {
  const auto it = kFmkTypeToString.find(fmk_type);
  if (it != kFmkTypeToString.end()) {
    return it->second;
  } else {
    GELOGW("[Util][Serialize] Framework type %d not support.", fmk_type);
    return "";
  }
}

bool TypeUtils::GetDataTypeLength(const ge::DataType data_type, uint32_t &length) {
  const auto it = kDataTypeToLength.find(data_type);
  if (it != kDataTypeToLength.end()) {
    length = it->second;
    return true;
  }

  const int32_t size = GetSizeByDataType(data_type);
  if (size > 0) {
    length = static_cast<uint32_t>(size);
    return true;
  } else {
    REPORT_INNER_ERROR("E19999", "data_type not support %d", data_type);
    GELOGE(GRAPH_FAILED, "[Check][Param] data_type not support %d", data_type);
    return false;
  }
}
bool TypeUtils::CheckUint64MulOverflow(const uint64_t a, const uint32_t b) {
  // Not overflow
  if (a == 0U) {
    return false;
  }
  if (b <= (ULLONG_MAX / a)) {
    return false;
  }
  return true;
}
}  // namespace ge
