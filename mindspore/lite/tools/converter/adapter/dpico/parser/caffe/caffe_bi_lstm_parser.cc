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

#include "parser/caffe/caffe_bi_lstm_parser.h"
#include <memory>
#include <vector>
#include <functional>
#include "ops/custom.h"
#include "common/op_attr.h"

namespace mindspore {
namespace lite {
BaseOperatorPtr CaffeBiLstmParser::Parse(const caffe::LayerParameter &proto, const caffe::LayerParameter &weight) {
  auto prim = std::make_shared<ops::Custom>();
  if (prim == nullptr) {
    MS_LOG(ERROR) << "prim is nullptr.";
    return nullptr;
  }
  prim->set_type("BiLstm");
  if (proto.has_recurrent_param()) {
    const auto &bi_lstm_param = proto.recurrent_param();
    if (bi_lstm_param.has_num_output()) {
      prim->AddAttr(dpico::kNumOutput, api::MakeValue<int64_t>(bi_lstm_param.num_output()));
      prim->AddAttr(dpico::kOutputChannel, api::MakeValue<int64_t>(bi_lstm_param.num_output()));
    }
    if (bi_lstm_param.has_expose_hidden()) {
      prim->AddAttr(dpico::kExposeHidden, api::MakeValue<bool>(bi_lstm_param.expose_hidden()));
    }
  }

  return prim;
}

CaffeNodeRegistrar g_caffeBiLstmParser("BILSTM", new CaffeBiLstmParser());
}  // namespace lite
}  // namespace mindspore
