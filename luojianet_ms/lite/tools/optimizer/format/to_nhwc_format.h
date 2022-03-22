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

#ifndef LUOJIANET_MS_LITE_TOOLS_OPTIMIZER_FORMAT_TO_NHWC_FORMAT_H_
#define LUOJIANET_MS_LITE_TOOLS_OPTIMIZER_FORMAT_TO_NHWC_FORMAT_H_

#include "tools/optimizer/format/to_format_base.h"

namespace luojianet_ms {
namespace opt {
class ToNHWCFormat : public ToFormatBase {
 public:
  explicit ToNHWCFormat(FmkType fmk_type = converter::kFmkTypeMs, bool train_flag = false)
      : ToFormatBase(fmk_type, train_flag, "ToNHWCFormat") {}
  ~ToNHWCFormat() = default;

 private:
  STATUS GetTransNodeFormatType(const CNodePtr &cnode, opt::TransTypePair *trans_info) override;
  STATUS DecideConvWeightSrcAndDstFormat(const CNodePtr &cnode, schema::Format *src_format,
                                         schema::Format *dst_format) override;
};
}  // namespace opt
}  // namespace luojianet_ms
#endif  // LUOJIANET_MS_LITE_TOOLS_OPTIMIZER_FORMAT_TO_NHWC_FORMAT_H_
