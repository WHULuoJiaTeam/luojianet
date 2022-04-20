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
#include "pybind11/pybind11.h"
#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/include/dataset/transforms.h"
#include "minddata/dataset/kernels/ir/vision/adjust_gamma_ir.h"
#include "minddata/dataset/kernels/ir/vision/andwi_ir.h"
#include "minddata/dataset/kernels/ir/vision/auto_augment_ir.h"
#include "minddata/dataset/kernels/ir/vision/auto_contrast_ir.h"
#include "minddata/dataset/kernels/ir/vision/awei_ir.h"
#include "minddata/dataset/kernels/ir/vision/bmi_ir.h"
#include "minddata/dataset/kernels/ir/vision/bounding_box_augment_ir.h"
#include "minddata/dataset/kernels/ir/vision/center_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/ciwi_ir.h"
#include "minddata/dataset/kernels/ir/vision/convert_color_ir.h"
#include "minddata/dataset/kernels/ir/vision/crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/csi_ir.h"
#include "minddata/dataset/kernels/ir/vision/cutmix_batch_ir.h"
#include "minddata/dataset/kernels/ir/vision/cutout_ir.h"
#include "minddata/dataset/kernels/ir/vision/decode_ir.h"
#include "minddata/dataset/kernels/ir/vision/dvi_ir.h"
#include "minddata/dataset/kernels/ir/vision/equalize_ir.h"
#include "minddata/dataset/kernels/ir/vision/evi_ir.h"
#include "minddata/dataset/kernels/ir/vision/ewi_w_ir.h"
#include "minddata/dataset/kernels/ir/vision/ewi_y_ir.h"
#include "minddata/dataset/kernels/ir/vision/fndwi_ir.h"
#include "minddata/dataset/kernels/ir/vision/gaussian_blur_ir.h"
#include "minddata/dataset/kernels/ir/vision/gabor_ir.h"
#include "minddata/dataset/kernels/ir/vision/glcm_ir.h"
#include "minddata/dataset/kernels/ir/vision/gndwi_ir.h"
#include "minddata/dataset/kernels/ir/vision/horizontal_flip_ir.h"
#include "minddata/dataset/kernels/ir/vision/hwc_to_chw_ir.h"
#include "minddata/dataset/kernels/ir/vision/invert_ir.h"
#include "minddata/dataset/kernels/ir/vision/lbp_ir.h"
#include "minddata/dataset/kernels/ir/vision/mbi_ir.h"
#include "minddata/dataset/kernels/ir/vision/mbwi_ir.h"
#include "minddata/dataset/kernels/ir/vision/mciwi_ir.h"
#include "minddata/dataset/kernels/ir/vision/mndwi_ir.h"
#include "minddata/dataset/kernels/ir/vision/mixup_batch_ir.h"
#include "minddata/dataset/kernels/ir/vision/msavi_ir.h"
#include "minddata/dataset/kernels/ir/vision/ndpi_ir.h"
#include "minddata/dataset/kernels/ir/vision/ndvi_ir.h"
#include "minddata/dataset/kernels/ir/vision/ndwi_ir.h"
#include "minddata/dataset/kernels/ir/vision/nwi_ir.h"
#include "minddata/dataset/kernels/ir/vision/normalize_ir.h"
#include "minddata/dataset/kernels/ir/vision/normalize_pad_ir.h"
#include "minddata/dataset/kernels/ir/vision/osavi_ir.h"
#include "minddata/dataset/kernels/ir/vision/pad_ir.h"
#include "minddata/dataset/kernels/ir/vision/psi_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_adjust_sharpness_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_affine_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_auto_contrast_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_color_adjust_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_color_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_decode_resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_crop_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_equalize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_horizontal_flip_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_horizontal_flip_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_invert_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_lighting_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_posterize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resized_crop_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resized_crop_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_resize_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_rotation_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_select_subpolicy_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_sharpness_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_solarize_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_vertical_flip_ir.h"
#include "minddata/dataset/kernels/ir/vision/random_vertical_flip_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/rdvi_ir.h"
#include "minddata/dataset/kernels/ir/vision/rescale_ir.h"
#include "minddata/dataset/kernels/ir/vision/resize_ir.h"
#include "minddata/dataset/kernels/ir/vision/resize_with_bbox_ir.h"
#include "minddata/dataset/kernels/ir/vision/rfdi_ir.h"
#include "minddata/dataset/kernels/ir/vision/rgb_to_bgr_ir.h"
#include "minddata/dataset/kernels/ir/vision/rotate_ir.h"
#include "minddata/dataset/kernels/ir/vision/rvi_ir.h"
#include "minddata/dataset/kernels/ir/vision/rvi_sar_ir.h"
#include "minddata/dataset/kernels/ir/vision/savi_ir.h"
#include "minddata/dataset/kernels/ir/vision/slice_patches_ir.h"
#include "minddata/dataset/kernels/ir/vision/softdvpp_decode_random_crop_resize_jpeg_ir.h"
#include "minddata/dataset/kernels/ir/vision/softdvpp_decode_resize_jpeg_ir.h"
#include "minddata/dataset/kernels/ir/vision/srwi_ir.h"
#include "minddata/dataset/kernels/ir/vision/tvi_ir.h"
#include "minddata/dataset/kernels/ir/vision/uniform_aug_ir.h"
#include "minddata/dataset/kernels/ir/vision/vertical_flip_ir.h"
#include "minddata/dataset/kernels/ir/vision/vsi_ir.h"
#include "minddata/dataset/kernels/ir/vision/wdrvi_ir.h"
#include "minddata/dataset/kernels/ir/vision/wi_f_ir.h"
#include "minddata/dataset/kernels/ir/vision/wi_h_ir.h"
#include "minddata/dataset/kernels/ir/vision/wndwi_ir.h"

namespace luojianet_ms {
namespace dataset {
PYBIND_REGISTER(
  AdjustGammaOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::AdjustGammaOperation, TensorOperation, std::shared_ptr<vision::AdjustGammaOperation>>(
      *m, "AdjustGammaOperation")
      .def(py::init([](float gamma, float gain) {
        auto ajust_gamma = std::make_shared<vision::AdjustGammaOperation>(gamma, gain);
        THROW_IF_ERROR(ajust_gamma->ValidateParams());
        return ajust_gamma;
      }));
  }));

PYBIND_REGISTER(
  AutoAugmentOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::AutoAugmentOperation, TensorOperation, std::shared_ptr<vision::AutoAugmentOperation>>(
      *m, "AutoAugmentOperation")
      .def(
        py::init([](AutoAugmentPolicy policy, InterpolationMode interpolation, const std::vector<uint8_t> &fill_value) {
          auto auto_augment = std::make_shared<vision::AutoAugmentOperation>(policy, interpolation, fill_value);
          THROW_IF_ERROR(auto_augment->ValidateParams());
          return auto_augment;
        }));
  }));

PYBIND_REGISTER(
  AutoContrastOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::AutoContrastOperation, TensorOperation, std::shared_ptr<vision::AutoContrastOperation>>(
      *m, "AutoContrastOperation")
      .def(py::init([](float cutoff, const std::vector<uint32_t> &ignore) {
        auto auto_contrast = std::make_shared<vision::AutoContrastOperation>(cutoff, ignore);
        THROW_IF_ERROR(auto_contrast->ValidateParams());
        return auto_contrast;
      }));
  }));

PYBIND_REGISTER(BoundingBoxAugmentOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::BoundingBoxAugmentOperation, TensorOperation,
                                   std::shared_ptr<vision::BoundingBoxAugmentOperation>>(*m,
                                                                                         "BoundingBoxAugmentOperation")
                    .def(py::init([](const py::object transform, float ratio) {
                      auto bounding_box_augment = std::make_shared<vision::BoundingBoxAugmentOperation>(
                        std::move(toTensorOperation(transform)), ratio);
                      THROW_IF_ERROR(bounding_box_augment->ValidateParams());
                      return bounding_box_augment;
                    }));
                }));

PYBIND_REGISTER(
  CenterCropOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::CenterCropOperation, TensorOperation, std::shared_ptr<vision::CenterCropOperation>>(
      *m, "CenterCropOperation", "Tensor operation to crop and image in the middle. Takes height and width (optional)")
      .def(py::init([](const std::vector<int32_t> &size) {
        auto center_crop = std::make_shared<vision::CenterCropOperation>(size);
        THROW_IF_ERROR(center_crop->ValidateParams());
        return center_crop;
      }));
  }));

PYBIND_REGISTER(
  ConvertColorOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::ConvertColorOperation, TensorOperation, std::shared_ptr<vision::ConvertColorOperation>>(
      *m, "ConvertColorOperation", "Tensor operation to change the color space of the image.")
      .def(py::init([](ConvertMode convert_mode) {
        auto convert = std::make_shared<vision::ConvertColorOperation>(convert_mode);
        THROW_IF_ERROR(convert->ValidateParams());
        return convert;
      }));
  }));

PYBIND_REGISTER(CropOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::CropOperation, TensorOperation, std::shared_ptr<vision::CropOperation>>(
                    *m, "CropOperation", "Tensor operation to crop images")
                    .def(py::init([](std::vector<int32_t> coordinates, const std::vector<int32_t> &size) {
                      // In Python API, the order of coordinates is first top then left, which is different from
                      // those in CropOperation. So we need to swap the coordinates.
                      std::swap(coordinates[0], coordinates[1]);
                      auto crop = std::make_shared<vision::CropOperation>(coordinates, size);
                      THROW_IF_ERROR(crop->ValidateParams());
                      return crop;
                    }));
                }));

PYBIND_REGISTER(
  CutMixBatchOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::CutMixBatchOperation, TensorOperation, std::shared_ptr<vision::CutMixBatchOperation>>(
      *m, "CutMixBatchOperation", "Tensor operation to cutmix a batch of images")
      .def(py::init([](ImageBatchFormat image_batch_format, float alpha, float prob) {
        auto cut_mix_batch = std::make_shared<vision::CutMixBatchOperation>(image_batch_format, alpha, prob);
        THROW_IF_ERROR(cut_mix_batch->ValidateParams());
        return cut_mix_batch;
      }));
  }));

PYBIND_REGISTER(CutOutOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::CutOutOperation, TensorOperation, std::shared_ptr<vision::CutOutOperation>>(
                    *m, "CutOutOperation",
                    "Tensor operation to randomly erase a portion of the image. Takes height and width.")
                    .def(py::init([](int32_t length, int32_t num_patches) {
                      auto cut_out = std::make_shared<vision::CutOutOperation>(length, num_patches);
                      THROW_IF_ERROR(cut_out->ValidateParams());
                      return cut_out;
                    }));
                }));

PYBIND_REGISTER(DecodeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::DecodeOperation, TensorOperation, std::shared_ptr<vision::DecodeOperation>>(
                    *m, "DecodeOperation")
                    .def(py::init([](bool rgb) {
                      auto decode = std::make_shared<vision::DecodeOperation>(rgb);
                      THROW_IF_ERROR(decode->ValidateParams());
                      return decode;
                    }))
                    .def(py::init([](bool rgb) {
                      auto decode = std::make_shared<vision::DecodeOperation>(rgb);
                      THROW_IF_ERROR(decode->ValidateParams());
                      return decode;
                    }));
                }));

PYBIND_REGISTER(EqualizeOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::EqualizeOperation, TensorOperation, std::shared_ptr<vision::EqualizeOperation>>(
                      *m, "EqualizeOperation")
                      .def(py::init([]() {
                        auto equalize = std::make_shared<vision::EqualizeOperation>();
                        THROW_IF_ERROR(equalize->ValidateParams());
                        return equalize;
                      }));
                }));

PYBIND_REGISTER(
  GaussianBlurOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::GaussianBlurOperation, TensorOperation, std::shared_ptr<vision::GaussianBlurOperation>>(
      *m, "GaussianBlurOperation")
      .def(py::init([](std::vector<int32_t> kernel_size, std::vector<float> sigma) {
        auto gaussian_blur = std::make_shared<vision::GaussianBlurOperation>(kernel_size, sigma);
        THROW_IF_ERROR(gaussian_blur->ValidateParams());
        return gaussian_blur;
      }));
  }));

PYBIND_REGISTER(HorizontalFlipOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::HorizontalFlipOperation, TensorOperation,
                                   std::shared_ptr<vision::HorizontalFlipOperation>>(*m, "HorizontalFlipOperation")
                    .def(py::init([]() {
                      auto horizontal_flip = std::make_shared<vision::HorizontalFlipOperation>();
                      THROW_IF_ERROR(horizontal_flip->ValidateParams());
                      return horizontal_flip;
                    }));
                }));

PYBIND_REGISTER(HwcToChwOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::HwcToChwOperation, TensorOperation, std::shared_ptr<vision::HwcToChwOperation>>(
                      *m, "HwcToChwOperation")
                      .def(py::init([]() {
                        auto hwc_to_chw = std::make_shared<vision::HwcToChwOperation>();
                        THROW_IF_ERROR(hwc_to_chw->ValidateParams());
                        return hwc_to_chw;
                      }));
                }));

PYBIND_REGISTER(InvertOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::InvertOperation, TensorOperation, std::shared_ptr<vision::InvertOperation>>(
                    *m, "InvertOperation")
                    .def(py::init([]() {
                      auto invert = std::make_shared<vision::InvertOperation>();
                      THROW_IF_ERROR(invert->ValidateParams());
                      return invert;
                    }));
                }));

PYBIND_REGISTER(
  MixUpBatchOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::MixUpBatchOperation, TensorOperation, std::shared_ptr<vision::MixUpBatchOperation>>(
      *m, "MixUpBatchOperation")
      .def(py::init([](float alpha) {
        auto mix_up_batch = std::make_shared<vision::MixUpBatchOperation>(alpha);
        THROW_IF_ERROR(mix_up_batch->ValidateParams());
        return mix_up_batch;
      }));
  }));

PYBIND_REGISTER(
  NormalizeOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::NormalizeOperation, TensorOperation, std::shared_ptr<vision::NormalizeOperation>>(
      *m, "NormalizeOperation")
      .def(py::init([](const std::vector<float> &mean, const std::vector<float> &std) {
        auto normalize = std::make_shared<vision::NormalizeOperation>(mean, std);
        THROW_IF_ERROR(normalize->ValidateParams());
        return normalize;
      }));
  }));

PYBIND_REGISTER(
  NormalizePadOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::NormalizePadOperation, TensorOperation, std::shared_ptr<vision::NormalizePadOperation>>(
      *m, "NormalizePadOperation")
      .def(py::init([](const std::vector<float> &mean, const std::vector<float> &std, const std::string &dtype) {
        auto normalize_pad = std::make_shared<vision::NormalizePadOperation>(mean, std, dtype);
        THROW_IF_ERROR(normalize_pad->ValidateParams());
        return normalize_pad;
      }));
  }));

PYBIND_REGISTER(PadOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::PadOperation, TensorOperation, std::shared_ptr<vision::PadOperation>>(
                    *m, "PadOperation")
                    .def(py::init([](const std::vector<int32_t> &padding, const std::vector<uint8_t> &fill_value,
                                     BorderType padding_mode) {
                      auto pad = std::make_shared<vision::PadOperation>(padding, fill_value, padding_mode);
                      THROW_IF_ERROR(pad->ValidateParams());
                      return pad;
                    }));
                }));

PYBIND_REGISTER(RandomAdjustSharpnessOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomAdjustSharpnessOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomAdjustSharpnessOperation>>(
                    *m, "RandomAdjustSharpnessOperation")
                    .def(py::init([](float degree, float prob) {
                      auto random_adjust_sharpness =
                        std::make_shared<vision::RandomAdjustSharpnessOperation>(degree, prob);
                      THROW_IF_ERROR(random_adjust_sharpness->ValidateParams());
                      return random_adjust_sharpness;
                    }));
                }));

PYBIND_REGISTER(
  RandomAffineOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomAffineOperation, TensorOperation, std::shared_ptr<vision::RandomAffineOperation>>(
      *m, "RandomAffineOperation", "Tensor operation to apply random affine transformations on an image.")
      .def(py::init([](const std::vector<float_t> &degrees, const std::vector<float_t> &translate_range,
                       const std::vector<float_t> &scale_range, const std::vector<float_t> &shear_ranges,
                       InterpolationMode interpolation, const std::vector<uint8_t> &fill_value) {
        auto random_affine = std::make_shared<vision::RandomAffineOperation>(degrees, translate_range, scale_range,
                                                                             shear_ranges, interpolation, fill_value);
        THROW_IF_ERROR(random_affine->ValidateParams());
        return random_affine;
      }));
  }));

PYBIND_REGISTER(RandomAutoContrastOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomAutoContrastOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomAutoContrastOperation>>(*m,
                                                                                         "RandomAutoContrastOperation")
                    .def(py::init([](float cutoff, const std::vector<uint32_t> &ignore, float prob) {
                      auto random_auto_contrast =
                        std::make_shared<vision::RandomAutoContrastOperation>(cutoff, ignore, prob);
                      THROW_IF_ERROR(random_auto_contrast->ValidateParams());
                      return random_auto_contrast;
                    }));
                }));

PYBIND_REGISTER(RandomColorAdjustOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomColorAdjustOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomColorAdjustOperation>>(*m,
                                                                                        "RandomColorAdjustOperation")
                    .def(py::init([](const std::vector<float> &brightness, const std::vector<float> &contrast,
                                     const std::vector<float> &saturation, const std::vector<float> &hue) {
                      auto random_color_adjust =
                        std::make_shared<vision::RandomColorAdjustOperation>(brightness, contrast, saturation, hue);
                      THROW_IF_ERROR(random_color_adjust->ValidateParams());
                      return random_color_adjust;
                    }));
                }));

PYBIND_REGISTER(
  RandomColorOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomColorOperation, TensorOperation, std::shared_ptr<vision::RandomColorOperation>>(
      *m, "RandomColorOperation")
      .def(py::init([](float t_lb, float t_ub) {
        auto random_color = std::make_shared<vision::RandomColorOperation>(t_lb, t_ub);
        THROW_IF_ERROR(random_color->ValidateParams());
        return random_color;
      }));
  }));

PYBIND_REGISTER(
  RandomCropDecodeResizeOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomCropDecodeResizeOperation, TensorOperation,
                     std::shared_ptr<vision::RandomCropDecodeResizeOperation>>(*m, "RandomCropDecodeResizeOperation")
      .def(py::init([](const std::vector<int32_t> &size, const std::vector<float> &scale,
                       const std::vector<float> &ratio, InterpolationMode interpolation, int32_t max_attempts) {
        auto random_crop_decode_resize =
          std::make_shared<vision::RandomCropDecodeResizeOperation>(size, scale, ratio, interpolation, max_attempts);
        THROW_IF_ERROR(random_crop_decode_resize->ValidateParams());
        return random_crop_decode_resize;
      }));
  }));

PYBIND_REGISTER(
  RandomCropOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomCropOperation, TensorOperation, std::shared_ptr<vision::RandomCropOperation>>(
      *m, "RandomCropOperation")
      .def(py::init([](const std::vector<int32_t> &size, const std::vector<int32_t> &padding, bool pad_if_needed,
                       const std::vector<uint8_t> &fill_value, BorderType padding_mode) {
        auto random_crop =
          std::make_shared<vision::RandomCropOperation>(size, padding, pad_if_needed, fill_value, padding_mode);
        THROW_IF_ERROR(random_crop->ValidateParams());
        return random_crop;
      }));
  }));

PYBIND_REGISTER(RandomCropWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomCropWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomCropWithBBoxOperation>>(*m,
                                                                                         "RandomCropWithBBoxOperation")
                    .def(
                      py::init([](const std::vector<int32_t> &size, const std::vector<int32_t> &padding,
                                  bool pad_if_needed, const std::vector<uint8_t> &fill_value, BorderType padding_mode) {
                        auto random_crop_with_bbox = std::make_shared<vision::RandomCropWithBBoxOperation>(
                          size, padding, pad_if_needed, fill_value, padding_mode);
                        THROW_IF_ERROR(random_crop_with_bbox->ValidateParams());
                        return random_crop_with_bbox;
                      }));
                }));

PYBIND_REGISTER(RandomEqualizeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomEqualizeOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomEqualizeOperation>>(*m, "RandomEqualizeOperation")
                    .def(py::init([](float prob) {
                      auto random_equalize = std::make_shared<vision::RandomEqualizeOperation>(prob);
                      THROW_IF_ERROR(random_equalize->ValidateParams());
                      return random_equalize;
                    }));
                }));

PYBIND_REGISTER(RandomHorizontalFlipOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomHorizontalFlipOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomHorizontalFlipOperation>>(
                    *m, "RandomHorizontalFlipOperation")
                    .def(py::init([](float prob) {
                      auto random_horizontal_flip = std::make_shared<vision::RandomHorizontalFlipOperation>(prob);
                      THROW_IF_ERROR(random_horizontal_flip->ValidateParams());
                      return random_horizontal_flip;
                    }));
                }));

PYBIND_REGISTER(RandomHorizontalFlipWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomHorizontalFlipWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomHorizontalFlipWithBBoxOperation>>(
                    *m, "RandomHorizontalFlipWithBBoxOperation")
                    .def(py::init([](float prob) {
                      auto random_horizontal_flip_with_bbox =
                        std::make_shared<vision::RandomHorizontalFlipWithBBoxOperation>(prob);
                      THROW_IF_ERROR(random_horizontal_flip_with_bbox->ValidateParams());
                      return random_horizontal_flip_with_bbox;
                    }));
                }));

PYBIND_REGISTER(
  RandomInvertOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomInvertOperation, TensorOperation, std::shared_ptr<vision::RandomInvertOperation>>(
      *m, "RandomInvertOperation")
      .def(py::init([](float prob) {
        auto random_invert = std::make_shared<vision::RandomInvertOperation>(prob);
        THROW_IF_ERROR(random_invert->ValidateParams());
        return random_invert;
      }));
  }));

PYBIND_REGISTER(RandomLightingOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomLightingOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomLightingOperation>>(*m, "RandomLightingOperation")
                    .def(py::init([](float alpha) {
                      auto random_lighting = std::make_shared<vision::RandomLightingOperation>(alpha);
                      THROW_IF_ERROR(random_lighting->ValidateParams());
                      return random_lighting;
                    }));
                }));

PYBIND_REGISTER(RandomPosterizeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomPosterizeOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomPosterizeOperation>>(*m, "RandomPosterizeOperation")
                    .def(py::init([](const std::vector<uint8_t> &bit_range) {
                      auto random_posterize = std::make_shared<vision::RandomPosterizeOperation>(bit_range);
                      THROW_IF_ERROR(random_posterize->ValidateParams());
                      return random_posterize;
                    }));
                }));

PYBIND_REGISTER(
  RandomResizedCropOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomResizedCropOperation, TensorOperation,
                     std::shared_ptr<vision::RandomResizedCropOperation>>(*m, "RandomResizedCropOperation")
      .def(py::init([](const std::vector<int32_t> &size, const std::vector<float> &scale,
                       const std::vector<float> &ratio, InterpolationMode interpolation, int32_t max_attempts) {
        auto random_resized_crop =
          std::make_shared<vision::RandomResizedCropOperation>(size, scale, ratio, interpolation, max_attempts);
        THROW_IF_ERROR(random_resized_crop->ValidateParams());
        return random_resized_crop;
      }));
  }));

PYBIND_REGISTER(RandomResizedCropWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomResizedCropWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomResizedCropWithBBoxOperation>>(
                    *m, "RandomResizedCropWithBBoxOperation")
                    .def(py::init([](const std::vector<int32_t> &size, const std::vector<float> &scale,
                                     const std::vector<float> &ratio, InterpolationMode interpolation,
                                     int32_t max_attempts) {
                      auto random_resized_crop_with_bbox = std::make_shared<vision::RandomResizedCropWithBBoxOperation>(
                        size, scale, ratio, interpolation, max_attempts);
                      THROW_IF_ERROR(random_resized_crop_with_bbox->ValidateParams());
                      return random_resized_crop_with_bbox;
                    }));
                }));

PYBIND_REGISTER(
  RandomResizeOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomResizeOperation, TensorOperation, std::shared_ptr<vision::RandomResizeOperation>>(
      *m, "RandomResizeOperation")
      .def(py::init([](const std::vector<int32_t> &size) {
        auto random_resize = std::make_shared<vision::RandomResizeOperation>(size);
        THROW_IF_ERROR(random_resize->ValidateParams());
        return random_resize;
      }));
  }));

PYBIND_REGISTER(RandomResizeWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomResizeWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomResizeWithBBoxOperation>>(
                    *m, "RandomResizeWithBBoxOperation")
                    .def(py::init([](const std::vector<int32_t> &size) {
                      auto random_resize_with_bbox = std::make_shared<vision::RandomResizeWithBBoxOperation>(size);
                      THROW_IF_ERROR(random_resize_with_bbox->ValidateParams());
                      return random_resize_with_bbox;
                    }));
                }));

PYBIND_REGISTER(RandomRotationOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomRotationOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomRotationOperation>>(*m, "RandomRotationOperation")
                    .def(py::init([](std::vector<float> degrees, InterpolationMode interpolation_mode, bool expand,
                                     std::vector<float> center, std::vector<uint8_t> fill_value) {
                      auto random_rotation = std::make_shared<vision::RandomRotationOperation>(
                        degrees, interpolation_mode, expand, center, fill_value);
                      THROW_IF_ERROR(random_rotation->ValidateParams());
                      return random_rotation;
                    }));
                }));

PYBIND_REGISTER(
  RandomSelectSubpolicyOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RandomSelectSubpolicyOperation, TensorOperation,
                     std::shared_ptr<vision::RandomSelectSubpolicyOperation>>(*m, "RandomSelectSubpolicyOperation")
      .def(py::init([](const py::list py_policy) {
        std::vector<std::vector<std::pair<std::shared_ptr<TensorOperation>, double>>> cpp_policy;
        for (auto &py_sub : py_policy) {
          cpp_policy.push_back({});
          for (auto handle : py_sub.cast<py::list>()) {
            py::tuple tp = handle.cast<py::tuple>();
            if (tp.is_none() || tp.size() != 2) {
              THROW_IF_ERROR(Status(StatusCode::kMDUnexpectedError, "Each tuple in subpolicy should be (op, prob)."));
            }
            std::shared_ptr<TensorOperation> t_op;
            if (py::isinstance<TensorOperation>(tp[0])) {
              t_op = (tp[0]).cast<std::shared_ptr<TensorOperation>>();
            } else if (py::isinstance<TensorOp>(tp[0])) {
              t_op = std::make_shared<transforms::PreBuiltOperation>((tp[0]).cast<std::shared_ptr<TensorOp>>());
            } else if (py::isinstance<py::function>(tp[0])) {
              t_op = std::make_shared<transforms::PreBuiltOperation>(
                std::make_shared<PyFuncOp>((tp[0]).cast<py::function>()));
            } else {
              THROW_IF_ERROR(
                Status(StatusCode::kMDUnexpectedError, "op is neither a tensorOp, tensorOperation nor a pyfunc."));
            }
            double prob = (tp[1]).cast<py::float_>();
            if (prob < 0 || prob > 1) {
              THROW_IF_ERROR(Status(StatusCode::kMDUnexpectedError, "prob needs to be with [0,1]."));
            }
            cpp_policy.back().emplace_back(std::make_pair(t_op, prob));
          }
        }
        auto random_select_subpolicy = std::make_shared<vision::RandomSelectSubpolicyOperation>(cpp_policy);
        THROW_IF_ERROR(random_select_subpolicy->ValidateParams());
        return random_select_subpolicy;
      }));
  }));

PYBIND_REGISTER(RandomSharpnessOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomSharpnessOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomSharpnessOperation>>(*m, "RandomSharpnessOperation")
                    .def(py::init([](const std::vector<float> &degrees) {
                      auto random_sharpness = std::make_shared<vision::RandomSharpnessOperation>(degrees);
                      THROW_IF_ERROR(random_sharpness->ValidateParams());
                      return random_sharpness;
                    }));
                }));

PYBIND_REGISTER(RandomSolarizeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomSolarizeOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomSolarizeOperation>>(*m, "RandomSolarizeOperation")
                    .def(py::init([](const std::vector<uint8_t> &threshold) {
                      auto random_solarize = std::make_shared<vision::RandomSolarizeOperation>(threshold);
                      THROW_IF_ERROR(random_solarize->ValidateParams());
                      return random_solarize;
                    }));
                }));

PYBIND_REGISTER(RandomVerticalFlipOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomVerticalFlipOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomVerticalFlipOperation>>(*m,
                                                                                         "RandomVerticalFlipOperation")
                    .def(py::init([](float prob) {
                      auto random_vertical_flip = std::make_shared<vision::RandomVerticalFlipOperation>(prob);
                      THROW_IF_ERROR(random_vertical_flip->ValidateParams());
                      return random_vertical_flip;
                    }));
                }));

PYBIND_REGISTER(RandomVerticalFlipWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RandomVerticalFlipWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::RandomVerticalFlipWithBBoxOperation>>(
                    *m, "RandomVerticalFlipWithBBoxOperation")
                    .def(py::init([](float prob) {
                      auto random_vertical_flip_with_bbox =
                        std::make_shared<vision::RandomVerticalFlipWithBBoxOperation>(prob);
                      THROW_IF_ERROR(random_vertical_flip_with_bbox->ValidateParams());
                      return random_vertical_flip_with_bbox;
                    }));
                }));

PYBIND_REGISTER(RescaleOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::RescaleOperation, TensorOperation, std::shared_ptr<vision::RescaleOperation>>(
                      *m, "RescaleOperation")
                      .def(py::init([](float rescale, float shift) {
                        auto rescale_op = std::make_shared<vision::RescaleOperation>(rescale, shift);
                        THROW_IF_ERROR(rescale_op->ValidateParams());
                        return rescale_op;
                      }));
                }));

PYBIND_REGISTER(ResizeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::ResizeOperation, TensorOperation, std::shared_ptr<vision::ResizeOperation>>(
                    *m, "ResizeOperation")
                    .def(py::init([](const std::vector<int32_t> &size, InterpolationMode interpolation_mode) {
                      auto resize = std::make_shared<vision::ResizeOperation>(size, interpolation_mode);
                      THROW_IF_ERROR(resize->ValidateParams());
                      return resize;
                    }));
                }));

PYBIND_REGISTER(ResizeWithBBoxOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::ResizeWithBBoxOperation, TensorOperation,
                                   std::shared_ptr<vision::ResizeWithBBoxOperation>>(*m, "ResizeWithBBoxOperation")
                    .def(py::init([](const std::vector<int32_t> &size, InterpolationMode interpolation_mode) {
                      auto resize_with_bbox =
                        std::make_shared<vision::ResizeWithBBoxOperation>(size, interpolation_mode);
                      THROW_IF_ERROR(resize_with_bbox->ValidateParams());
                      return resize_with_bbox;
                    }));
                }));

PYBIND_REGISTER(RgbToBgrOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<vision::RgbToBgrOperation, TensorOperation, std::shared_ptr<vision::RgbToBgrOperation>>(
                      *m, "RgbToBgrOperation")
                      .def(py::init([]() {
                        auto rgb2bgr = std::make_shared<vision::RgbToBgrOperation>();
                        THROW_IF_ERROR(rgb2bgr->ValidateParams());
                        return rgb2bgr;
                      }));
                }));

PYBIND_REGISTER(RotateOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::RotateOperation, TensorOperation, std::shared_ptr<vision::RotateOperation>>(
                    *m, "RotateOperation")
                    .def(py::init([](float degrees, InterpolationMode resample, bool expand, std::vector<float> center,
                                     std::vector<uint8_t> fill_value) {
                      auto rotate =
                        std::make_shared<vision::RotateOperation>(degrees, resample, expand, center, fill_value);
                      THROW_IF_ERROR(rotate->ValidateParams());
                      return rotate;
                    }));
                }));

PYBIND_REGISTER(
  SlicePatchesOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::SlicePatchesOperation, TensorOperation, std::shared_ptr<vision::SlicePatchesOperation>>(
      *m, "SlicePatchesOperation")
      .def(py::init([](int32_t num_height, int32_t num_width, SliceMode slice_mode, uint8_t fill_value) {
        auto slice_patches =
          std::make_shared<vision::SlicePatchesOperation>(num_height, num_width, slice_mode, fill_value);
        THROW_IF_ERROR(slice_patches->ValidateParams());
        return slice_patches;
      }));
  }));

PYBIND_REGISTER(SoftDvppDecodeRandomCropResizeJpegOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::SoftDvppDecodeRandomCropResizeJpegOperation, TensorOperation,
                                   std::shared_ptr<vision::SoftDvppDecodeRandomCropResizeJpegOperation>>(
                    *m, "SoftDvppDecodeRandomCropResizeJpegOperation")
                    .def(py::init([](const std::vector<int32_t> &size, const std::vector<float> &scale,
                                     const std::vector<float> &ratio, int32_t max_attempts) {
                      auto soft_dvpp_decode_random_crop_resize_jpeg =
                        std::make_shared<vision::SoftDvppDecodeRandomCropResizeJpegOperation>(size, scale, ratio,
                                                                                              max_attempts);
                      THROW_IF_ERROR(soft_dvpp_decode_random_crop_resize_jpeg->ValidateParams());
                      return soft_dvpp_decode_random_crop_resize_jpeg;
                    }));
                }));

PYBIND_REGISTER(SoftDvppDecodeResizeJpegOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::SoftDvppDecodeResizeJpegOperation, TensorOperation,
                                   std::shared_ptr<vision::SoftDvppDecodeResizeJpegOperation>>(
                    *m, "SoftDvppDecodeResizeJpegOperation")
                    .def(py::init([](const std::vector<int32_t> &size) {
                      auto soft_dvpp_decode_resize_jpeg =
                        std::make_shared<vision::SoftDvppDecodeResizeJpegOperation>(size);
                      THROW_IF_ERROR(soft_dvpp_decode_resize_jpeg->ValidateParams());
                      return soft_dvpp_decode_resize_jpeg;
                    }));
                }));

PYBIND_REGISTER(
  UniformAugOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::UniformAugOperation, TensorOperation, std::shared_ptr<vision::UniformAugOperation>>(
      *m, "UniformAugOperation")
      .def(py::init([](const py::list transforms, int32_t num_ops) {
        auto uniform_aug =
          std::make_shared<vision::UniformAugOperation>(std::move(toTensorOperations(transforms)), num_ops);
        THROW_IF_ERROR(uniform_aug->ValidateParams());
        return uniform_aug;
      }));
  }));

PYBIND_REGISTER(
  VerticalFlipOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::VerticalFlipOperation, TensorOperation, std::shared_ptr<vision::VerticalFlipOperation>>(
      *m, "VerticalFlipOperation")
      .def(py::init([]() {
        auto vertical_flip = std::make_shared<vision::VerticalFlipOperation>();
        THROW_IF_ERROR(vertical_flip->ValidateParams());
        return vertical_flip;
      }));
  }));

//#####################################
//RS index
// ANDWI 
PYBIND_REGISTER(
  ANDWIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::ANDWIOperation, TensorOperation, std::shared_ptr<vision::ANDWIOperation>>(
      *m, "ANDWIOperation")
      .def(py::init([]() {
        auto ANDWI_out = std::make_shared<vision::ANDWIOperation>();
        THROW_IF_ERROR(ANDWI_out->ValidateParams());
        return ANDWI_out;
      }));
  }));
  
// AWEI 
PYBIND_REGISTER(
  AWEIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::AWEIOperation, TensorOperation, std::shared_ptr<vision::AWEIOperation>>(
      *m, "AWEIOperation")
      .def(py::init([]() {
        auto AWEI_out = std::make_shared<vision::AWEIOperation>();
        THROW_IF_ERROR(AWEI_out->ValidateParams());
        return AWEI_out;
      }));
  }));
  
// BMI 
PYBIND_REGISTER(
  BMIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::BMIOperation, TensorOperation, std::shared_ptr<vision::BMIOperation>>(
      *m, "BMIOperation")
      .def(py::init([]() {
        auto BMI_out = std::make_shared<vision::BMIOperation>();
        THROW_IF_ERROR(BMI_out->ValidateParams());
        return BMI_out;
      }));
  }));

// CIWI
PYBIND_REGISTER(CIWIOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::CIWIOperation, TensorOperation, std::shared_ptr<vision::CIWIOperation>>(
                      *m, "CIWIOperation")
                      .def(py::init([](float digital_C) {
                        auto CIWI_op = std::make_shared<vision::CIWIOperation>(digital_C);
                        THROW_IF_ERROR(CIWI_op->ValidateParams());
                        return CIWI_op;
                      }));
                }));
				
// CSI 
PYBIND_REGISTER(
  CSIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::CSIOperation, TensorOperation, std::shared_ptr<vision::CSIOperation>>(
      *m, "CSIOperation")
      .def(py::init([]() {
        auto CSI_out = std::make_shared<vision::CSIOperation>();
        THROW_IF_ERROR(CSI_out->ValidateParams());
        return CSI_out;
      }));
  }));

// EWI_W
PYBIND_REGISTER(EWI_WOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::EWI_WOperation, TensorOperation,
                                   std::shared_ptr<vision::EWI_WOperation>>(*m, "EWI_WOperation")
                    .def(py::init([](float m, float n) {
                      auto EWI_W_out = std::make_shared<vision::EWI_WOperation>(
                        m, n);
                      THROW_IF_ERROR(EWI_W_out->ValidateParams());
                      return EWI_W_out;
                    }));
                }));

// EWI_Y 
PYBIND_REGISTER(
  EWI_YOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::EWI_YOperation, TensorOperation, std::shared_ptr<vision::EWI_YOperation>>(
      *m, "EWI_YOperation")
      .def(py::init([]() {
        auto EWI_Y_out = std::make_shared<vision::EWI_YOperation>();
        THROW_IF_ERROR(EWI_Y_out->ValidateParams());
        return EWI_Y_out;
      }));
  }));
  
// FNDWI
PYBIND_REGISTER(FNDWIOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::FNDWIOperation, TensorOperation,
                                   std::shared_ptr<vision::FNDWIOperation>>(*m, "FNDWIOperation")
                    .def(py::init([](int S, int CNIR) {
                      auto FNDWI_out = std::make_shared<vision::FNDWIOperation>(
                        S, CNIR);
                      THROW_IF_ERROR(FNDWI_out->ValidateParams());
                      return FNDWI_out;
                    }));
                }));

// Gabor
PYBIND_REGISTER(GaborOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::GaborOperation, TensorOperation,
                                   std::shared_ptr<vision::GaborOperation>>(*m, "GaborOperation")
                    .def(py::init([](bool if_opencv_kernal) {
                      auto Gabor_out = std::make_shared<vision::GaborOperation>(if_opencv_kernal);
                      THROW_IF_ERROR(Gabor_out->ValidateParams());
                      return Gabor_out;
                    }));
                }));
				
// GLCM
PYBIND_REGISTER(GLCMOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::GLCMOperation, TensorOperation,
                                   std::shared_ptr<vision::GLCMOperation>>(*m, "GLCMOperation")
                    .def(py::init([](int N) {
                      auto GLCM_out = std::make_shared<vision::GLCMOperation>(N);
                      THROW_IF_ERROR(GLCM_out->ValidateParams());
                      return GLCM_out;
                    }));
                }));
				
// GNDWI
PYBIND_REGISTER(GNDWIOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::GNDWIOperation, TensorOperation,
                                   std::shared_ptr<vision::GNDWIOperation>>(*m, "GNDWIOperation")
                    .def(py::init([]() {
                      auto GNDWI_out = std::make_shared<vision::GNDWIOperation>();
                      THROW_IF_ERROR(GNDWI_out->ValidateParams());
                      return GNDWI_out;
                    }));
                }));

// LBP
PYBIND_REGISTER(LBPOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::LBPOperation, TensorOperation,
                                   std::shared_ptr<vision::LBPOperation>>(*m, "LBPOperation")
                    .def(py::init([](int N) {
                      auto LBP_out = std::make_shared<vision::LBPOperation>(N);
                      THROW_IF_ERROR(LBP_out->ValidateParams());
                      return LBP_out;
                    }));
                }));

// MBWI 
PYBIND_REGISTER(
   MBWIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::MBWIOperation, TensorOperation, std::shared_ptr<vision::MBWIOperation>>(
      *m, "MBWIOperation")
      .def(py::init([]() {
        auto MBWI_out = std::make_shared<vision::MBWIOperation>();
        THROW_IF_ERROR(MBWI_out->ValidateParams());
        return MBWI_out;
      }));
  }));

// MCIWI 
PYBIND_REGISTER(
  MCIWIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::MCIWIOperation, TensorOperation, std::shared_ptr<vision::MCIWIOperation>>(
      *m, "MCIWIOperation")
      .def(py::init([]() {
        auto MCIWI_out = std::make_shared<vision::MCIWIOperation>();
        THROW_IF_ERROR(MCIWI_out->ValidateParams());
        return MCIWI_out;
      }));
  }));

// MNDWI 
PYBIND_REGISTER(
  MNDWIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::MNDWIOperation, TensorOperation, std::shared_ptr<vision::MNDWIOperation>>(
      *m, "MNDWIOperation")
      .def(py::init([]() {
        auto MNDWI_out = std::make_shared<vision::MNDWIOperation>();
        THROW_IF_ERROR(MNDWI_out->ValidateParams());
        return MNDWI_out;
      }));
  }));

// NDPI 
PYBIND_REGISTER(
  NDPIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::NDPIOperation, TensorOperation, std::shared_ptr<vision::NDPIOperation>>(
      *m, "NDPIOperation")
      .def(py::init([]() {
        auto NDPI_out = std::make_shared<vision::NDPIOperation>();
        THROW_IF_ERROR(NDPI_out->ValidateParams());
        return NDPI_out;
      }));
  }));

// NDVI 
PYBIND_REGISTER(
  NDVIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::NDVIOperation, TensorOperation, std::shared_ptr<vision::NDVIOperation>>(
      *m, "NDVIOperation")
      .def(py::init([]() {
        auto NDVI_out = std::make_shared<vision::NDVIOperation>();
        THROW_IF_ERROR(NDVI_out->ValidateParams());
        return NDVI_out;
      }));
  }));
  
// NDWI 
PYBIND_REGISTER(
  NDWIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::NDWIOperation, TensorOperation, std::shared_ptr<vision::NDWIOperation>>(
      *m, "NDWIOperation")
      .def(py::init([]() {
        auto NDWI_out = std::make_shared<vision::NDWIOperation>();
        THROW_IF_ERROR(NDWI_out->ValidateParams());
        return NDWI_out;
      }));
  }));
  
// NWI 
PYBIND_REGISTER(
  NWIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::NWIOperation, TensorOperation, std::shared_ptr<vision::NWIOperation>>(
      *m, "NWIOperation")
      .def(py::init([]() {
        auto NWI_out = std::make_shared<vision::NWIOperation>();
        THROW_IF_ERROR(NWI_out->ValidateParams());
        return NWI_out;
      }));
  }));
  
// PSI 
PYBIND_REGISTER(
  PSIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::PSIOperation, TensorOperation, std::shared_ptr<vision::PSIOperation>>(
      *m, "PSIOperation")
      .def(py::init([]() {
        auto PSI_out = std::make_shared<vision::PSIOperation>();
        THROW_IF_ERROR(PSI_out->ValidateParams());
        return PSI_out;
      }));
  }));

// RVI 
PYBIND_REGISTER(
  RVIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RVIOperation, TensorOperation, std::shared_ptr<vision::RVIOperation>>(
      *m, "RVIOperation")
      .def(py::init([]() {
        auto RVI_out = std::make_shared<vision::RVIOperation>();
        THROW_IF_ERROR(RVI_out->ValidateParams());
        return RVI_out;
      }));
  }));

// DVI 
PYBIND_REGISTER(
  DVIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::DVIOperation, TensorOperation, std::shared_ptr<vision::DVIOperation>>(
      *m, "DVIOperation")
      .def(py::init([]() {
        auto DVI_out = std::make_shared<vision::DVIOperation>();
        THROW_IF_ERROR(DVI_out->ValidateParams());
        return DVI_out;
      }));
  }));
  
// EVI 
PYBIND_REGISTER(
  EVIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::EVIOperation, TensorOperation, std::shared_ptr<vision::EVIOperation>>(
      *m, "EVIOperation")
      .def(py::init([]() {
        auto EVI_out = std::make_shared<vision::EVIOperation>();
        THROW_IF_ERROR(EVI_out->ValidateParams());
        return EVI_out;
      }));
  }));
  
// OSAVI
PYBIND_REGISTER(OSAVIOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::OSAVIOperation, TensorOperation,
                                   std::shared_ptr<vision::OSAVIOperation>>(*m, "OSAVIOperation")
                    .def(py::init([](float theta) {
                      auto OSAVI_out = std::make_shared<vision::OSAVIOperation>(theta);
                      THROW_IF_ERROR(OSAVI_out->ValidateParams());
                      return OSAVI_out;
                    }));
                }));

// MBI
PYBIND_REGISTER(MBIOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::MBIOperation, TensorOperation,
                                   std::shared_ptr<vision::MBIOperation>>(*m, "MBIOperation")
                    .def(py::init([](int32_t s_min, int32_t s_max, int32_t delta_s) {
                      auto MBI_out = std::make_shared<vision::MBIOperation>(
                        s_min, s_max, delta_s);
                      THROW_IF_ERROR(MBI_out->ValidateParams());
                      return MBI_out;
                    }));
                }));

// MSAVI 
PYBIND_REGISTER(
  MSAVIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::MSAVIOperation, TensorOperation, std::shared_ptr<vision::MSAVIOperation>>(
      *m, "MSAVIOperation")
      .def(py::init([]() {
        auto MSAVI_out = std::make_shared<vision::MSAVIOperation>();
        THROW_IF_ERROR(MSAVI_out->ValidateParams());
        return MSAVI_out;
      }));
  }));
  
// RDVI 
PYBIND_REGISTER(
  RDVIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RDVIOperation, TensorOperation, std::shared_ptr<vision::RDVIOperation>>(
      *m, "RDVIOperation")
      .def(py::init([]() {
        auto RDVI_out = std::make_shared<vision::RDVIOperation>();
        THROW_IF_ERROR(RDVI_out->ValidateParams());
        return RDVI_out;
      }));
  }));
  
// RFDI 
PYBIND_REGISTER(
  RFDIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RFDIOperation, TensorOperation, std::shared_ptr<vision::RFDIOperation>>(
      *m, "RFDIOperation")
      .def(py::init([]() {
        auto RFDI_out = std::make_shared<vision::RFDIOperation>();
        THROW_IF_ERROR(RFDI_out->ValidateParams());
        return RFDI_out;
      }));
  }));
  
// RVI_SAR 
PYBIND_REGISTER(
  RVI_SAROperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::RVI_SAROperation, TensorOperation, std::shared_ptr<vision::RVI_SAROperation>>(
      *m, "RVI_SAROperation")
      .def(py::init([]() {
        auto RVI_SAR_out = std::make_shared<vision::RVI_SAROperation>();
        THROW_IF_ERROR(RVI_SAR_out->ValidateParams());
        return RVI_SAR_out;
      }));
  }));

// SAVI
PYBIND_REGISTER(SAVIOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::SAVIOperation, TensorOperation,
                                   std::shared_ptr<vision::SAVIOperation>>(*m, "SAVIOperation")
                    .def(py::init([](float L) {
                      auto SAVI_out = std::make_shared<vision::SAVIOperation>(L);
                      THROW_IF_ERROR(SAVI_out->ValidateParams());
                      return SAVI_out;
                    }));
                }));
				
// SRWI 
PYBIND_REGISTER(
  SRWIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::SRWIOperation, TensorOperation, std::shared_ptr<vision::SRWIOperation>>(
      *m, "SRWIOperation")
      .def(py::init([]() {
        auto SRWI_out = std::make_shared<vision::SRWIOperation>();
        THROW_IF_ERROR(SRWI_out->ValidateParams());
        return SRWI_out;
      }));
  }));

// TVI 
PYBIND_REGISTER(
  TVIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::TVIOperation, TensorOperation, std::shared_ptr<vision::TVIOperation>>(
      *m, "TVIOperation")
      .def(py::init([]() {
        auto TVI_out = std::make_shared<vision::TVIOperation>();
        THROW_IF_ERROR(TVI_out->ValidateParams());
        return TVI_out;
      }));
  }));
  
// VSI 
PYBIND_REGISTER(
  VSIOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::VSIOperation, TensorOperation, std::shared_ptr<vision::VSIOperation>>(
      *m, "VSIOperation")
      .def(py::init([]() {
        auto VSI_out = std::make_shared<vision::VSIOperation>();
        THROW_IF_ERROR(VSI_out->ValidateParams());
        return VSI_out;
      }));
  }));
  
// WDRVI
PYBIND_REGISTER(WDRVIOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::WDRVIOperation, TensorOperation,
                                   std::shared_ptr<vision::WDRVIOperation>>(*m, "WDRVIOperation")
                    .def(py::init([](float alpha) {
                      auto WDRVI_out = std::make_shared<vision::WDRVIOperation>(alpha);
                      THROW_IF_ERROR(WDRVI_out->ValidateParams());
                      return WDRVI_out;
                    }));
                }));
				
// WI_F 
PYBIND_REGISTER(
  WI_FOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::WI_FOperation, TensorOperation, std::shared_ptr<vision::WI_FOperation>>(
      *m, "WI_FOperation")
      .def(py::init([]() {
        auto WI_F_out = std::make_shared<vision::WI_FOperation>();
        THROW_IF_ERROR(WI_F_out->ValidateParams());
        return WI_F_out;
      }));
  }));
  
// WI_H 
PYBIND_REGISTER(
  WI_HOperation, 1, ([](const py::module *m) {
    (void)py::class_<vision::WI_HOperation, TensorOperation, std::shared_ptr<vision::WI_HOperation>>(
      *m, "WI_HOperation")
      .def(py::init([]() {
        auto WI_H_out = std::make_shared<vision::WI_HOperation>();
        THROW_IF_ERROR(WI_H_out->ValidateParams());
        return WI_H_out;
      }));
  }));
  
// WNDWI
PYBIND_REGISTER(WNDWIOperation, 1, ([](const py::module *m) {
                  (void)py::class_<vision::WNDWIOperation, TensorOperation,
                                   std::shared_ptr<vision::WNDWIOperation>>(*m, "WNDWIOperation")
                    .def(py::init([](float alpha) {
                      auto WNDWI_out = std::make_shared<vision::WNDWIOperation>(alpha);
                      THROW_IF_ERROR(WNDWI_out->ValidateParams());
                      return WNDWI_out;
                    }));
                }));

}  // namespace dataset
}  // namespace luojianet_ms
