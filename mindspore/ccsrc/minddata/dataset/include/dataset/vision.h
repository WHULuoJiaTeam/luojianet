/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/api/dual_abi_helper.h"
#include "include/api/status.h"
#include "include/dataset/constants.h"
#include "include/dataset/transforms.h"
#include "include/dataset/vision_lite.h"

namespace mindspore {
namespace dataset {
class TensorOperation;

// Transform operations for performing computer vision.
namespace vision {
/// \brief AdjustGamma TensorTransform.
/// \note Apply gamma correction on input image.
class MS_API AdjustGamma final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] gamma Non negative real number, which makes the output image pixel value
  ///     exponential in relation to the input image pixel value.
  /// \param[in] gain The constant multiplier.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto adjust_gamma_op = vision::AdjustGamma(10.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, adjust_gamma_op},  // operations
  ///                            {"image"});                   // input columns
  /// \endcode
  explicit AdjustGamma(float gamma, float gain = 1);

  /// \brief Destructor.
  ~AdjustGamma() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply AutoAugment data augmentation method.
class MS_API AutoAugment final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] policy An enum for the data auto augmentation policy (default=AutoAugmentPolicy::kImageNet).
  ///     - AutoAugmentPolicy::kImageNet, AutoAugment policy learned on the ImageNet dataset.
  ///     - AutoAugmentPolicy::kCifar10, AutoAugment policy learned on the Cifar10 dataset.
  ///     - AutoAugmentPolicy::kSVHN, AutoAugment policy learned on the SVHN dataset.
  /// \param[in] interpolation An enum for the mode of interpolation (default=InterpolationMode::kNearestNeighbour).
  ///     - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///     - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///     - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///     - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  /// \param[in] fill_value A vector representing the pixel intensity of the borders (default={0, 0, 0}).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto auto_augment_op = vision::AutoAugment(AutoAugmentPolicy::kImageNet,
  ///                                                InterpolationMode::kNearestNeighbour, {0, 0, 0});
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, auto_augment_op}, // operations
  ///                            {"image"});                   // input columns
  /// \endcode
  explicit AutoAugment(AutoAugmentPolicy policy = AutoAugmentPolicy::kImageNet,
                       InterpolationMode interpolation = InterpolationMode::kNearestNeighbour,
                       const std::vector<uint8_t> &fill_value = {0, 0, 0});

  /// \brief Destructor.
  ~AutoAugment() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply automatic contrast on the input image.
class MS_API AutoContrast final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] cutoff Percent of pixels to cut off from the histogram, the valid range of cutoff value is 0 to 50.
  /// \param[in] ignore Pixel values to ignore.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto autocontrast_op = vision::AutoContrast(10.0, {10, 20});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, autocontrast_op},  // operations
  ///                            {"image"});                    // input columns
  /// \endcode
  explicit AutoContrast(float cutoff = 0.0, const std::vector<uint32_t> &ignore = {});

  /// \brief Destructor.
  ~AutoContrast() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief BoundingBoxAugment TensorTransform.
/// \note  Apply a given image transform on a random selection of bounding box regions of a given image.
class MS_API BoundingBoxAugment final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] transform Raw pointer to the TensorTransform operation.
  /// \param[in] ratio Ratio of bounding boxes to apply augmentation on. Range: [0, 1] (default=0.3).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     TensorTransform *rotate_op = new vision::RandomRotation({-180, 180});
  ///     auto bbox_aug_op = vision::BoundingBoxAugment(rotate_op, 0.5);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({bbox_aug_op},       // operations
  ///                            {"image", "bbox"});  // input columns
  /// \endcode
  explicit BoundingBoxAugment(TensorTransform *transform, float ratio = 0.3);

  /// \brief Constructor.
  /// \param[in] transform Smart pointer to the TensorTransform operation.
  /// \param[in] ratio Ratio of bounding boxes where augmentation is applied to. Range: [0, 1] (default=0.3).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     std::shared_ptr<TensorTransform> flip_op = std::make_shared<vision::RandomHorizontalFlip>(0.5);
  ///     std::shared_ptr<TensorTransform> bbox_aug_op = std::make_shared<vision::BoundingBoxAugment>(flip_op, 0.1);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({bbox_aug_op},       // operations
  ///                            {"image", "bbox"});  // input columns
  /// \endcode
  explicit BoundingBoxAugment(const std::shared_ptr<TensorTransform> &transform, float ratio = 0.3);

  /// \brief Constructor.
  /// \param[in] transform Object pointer to the TensorTransform operation.
  /// \param[in] ratio Ratio of bounding boxes where augmentation is applied to. Range: [0, 1] (default=0.3).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     vision::RandomColor random_color_op = vision::RandomColor(0.5, 1.0);
  ///     vision::BoundingBoxAugment bbox_aug_op = vision::BoundingBoxAugment(random_color_op, 0.8);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({bbox_aug_op},       // operations
  ///                            {"image", "bbox"});  // input columns
  /// \endcode
  explicit BoundingBoxAugment(const std::reference_wrapper<TensorTransform> &transform, float ratio = 0.3);

  /// \brief Destructor.
  ~BoundingBoxAugment() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Change the color space of the image.
class MS_API ConvertColor final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] convert_mode The mode of image channel conversion.
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({std::make_shared<vision::Decode>(),
  ///                             std::make_shared<vision::ConvertColor>(ConvertMode::COLOR_BGR2RGB)}, // operations
  ///                            {"image"});                                                           // input columns
  /// \endcode
  explicit ConvertColor(ConvertMode convert_mode);

  /// \brief Destructor.
  ~ConvertColor() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Mask a random section of each image with the corresponding part of another randomly
///     selected image in that batch.
class MS_API CutMixBatch final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] image_batch_format The format of the batch.
  /// \param[in] alpha The hyperparameter of beta distribution (default = 1.0).
  /// \param[in] prob The probability by which CutMix is applied to each image (default = 1.0).
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Batch(5);
  ///     dataset = dataset->Map({std::make_shared<vision::CutMixBatch>(ImageBatchFormat::kNHWC)}, // operations
  ///                            {"image", "label"});                                             // input columns
  /// \endcode
  explicit CutMixBatch(ImageBatchFormat image_batch_format, float alpha = 1.0, float prob = 1.0);

  /// \brief Destructor.
  ~CutMixBatch() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly cut (mask) out a given number of square patches from the input image.
class MS_API CutOut final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] length Integer representing the side length of each square patch.
  /// \param[in] num_patches Integer representing the number of patches to be cut out of an image.
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({std::make_shared<vision::Decode>(),
  ///                             std::make_shared<vision::CutOut>(1, 4)}, // operations
  ///                            {"image"});                               // input columns
  /// \endcode
  explicit CutOut(int32_t length, int32_t num_patches = 1);

  /// \brief Destructor.
  ~CutOut() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply histogram equalization on the input image.
class MS_API Equalize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({std::make_shared<vision::Decode>(),
  ///                             std::make_shared<vision::Equalize>()}, // operations
  ///                            {"image"});                             // input columns
  /// \endcode
  Equalize();

  /// \brief Destructor.
  ~Equalize() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Flip the input image horizontally.
class MS_API HorizontalFlip final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({std::make_shared<vision::Decode>(),
  ///                             std::make_shared<vision::HorizontalFlip>()}, // operations
  ///                            {"image"});                                   // input columns
  /// \endcode
  HorizontalFlip();

  /// \brief Destructor.
  ~HorizontalFlip() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Apply invert on the input image in RGB mode.
class MS_API Invert final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({std::make_shared<vision::Decode>(),
  ///                             std::make_shared<vision::Invert>()}, // operations
  ///                            {"image"});                           // input columns
  /// \endcode
  Invert();

  /// \brief Destructor.
  ~Invert() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Apply MixUp transformation on an input batch of images and labels. The labels must be in
///     one-hot format and Batch must be called before calling this function.
class MS_API MixUpBatch final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] alpha hyperparameter of beta distribution (default = 1.0).
  /// \par Example
  /// \code
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Batch(5);
  ///     dataset = dataset->Map({std::make_shared<vision::MixUpBatch>()}, // operations
  ///                            {"image"});                               // input columns
  /// \endcode
  explicit MixUpBatch(float alpha = 1);

  /// \brief Destructor.
  ~MixUpBatch() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Normalize the input image with respect to mean and standard deviation and pads an extra
///     channel with value zero.
class MS_API NormalizePad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] mean A vector of mean values for each channel, with respect to channel order.
  ///     The mean values must be in range [0.0, 255.0].
  /// \param[in] std A vector of standard deviations for each channel, with respect to channel order.
  ///     The standard deviation values must be in range (0.0, 255.0].
  /// \param[in] dtype The output datatype of Tensor.
  ///     The standard deviation values must be "float32" or "float16"（default = "float32"）.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto normalize_pad_op = vision::NormalizePad({121.0, 115.0, 100.0}, {70.0, 68.0, 71.0});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, normalize_pad_op},  // operations
  ///                            {"image"});                     // input columns
  /// \endcode
  NormalizePad(const std::vector<float> &mean, const std::vector<float> &std, const std::string &dtype = "float32")
      : NormalizePad(mean, std, StringToChar(dtype)) {}

  NormalizePad(const std::vector<float> &mean, const std::vector<float> &std, const std::vector<char> &dtype);

  /// \brief Destructor.
  ~NormalizePad() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Pad the image according to padding parameters.
class MS_API Pad final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] padding A vector representing the number of pixels to pad the image.
  ///    If the vector has one value, it pads all sides of the image with that value.
  ///    If the vector has two values, it pads left and top with the first and
  ///    right and bottom with the second value.
  ///    If the vector has four values, it pads left, top, right, and bottom with
  ///    those values respectively.
  /// \param[in] fill_value A vector representing the pixel intensity of the borders. Only valid if the
  ///    padding_mode is BorderType.kConstant. If 1 value is provided, it is used for all RGB channels.
  ///    If 3 values are provided, it is used to fill R, G, B channels respectively.
  /// \param[in] padding_mode The method of padding (default=BorderType.kConstant).
  ///    Can be any of
  ///    [BorderType.kConstant, BorderType.kEdge, BorderType.kReflect, BorderType.kSymmetric]
  ///    - BorderType.kConstant, means it fills the border with constant values
  ///    - BorderType.kEdge, means it pads with the last value on the edge
  ///    - BorderType.kReflect, means it reflects the values on the edge omitting the last value of edge
  ///    - BorderType.kSymmetric, means it reflects the values on the edge repeating the last value of edge
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto pad_op = vision::Pad({10, 10, 10, 10}, {255, 255, 255});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, pad_op},  // operations
  ///                            {"image"});           // input columns
  /// \endcode
  explicit Pad(const std::vector<int32_t> &padding, const std::vector<uint8_t> &fill_value = {0},
               BorderType padding_mode = BorderType::kConstant);

  /// \brief Destructor.
  ~Pad() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Automatically adjust the contrast of the image with a given probability.
class MS_API RandomAutoContrast final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] cutoff Percent of the lightest and darkest pixels to be cut off from
  ///     the histogram of the input image. The value must be in range of [0.0, 50.0) (default=0.0).
  /// \param[in] ignore The background pixel values to be ignored, each of which must be
  ///     in range of [0, 255] (default={}).
  /// \param[in] prob A float representing the probability of AutoContrast, which must be
  ///     in range of [0, 1] (default=0.5).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_auto_contrast_op = vision::RandomAutoContrast(5.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_auto_contrast_op},  // operations
  ///                            {"image"});                            // input columns
  /// \endcode
  explicit RandomAutoContrast(float cutoff = 0.0, const std::vector<uint32_t> &ignore = {}, float prob = 0.5);

  /// \brief Destructor.
  ~RandomAutoContrast() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly adjust the sharpness of the input image with a given probability.
class MS_API RandomAdjustSharpness final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] degree A float representing sharpness adjustment degree, which must be non negative.
  /// \param[in] prob A float representing the probability of the image being sharpness adjusted, which
  ///     must in range of [0, 1] (default=0.5).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_adjust_sharpness_op = vision::RandomAdjustSharpness(30.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_adjust_sharpness_op},  // operations
  ///                            {"image"});                               // input columns
  /// \endcode
  explicit RandomAdjustSharpness(float degree, float prob = 0.5);

  /// \brief Destructor.
  ~RandomAdjustSharpness() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Blend an image with its grayscale version with random weights
///        t and 1 - t generated from a given range. If the range is trivial
///        then the weights are determinate and t equals to the bound of the interval.
class MS_API RandomColor final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] t_lb Lower bound random weights.
  /// \param[in] t_ub Upper bound random weights.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_color_op = vision::RandomColor(5.0, 50.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_color_op},  // operations
  ///                            {"image"});                    // input columns
  /// \endcode
  RandomColor(float t_lb, float t_ub);

  /// \brief Destructor.
  ~RandomColor() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly adjust the brightness, contrast, saturation, and hue of the input image.
class MS_API RandomColorAdjust final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] brightness Brightness adjustment factor. Must be a vector of one or two values
  ///     if it is a vector of two values it needs to be in the form of [min, max] (Default={1, 1}).
  /// \param[in] contrast Contrast adjustment factor. Must be a vector of one or two values
  ///     if it is a vector of two values, it needs to be in the form of [min, max] (Default={1, 1}).
  /// \param[in] saturation Saturation adjustment factor. Must be a vector of one or two values
  ///     if it is a vector of two values, it needs to be in the form of [min, max] (Default={1, 1}).
  /// \param[in] hue Hue adjustment factor. Must be a vector of one or two values
  ///     if it is a vector of two values, it must be in the form of [min, max] where -0.5 <= min <= max <= 0.5
  ///     (Default={0, 0}).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_color_adjust_op = vision::RandomColorAdjust({1.0, 5.0}, {10.0, 20.0}, {40.0, 40.0});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_color_adjust_op},  // operations
  ///                            {"image"});                           // input columns
  /// \endcode
  explicit RandomColorAdjust(const std::vector<float> &brightness = {1.0, 1.0},
                             const std::vector<float> &contrast = {1.0, 1.0},
                             const std::vector<float> &saturation = {1.0, 1.0},
                             const std::vector<float> &hue = {0.0, 0.0});

  /// \brief Destructor.
  ~RandomColorAdjust() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Crop the input image at a random location.
class MS_API RandomCrop final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If the size is a single value, a squared crop of size (size, size) is returned.
  ///     If the size has 2 values, it should be (height, width).
  /// \param[in] padding A vector representing the number of pixels to pad the image.
  ///    If the vector has one value, it pads all sides of the image with that value.
  ///    If the vector has two values, it pads left and top with the first and
  ///    right and bottom with the second value.
  ///    If the vector has four values, it pads left, top, right, and bottom with
  ///    those values respectively.
  /// \param[in] pad_if_needed A boolean indicating that whether to pad the image
  ///    if either side is smaller than the given output size.
  /// \param[in] fill_value A vector representing the pixel intensity of the borders if the padding_mode is
  ///     BorderType.kConstant. If 1 value is provided, it is used for all RGB channels.
  ///     If 3 values are provided, it is used to fill R, G, B channels respectively.
  /// \param[in] padding_mode The method of padding (default=BorderType::kConstant).It can be any of
  ///     [BorderType::kConstant, BorderType::kEdge, BorderType::kReflect, BorderType::kSymmetric].
  ///   - BorderType::kConstant, Fill the border with constant values.
  ///   - BorderType::kEdge, Fill the border with the last value on the edge.
  ///   - BorderType::kReflect, Reflect the values on the edge omitting the last value of edge.
  ///   - BorderType::kSymmetric, Reflect the values on the edge repeating the last value of edge.
  /// \note If the input image is more than one, then make sure that the image size is the same.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_crop_op = vision::RandomCrop({255, 255}, {10, 10, 10, 10});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_crop_op},  // operations
  ///                            {"image"});                   // input columns
  /// \endcode
  explicit RandomCrop(const std::vector<int32_t> &size, const std::vector<int32_t> &padding = {0, 0, 0, 0},
                      bool pad_if_needed = false, const std::vector<uint8_t> &fill_value = {0, 0, 0},
                      BorderType padding_mode = BorderType::kConstant);

  /// \brief Destructor.
  ~RandomCrop() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Equivalent to RandomResizedCrop TensorTransform, but crop the image before decoding.
class MS_API RandomCropDecodeResize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///               If the size is a single value, a squared crop of size (size, size) is returned.
  ///               If the size has 2 values, it should be (height, width).
  /// \param[in] scale Range [min, max) of respective size of the
  ///               original size to be cropped (default=(0.08, 1.0)).
  /// \param[in] ratio Range [min, max) of aspect ratio to be
  ///               cropped (default=(3. / 4., 4. / 3.)).
  /// \param[in] interpolation An enum for the mode of interpolation.
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  ///   - InterpolationMode::kCubicPil, Interpolation method is bicubic interpolation like implemented in pillow.
  /// \param[in] max_attempts The maximum number of attempts to propose a valid crop_area (default=10).
  ///               If exceeded, fall back to use center_crop instead.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto random_op = vision::RandomCropDecodeResize({255, 255}, {0.1, 0.5});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomCropDecodeResize(const std::vector<int32_t> &size, const std::vector<float> &scale = {0.08, 1.0},
                                  const std::vector<float> &ratio = {3. / 4, 4. / 3},
                                  InterpolationMode interpolation = InterpolationMode::kLinear,
                                  int32_t max_attempts = 10);

  /// \brief Destructor.
  ~RandomCropDecodeResize() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Crop the input image at a random location and adjust bounding boxes accordingly.
///        If the cropped area is out of bbox, the returned bbox will be empty.
class MS_API RandomCropWithBBox final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If the size is a single value, a squared crop of size (size, size) is returned.
  ///     If the size has 2 values, it should be (height, width).
  /// \param[in] padding A vector representing the number of pixels to pad the image
  ///    If the vector has one value, it pads all sides of the image with that value.
  ///    If the vector has two values, it pads left and top with the first and
  ///    right and bottom with the second value.
  ///    If the vector has four values, it pads left, top, right, and bottom with
  ///    those values respectively.
  /// \param[in] pad_if_needed A boolean indicating that whether to pad the image
  ///    if either side is smaller than the given output size.
  /// \param[in] fill_value A vector representing the pixel intensity of the borders. Only valid
  ///    if the padding_mode is BorderType.kConstant. If 1 value is provided, it is used for all
  ///    RGB channels. If 3 values are provided, it is used to fill R, G, B channels respectively.
  /// \param[in] padding_mode The method of padding (default=BorderType::kConstant).It can be any of
  ///     [BorderType::kConstant, BorderType::kEdge, BorderType::kReflect, BorderType::kSymmetric].
  ///   - BorderType::kConstant, Fill the border with constant values.
  ///   - BorderType::kEdge, Fill the border with the last value on the edge.
  ///   - BorderType::kReflect, Reflect the values on the edge omitting the last value of edge.
  ///   - BorderType::kSymmetric, Reflect the values on the edge repeating the last value of edge.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto random_op = vision::RandomCropWithBBox({224, 224}, {0, 0, 0, 0});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_op},             // operations
  ///                            {"image", "bbox"});      // input columns
  /// \endcode
  explicit RandomCropWithBBox(const std::vector<int32_t> &size, const std::vector<int32_t> &padding = {0, 0, 0, 0},
                              bool pad_if_needed = false, const std::vector<uint8_t> &fill_value = {0, 0, 0},
                              BorderType padding_mode = BorderType::kConstant);

  /// \brief Destructor.
  ~RandomCropWithBBox() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly apply histogram equalization on the input image with a given probability.
class MS_API RandomEqualize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of equalization, which
  ///     must be in range of [0, 1] (default=0.5).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomEqualize(0.5);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomEqualize(float prob = 0.5);

  /// \brief Destructor.
  ~RandomEqualize() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly flip the input image horizontally with a given probability.
class MS_API RandomHorizontalFlip final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of flip.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomHorizontalFlip(0.8);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomHorizontalFlip(float prob = 0.5);

  /// \brief Destructor.
  ~RandomHorizontalFlip() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly flip the input image horizontally with a given probability and adjust bounding boxes accordingly.
class MS_API RandomHorizontalFlipWithBBox final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of flip.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto random_op = vision::RandomHorizontalFlipWithBBox(1.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_op},             // operations
  ///                            {"image", "bbox"});      // input columns
  /// \endcode
  explicit RandomHorizontalFlipWithBBox(float prob = 0.5);

  /// \brief Destructor.
  ~RandomHorizontalFlipWithBBox() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly invert the input image with a given probability.
class MS_API RandomInvert final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of the image being inverted, which
  ///     must be in range of [0, 1] (default=0.5).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomInvert(0.8);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomInvert(float prob = 0.5);

  /// \brief Destructor.
  ~RandomInvert() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Add AlexNet-style PCA-based noise to an image.
class MS_API RandomLighting final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] alpha A float representing the intensity of the image (default=0.05).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomLighting(0.1);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomLighting(float alpha = 0.05);

  /// \brief Destructor.
  ~RandomLighting() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Reduce the number of bits for each color channel randomly.
class MS_API RandomPosterize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] bit_range Range of random posterize to compress image.
  ///     uint8_t vector representing the minimum and maximum bit in range of [1,8] (Default={4, 8}).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomPosterize({4, 8});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomPosterize(const std::vector<uint8_t> &bit_range = {4, 8});

  /// \brief Destructor.
  ~RandomPosterize() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Resize the input image using a randomly selected interpolation mode.
class MS_API RandomResize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the resized image.
  ///     If the size is a single value, the smaller edge of the image will be resized to this value with
  ///      the same image aspect ratio. If the size has 2 values, it should be (height, width).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomResize({32, 32});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomResize(const std::vector<int32_t> &size);

  /// \brief Destructor.
  ~RandomResize() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Resize the input image using a randomly selected interpolation mode and adjust
///     bounding boxes accordingly.
class MS_API RandomResizeWithBBox final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the resized image.
  ///     If the size is a single value, the smaller edge of the image will be resized to this value with
  ///      the same image aspect ratio. If the size has 2 values, it should be (height, width).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto random_op = vision::RandomResizeWithBBox({50, 50});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_op},             // operations
  ///                            {"image", "bbox"});      // input columns
  /// \endcode
  explicit RandomResizeWithBBox(const std::vector<int32_t> &size);

  /// \brief Destructor.
  ~RandomResizeWithBBox() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Crop the input image to a random size and aspect ratio.
class MS_API RandomResizedCrop final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If the size is a single value, a squared crop of size (size, size) is returned.
  ///     If the size has 2 values, it should be (height, width).
  /// \param[in] scale Range [min, max) of respective size of the original
  ///     size to be cropped (default=(0.08, 1.0)).
  /// \param[in] ratio Range [min, max) of aspect ratio to be cropped
  ///     (default=(3. / 4., 4. / 3.)).
  /// \param[in] interpolation Image interpolation mode (default=InterpolationMode::kLinear).
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  ///   - InterpolationMode::kCubicPil, Interpolation method is bicubic interpolation like implemented in pillow.
  /// \param[in] max_attempts The maximum number of attempts to propose a valid.
  ///     crop_area (default=10). If exceeded, fall back to use center_crop instead.
  /// \note If the input image is more than one, then make sure that the image size is the same.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomResizedCrop({32, 32}, {0.08, 1.0});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomResizedCrop(const std::vector<int32_t> &size, const std::vector<float> &scale = {0.08, 1.0},
                             const std::vector<float> &ratio = {3. / 4., 4. / 3.},
                             InterpolationMode interpolation = InterpolationMode::kLinear, int32_t max_attempts = 10);

  /// \brief Destructor.
  ~RandomResizedCrop() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Crop the input image to a random size and aspect ratio.
///        If cropped area is out of bbox, the return bbox will be empty.
class MS_API RandomResizedCropWithBBox final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the cropped image.
  ///     If the size is a single value, a squared crop of size (size, size) is returned.
  ///     If the size has 2 values, it should be (height, width).
  /// \param[in] scale Range [min, max) of respective size of the original
  ///     size to be cropped (default=(0.08, 1.0)).
  /// \param[in] ratio Range [min, max) of aspect ratio to be cropped
  ///     (default=(3. / 4., 4. / 3.)).
  /// \param[in] interpolation Image interpolation mode (default=InterpolationMode::kLinear).
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  ///   - InterpolationMode::kCubicPil, Interpolation method is bicubic interpolation like implemented in pillow.
  /// \param[in] max_attempts The maximum number of attempts to propose a valid
  ///     crop_area (default=10). If exceeded, fall back to use center_crop instead.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto random_op = vision::RandomResizedCropWithBBox({50, 50}, {0.05, 0.5}, {0.2, 0.4},
  ///                                                        InterpolationMode::kCubic);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_op},             // operations
  ///                            {"image", "bbox"});      // input columns
  /// \endcode
  explicit RandomResizedCropWithBBox(const std::vector<int32_t> &size, const std::vector<float> &scale = {0.08, 1.0},
                                     const std::vector<float> &ratio = {3. / 4., 4. / 3.},
                                     InterpolationMode interpolation = InterpolationMode::kLinear,
                                     int32_t max_attempts = 10);

  /// \brief Destructor.
  ~RandomResizedCropWithBBox() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Rotate the image according to parameters.
class MS_API RandomRotation final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] degrees A float vector of size 2, representing the starting and ending degrees.
  /// \param[in] resample An enum for the mode of interpolation.
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  ///   - InterpolationMode::kCubicPil, Interpolation method is bicubic interpolation like implemented in pillow.
  /// \param[in] expand A boolean representing whether the image is expanded after rotation.
  /// \param[in] center A float vector of size 2 or empty, representing the x and y center of rotation
  ///     or the center of the image.
  /// \param[in] fill_value A vector representing the value to fill the area outside the transform
  ///    in the output image. If 1 value is provided, it is used for all RGB channels.
  ///    If 3 values are provided, it is used to fill R, G, B channels respectively.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomRotation({30, 60}, InterpolationMode::kNearestNeighbour);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomRotation(const std::vector<float> &degrees,
                          InterpolationMode resample = InterpolationMode::kNearestNeighbour, bool expand = false,
                          const std::vector<float> &center = {}, const std::vector<uint8_t> &fill_value = {0, 0, 0});

  /// \brief Destructor.
  ~RandomRotation() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Choose a random sub-policy from a list to be applied on the input image. A sub-policy is a list of tuples
///     (operation, prob), where operation is a TensorTransform operation and prob is the probability that this
///     operation will be applied. Once a sub-policy is selected, each operation within the sub-policy with be
///     applied in sequence according to its probability.
class MS_API RandomSelectSubpolicy final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] policy Vector of sub-policies to choose from, in which the TensorTransform objects are raw pointers.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto invert_op(new vision::Invert());
  ///     auto equalize_op(new vision::Equalize());
  ///
  ///     std::vector<std::pair<TensorTransform *, double>> policy = {{invert_op, 0.5}, {equalize_op, 0.4}};
  ///     vision::RandomSelectSubpolicy random_select_subpolicy_op = vision::RandomSelectSubpolicy({policy});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_select_subpolicy_op},    // operations
  ///                            {"image"});                      // input columns
  /// \endcode
  explicit RandomSelectSubpolicy(const std::vector<std::vector<std::pair<TensorTransform *, double>>> &policy);

  /// \brief Constructor.
  /// \param[in] policy Vector of sub-policies to choose from, in which the TensorTransform objects are shared pointers.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     std::shared_ptr<TensorTransform> invert_op(new vision::Invert());
  ///     std::shared_ptr<TensorTransform> equalize_op(new vision::Equalize());
  ///     std::shared_ptr<TensorTransform> resize_op(new vision::Resize({15, 15}));
  ///
  ///     auto random_select_subpolicy_op = vision::RandomSelectSubpolicy({
  ///                                          {{invert_op, 0.5}, {equalize_op, 0.4}},
  ///                                          {{resize_op, 0.1}}
  ///                                       });
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_select_subpolicy_op},    // operations
  ///                            {"image"});                      // input columns
  /// \endcode
  explicit RandomSelectSubpolicy(
    const std::vector<std::vector<std::pair<std::shared_ptr<TensorTransform>, double>>> &policy);

  /// \brief Constructor.
  /// \param[in] policy Vector of sub-policies to choose from, in which the TensorTransform objects are object pointers.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     vision::Invert invert_op = vision::Invert();
  ///     vision::Equalize equalize_op = vision::Equalize();
  ///     vision::Resize resize_op = vision::Resize({15, 15});
  ///
  ///     auto random_select_subpolicy_op = vision::RandomSelectSubpolicy({
  ///                                          {{invert_op, 0.5}, {equalize_op, 0.4}},
  ///                                          {{resize_op, 0.1}}
  ///                                       });
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_select_subpolicy_op},    // operations
  ///                            {"image"});                      // input columns
  /// \endcode
  explicit RandomSelectSubpolicy(
    const std::vector<std::vector<std::pair<std::reference_wrapper<TensorTransform>, double>>> &policy);

  /// \brief Destructor.
  ~RandomSelectSubpolicy() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Adjust the sharpness of the input image by a fixed or random degree.
class MS_API RandomSharpness final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] degrees A float vector of size 2, representing the range of random sharpness
  ///     adjustment degrees. It should be in (min, max) format. If min=max, then it is a
  ///     single fixed magnitude operation (default = (0.1, 1.9)).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomSharpness({0.1, 1.5});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomSharpness(const std::vector<float> &degrees = {0.1, 1.9});

  /// \brief Destructor.
  ~RandomSharpness() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Invert pixels randomly within a specified range.
class MS_API RandomSolarize final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] threshold A vector with two elements specifying the pixel range to invert.
  ///     Threshold values should always be in (min, max) format.
  ///     If min=max, it will to invert all pixels above min(max).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomSharpness({0, 255});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomSolarize(const std::vector<uint8_t> &threshold = {0, 255});

  /// \brief Destructor.
  ~RandomSolarize() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly flip the input image vertically with a given probability.
class MS_API RandomVerticalFlip final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of flip.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto random_op = vision::RandomVerticalFlip();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, random_op},  // operations
  ///                            {"image"});              // input columns
  /// \endcode
  explicit RandomVerticalFlip(float prob = 0.5);

  /// \brief Destructor.
  ~RandomVerticalFlip() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Randomly flip the input image vertically with a given probability and adjust bounding boxes accordingly.
class MS_API RandomVerticalFlipWithBBox final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] prob A float representing the probability of flip.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto random_op = vision::RandomVerticalFlipWithBBox();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_op},             // operations
  ///                            {"image", "bbox"});      // input columns
  /// \endcode
  explicit RandomVerticalFlipWithBBox(float prob = 0.5);

  /// \brief Destructor.
  ~RandomVerticalFlipWithBBox() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Rescale the pixel value of input image.
class MS_API Rescale final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] rescale Rescale factor.
  /// \param[in] shift Shift factor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto rescale_op = vision::Rescale(1.0, 0.0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, rescale_op},  // operations
  ///                            {"image"});               // input columns
  /// \endcode
  Rescale(float rescale, float shift);

  /// \brief Destructor.
  ~Rescale() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Resize the input image to the given size and adjust bounding boxes accordingly.
class MS_API ResizeWithBBox final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size The output size of the resized image.
  ///     If the size is an integer, smaller edge of the image will be resized to this value with the same image aspect
  ///     ratio. If the size is a sequence of length 2, it should be (height, width).
  /// \param[in] interpolation An enum for the mode of interpolation (default=InterpolationMode::kLinear).
  ///   - InterpolationMode::kLinear, Interpolation method is blinear interpolation.
  ///   - InterpolationMode::kNearestNeighbour, Interpolation method is nearest-neighbor interpolation.
  ///   - InterpolationMode::kCubic, Interpolation method is bicubic interpolation.
  ///   - InterpolationMode::kArea, Interpolation method is pixel area interpolation.
  ///   - InterpolationMode::kCubicPil, Interpolation method is bicubic interpolation like implemented in pillow.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto random_op = vision::ResizeWithBBox({100, 100}, InterpolationMode::kNearestNeighbour);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({random_op},             // operations
  ///                            {"image", "bbox"});      // input columns
  /// \endcode
  explicit ResizeWithBBox(const std::vector<int32_t> &size,
                          InterpolationMode interpolation = InterpolationMode::kLinear);

  /// \brief Destructor.
  ~ResizeWithBBox() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Change the format of input tensor from 4-channel RGBA to 3-channel BGR.
class MS_API RGBA2BGR final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto rgb2bgr_op = vision::RGBA2BGR();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, rgb2bgr_op},  // operations
  ///                            {"image"});               // input columns
  /// \endcode
  RGBA2BGR();

  /// \brief Destructor.
  ~RGBA2BGR() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Change the input 4 channel RGBA tensor to 3 channel RGB.
class MS_API RGBA2RGB final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto rgba2rgb_op = vision::RGBA2RGB();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, rgba2rgb_op},  // operations
  ///                            {"image"});                // input columns
  /// \endcode
  RGBA2RGB();

  /// \brief Destructor.
  ~RGBA2RGB() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \note Slice the tensor to multiple patches in horizontal and vertical directions.
class MS_API SlicePatches final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] num_height The number of patches in vertical direction (default=1).
  /// \param[in] num_width The number of patches in horizontal direction (default=1).
  /// \param[in] slice_mode An enum for the mode of slice (default=SliceMode::kPad).
  /// \param[in] fill_value A value representing the pixel to fill the padding area in right and
  ///     bottom border if slice_mode is kPad. Then padded tensor could be just sliced to multiple patches (default=0).
  /// \note The usage scenerio is suitable to tensor with large height and width. The tensor will keep the same
  ///     if set both num_height and num_width to 1. And the number of output tensors is equal to num_height*num_width.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto slice_patch_op = vision::SlicePatches(255, 255);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, slice_patch_op},  // operations
  ///                            {"image"});                   // input columns
  /// \endcode
  explicit SlicePatches(int32_t num_height = 1, int32_t num_width = 1, SliceMode slice_mode = SliceMode::kPad,
                        uint8_t fill_value = 0);

  /// \brief Destructor.
  ~SlicePatches() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Decode, randomly crop and resize a JPEG image using the simulation algorithm of
///     Ascend series chip DVPP module. The application scenario is consistent with SoftDvppDecodeResizeJpeg.
///     The input image size should be in range [32*32, 8192*8192].
///     The zoom-out and zoom-in multiples of the image length and width should be in the range [1/32, 16].
///     Only images with an even resolution can be output. The output of odd resolution is not supported.
class MS_API SoftDvppDecodeRandomCropResizeJpeg final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the resized image.
  ///     If the size is a single value, smaller edge of the image will be resized to this value with
  ///     the same image aspect ratio. If the size has 2 values, it should be (height, width).
  /// \param[in] scale Range [min, max) of respective size of the original
  ///     size to be cropped (default=(0.08, 1.0)).
  /// \param[in] ratio Range [min, max) of aspect ratio to be cropped
  ///     (default=(3. / 4., 4. / 3.)).
  /// \param[in] max_attempts The maximum number of attempts to propose a valid
  ///     crop_area (default=10). If exceeded, fall back to use center_crop instead.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto dvpp_op = vision::SoftDvppDecodeRandomCropResizeJpeg({255, 255}, {0.1, 1.0});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({dvpp_op},   // operations
  ///                            {"image"});  // input columns
  /// \endcode
  explicit SoftDvppDecodeRandomCropResizeJpeg(const std::vector<int32_t> &size,
                                              const std::vector<float> &scale = {0.08, 1.0},
                                              const std::vector<float> &ratio = {3. / 4., 4. / 3.},
                                              int32_t max_attempts = 10);

  /// \brief Destructor.
  ~SoftDvppDecodeRandomCropResizeJpeg() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Decode and resize a JPEG image using the simulation algorithm of Ascend series
///     chip DVPP module. It is recommended to use this algorithm in the following scenarios:
///     When training, the DVPP of the Ascend chip is not used,
///     and the DVPP of the Ascend chip is used during inference,
///     and the accuracy of inference is lower than the accuracy of training;
///     and the input image size should be in range [32*32, 8192*8192].
///     The zoom-out and zoom-in multiples of the image length and width should be in the range [1/32, 16].
///     Only images with an even resolution can be output. The output of odd resolution is not supported.
class MS_API SoftDvppDecodeResizeJpeg final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] size A vector representing the output size of the resized image.
  ///     If the size is a single value, smaller edge of the image will be resized to this value with
  ///     the same image aspect ratio. If the size has 2 values, it should be (height, width).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto dvpp_op = vision::SoftDvppDecodeResizeJpeg({255, 255});
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({dvpp_op},    // operations
  ///                            {"image"});   // input columns
  /// \endcode
  explicit SoftDvppDecodeResizeJpeg(const std::vector<int32_t> &size);

  /// \brief Destructor.
  ~SoftDvppDecodeResizeJpeg() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Swap the red and blue channels of the input image.
class MS_API SwapRedBlue final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto swap_red_blue_op = vision::SwapRedBlue();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, swap_red_blue_op},  // operations
  ///                            {"image"});                     // input columns
  /// \endcode
  SwapRedBlue();

  /// \brief Destructor.
  ~SwapRedBlue() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Randomly perform transformations, as selected from input transform list, on the input tensor.
class MS_API UniformAugment final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] transforms Raw pointer to vector of TensorTransform operations.
  /// \param[in] num_ops An integer representing the number of operations to be selected and applied.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto resize_op(new vision::Resize({30, 30}));
  ///     auto random_crop_op(new vision::RandomCrop({28, 28}));
  ///     auto center_crop_op(new vision::CenterCrop({16, 16}));
  ///     auto uniform_op(new vision::UniformAugment({random_crop_op, center_crop_op}, 2));
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({resize_op, uniform_op},  // operations
  ///                            {"image"});               // input columns
  /// \endcode
  explicit UniformAugment(const std::vector<TensorTransform *> &transforms, int32_t num_ops = 2);

  /// \brief Constructor.
  /// \param[in] transforms Smart pointer to vector of TensorTransform operations.
  /// \param[in] num_ops An integer representing the number of operations to be selected and applied.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     std::shared_ptr<TensorTransform> resize_op(new vision::Resize({30, 30}));
  ///     std::shared_ptr<TensorTransform> random_crop_op(new vision::RandomCrop({28, 28}));
  ///     std::shared_ptr<TensorTransform> center_crop_op(new vision::CenterCrop({16, 16}));
  ///     std::shared_ptr<TensorTransform> uniform_op(new vision::UniformAugment({random_crop_op, center_crop_op}, 2));
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({resize_op, uniform_op},  // operations
  ///                            {"image"});               // input columns
  /// \endcode
  explicit UniformAugment(const std::vector<std::shared_ptr<TensorTransform>> &transforms, int32_t num_ops = 2);

  /// \brief Constructor.
  /// \param[in] transforms Object pointer to vector of TensorTransform operations.
  /// \param[in] num_ops An integer representing the number of operations to be selected and applied.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     vision::Resize resize_op = vision::Resize({30, 30});
  ///     vision::RandomCrop random_crop_op = vision::RandomCrop({28, 28});
  ///     vision::CenterCrop center_crop_op = vision::CenterCrop({16, 16});
  ///     vision::UniformAugment uniform_op = vision::UniformAugment({random_crop_op, center_crop_op}, 2);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({resize_op, uniform_op},  // operations
  ///                            {"image"});               // input columns
  /// \endcode
  explicit UniformAugment(const std::vector<std::reference_wrapper<TensorTransform>> &transforms, int32_t num_ops = 2);

  /// \brief Destructor.
  ~UniformAugment() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Flip the input image vertically.
class MS_API VerticalFlip final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto decode_op = vision::Decode();
  ///     auto flip_op = vision::VerticalFlip();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({decode_op, flip_op},  // operations
  ///                            {"image"});            // input columns
  /// \endcode
  VerticalFlip();

  /// \brief Destructor.
  ~VerticalFlip() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_VISION_H_
