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

#include "graph/debug/ge_attr_define.h"

namespace ge {
// Public attribute
const std::string ATTR_NAME_OP_FILE_PATH = "_op_file_path";

const std::string ATTR_NAME_FORCE_UNKNOWN_SHAPE = "_force_unknown_shape";

const std::string ATTR_NAME_IS_UNKNOWN_SHAPE = "_is_unknown_shape";

const std::string ATTR_NAME_DYNAMIC_SHAPE_PARTITIONED = "_dynamic_shape_partitioned";

const std::string ATTR_NAME_UNKNOWN_SHAPE_TYPE = "_unknown_shape_type";

const std::string ATTR_NAME_NAME = "name";

const std::string ATTR_NAME_TYPE = "type";

const std::string ATTR_NAME_WEIGHT_NAME = "weight_name";

const std::string ATTR_NAME_IS_QUANTIZE_FACTOR = "quantize_factor";

const std::string ATTR_NAME_ALPHA = "alpha";

const std::string ATTR_NAME_BETA = "beta";

const std::string ATTR_NAME_PADMODE = "pad_mode";

const std::string ATTR_NAME_PADMODES = "padding";

const std::string ATTR_NAME_MODE = "mode";

const std::string ATTR_NAME_FILTER = "filter";

const std::string ATTR_NAME_BIAS = "bias";

const std::string ATTR_NAME_BIAS_TERM = "bias_term";

const std::string ATTR_NAME_HAS_BIAS_VALUE = "has_bias_value";

const std::string ATTR_NAME_PAD = "pad";

const std::string ATTR_NAME_PADS = "pad";

const std::string ATTR_NAME_PAD_SIZE = "pad size";

const std::string ATTR_NAME_PAD_MODE = "pad mode";

const std::string ATTR_NAME_SCALE = "scale";

const std::string ATTR_NAME_WINDOWS = "windows";

const std::string ATTR_NAME_GLOBAL_POOLING = "global_pooling";

const std::string ATTR_NAME_CEIL_MODE = "ceil_mode";

const std::string ATTR_NAME_RELUMODE = "relu_mode";

const std::string ATTR_NAME_STRIDE_SIZE = "stride size";

const std::string ATTR_NAME_RELU_FLAG = "relu_flag";

const std::string ATTR_NAME_ALGO = "algo";

const std::string ATTR_NAME_FORMAT = "format";

const std::string ATTR_NAME_STORAGE_FORMAT = "storage_format";

const std::string ATTR_NAME_ORIGIN_FORMAT_IS_SET = "origin_format_is_set";

const std::string ATTR_NAME_STORAGE_SHAPE = "storage_shape";

const std::string ATTR_NAME_FILTER_FORMAT = "filter_format";

const std::string ATTR_NAME_LRN_K = "lrn_k";

const std::string ATTR_NAME_LRN_NORM_REGION = "lrn_normregion";

const std::string ATTR_NAME_LRN_LOCAL_SIZE = "lrn_localsize";

const std::string ATTR_NAME_LRN_ALPHA = "lrn_alpha";

const std::string ATTR_NAME_LRN_BETA = "lrn_beta";

const std::string ATTR_NAME_AXIS = "axis";
const std::string ATTR_NAME_BROADCAST = "broadcast";

const std::string ATTR_NAME_OUTPUT = "output";
const std::string ATTR_NAME_OUTPUT_NUM = "output_num";
const std::string ATTR_NAME_TIDX = "t_idx";

const std::string ATTR_NAME_TPADDINGS = "t_paddings";
const std::string ATTR_IMG_H = "img_h";
const std::string ATTR_IMG_W = "img_w";
const std::string ATTR_NET_H = "net_h";
const std::string ATTR_NET_W = "net_w";

const std::string ATTR_NAME_TMULTIPLES = "t_multiples";

const std::string ATTR_NAME_MULTIPLES = "multiples";

const std::string ATTR_NAME_T = "T";
const std::string ATTR_NAME_N = "N";

const std::string ATTR_NAME_TSHAPE = "Tshape";
const std::string ATTR_NAME_NAN_OPT = "nan_opt";

const std::string ATTR_NAME_AIPP = "aipp";
const std::string NEW_AIPP_CONV_OP = "new_conv_op_for_aipp";

const std::string ATTR_NAME_AIPP_INPUTS = "_aipp_inputs";
const std::string ATTR_NAME_AIPP_OUTPUTS = "_aipp_outputs";

const std::string ATTR_NAME_INPUT_DIMS = "input_dims";
const std::string ATTR_DYNAMIC_AIPP_INPUT_DIMS = "_dynamic_aipp_input_dims";
const std::string ATTR_DATA_RELATED_AIPP_MODE = "_data_related_aipp_mode";
const std::string ATTR_DATA_AIPP_DATA_NAME_MAP = "_data_aipp_data_name_map";

const std::string ATTR_NAME_GRAPH_HAS_BEEN_ADDED = "_graph_has_been_added";

const std::string ATTR_NAME_SESSION_GRAPH_ID = "_session_graph_id";
const std::string ATTR_NAME_PARENT_GRAPH_NAME = "_parent_graph_name";

const std::string ATTR_NAME_MULTISHAPE_BATCHLIST = "multi_shape_batchlist";
const std::string ATTR_NAME_MULTISHAPE_BATCHLIST_SIZE = "multi_shape_batchlist_size";
const std::string ATTR_MODEL_BATCH_NUM = "batch_num";

const std::string ATTR_NAME_INPUT_FORMAT = "input_format";
const std::string ATTR_NAME_OUTPUT_FORMAT = "output_format";

const std::string ATTR_NAME_FRAMEWORK_NODE_DEF = "node_def";
const std::string ATTR_NAME_FRAMEWORK_OP_DEF = "op_def";
const std::string ATTR_NAME_FRAMEWORK_FWK_TYPE = "framework_type";
const std::string ATTR_NAME_FRAMEWORK_FUNC_DEF = "func_def";
const std::string ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE = "original_type";

const std::string ATTR_NAME_INPUT_TENSOR_DESC = "input_tensor_desc";
const std::string ATTR_NAME_OUTPUT_TENSOR_DESC = "output_tensor_desc";

const std::string ATTR_NAME_INFERRED_FORMAT = "inferred_format";
const std::string ATTR_NAME_PRED_PERMUTE_DELETED = "pred_permute_deleted";
const std::string ATTR_NAME_IGNORE_PRED_FORMAT = "ignore_pred_format";
const std::string ATTR_NAME_WEIGHTS = "value";
const std::string ATTR_NAME_WEIGHTS_DATA = "weights_data";
const std::string ATTR_NAME_BROACAST_REAL_DIM_CNT = "broacast_real_dim_cnt";
const std::string ATTR_NAME_DIM_ALIGN = "dim_align";
const std::string ATTR_NAME_STREAM_LABEL = "_stream_label";
const std::string ATTR_NAME_RTS_LABEL_NODE = "_rts_label_node";
const std::string ATTR_NAME_CONTINUOUS_STREAM_LABEL = "_continuous_stream_label";
const std::string ATTR_NAME_STREAM_CYCLE_EVENT_FLAG = "need_stream_cycle_event";
const std::string ATTR_NAME_RTSWITCH_RECV_EVENT_ID = "rtswitch_event_id";
const std::string ATTR_NAME_AUTOMIC_ADD_START = "automic_add_addr_start";
const std::string ATTR_NAME_AUTOMIC_ADD_MEM_SIZE = "automic_add_mem_size";
const std::string ATTR_NAME_DYNAMIC_OUTPUT_DIMS = "_dynamic_output_dims";
const std::string ATTR_NAME_INPUT_ORIGIN_SIZE = "input_origin_size";

const std::string ATTR_NAME_ROOT_GRAPH_ID = "_root_graph_id";
const std::string ATTR_NAME_ROOT_GRAPH_NAME = "_root_graph_name";
// Identify node connecting to input and output
const std::string ATTR_NAME_NODE_CONNECT_INPUT = "_is_connected_to_data";
const std::string ATTR_NAME_NODE_CONNECT_OUTPUT = "_is_connected_to_netoutput";

// To be deleted
const std::string ATTR_TO_BE_DELETED = "to_be_deleted";
const std::string PERMUTE_RESHAPE_FUSION = "permute_reshape_fusion";
const std::string PERMUTE_RESHAPE_FUSION_CONV_PROPOSAL = "fusion_conv_proposal";
const std::string PERMUTE_RESHAPE_FUSION_CONV_DECODEBBOX = "fusion_conv_decodebbox";
const std::string PERMUTE_RESHAPE_FUSION_BOX_TYPE_NUM = "box_type_num";
const std::string SSD_MBOX_LOC_FUSION = "permute_flatten_fusion";
const std::string SSD_MBOX_CONF_FUSION = "permute_flatten_reshape_flatten_fusion";
const std::string SSD_MBOX_OCR_FUSION = "permute_flatten_ocr_fusion";
const std::string SSD_MBOX_FUSION_BOX_TYPE_NUM = "ssd_mbox_fusion_box_type_num";
const std::string SSD_RESHAPE_SLICE_CONCAT_FUSION = "reshape_slice_concat_fusion";

// Refinedet
const std::string REFINEDET_MBOX_LOC_FUSION = "permute_flatten_fusion";

const std::string REFINEDET_MBOX_CONF_FUSION = "permute_flatten_reshape_flatten_fusion";
const std::string REFINEDET_MBOX_FUSION_BOX_TYPE_NUM = "ssd_mbox_fusion_box_type_num";
const std::string REFINEDET_RESHAPE_SLICE_CONCAT_FUSION = "reshape_slice_concat_fusion";
const std::string SSD_PRIORBOX_CONCAT = "ssd_mbox_conf_priorbox_concat_flag";


// _Arg
const std::string ATTR_NAME_INDEX = "index";
// _RetVal
const std::string RETVAL_ATTR_NAME_INDEX = "retval_index";
// Data
const std::string DATA_ATTR_NAME_DATA_TYPE = "data_type";

// Send
const std::string SEND_ATTR_EVENT_ID = "event_id";

// Recv
const std::string RECV_ATTR_EVENT_ID = "event_id";

// convolution
const std::string ATTR_NAME_COEF = "coef";

const std::string ATTR_NAME_STRIDE = "stride";

const std::string ATTR_NAME_STRIDES = "stride";

const std::string ATTR_NAME_DILATION = "dilation";

const std::string ATTR_NAME_DILATIONS = "dilation";

const std::string CONV_ATTR_NAME_MODE = "mode";

const std::string CONV_ATTR_NAME_ALGO = "algo";

const std::string CONV_ATTR_NAME_GROUP = "group";

const std::string CONV_ATTR_NAME_PAD_MODE = "pad_mode";

const std::string CONV_ATTR_NAME_PAD = "pad";

const std::string CONV_ATTR_NAME_STRIDE = "stride";

const std::string CONV_ATTR_NAME_DILATION = "dilation";

const std::string CONV_ATTR_NAME_NUM_OUTPUT = "num_output";

const std::string CONV_ATTR_NAME_KERNEL = "kernel";

const std::string CONV_ATTR_NAME_FILTER = "filter";

const std::string CONV_ATTR_NAME_BIAS = "bias";

const std::string CONV_ATTR_NAME_RELU_FLAG = "relu_flag";

const std::string CONV_ATTR_NAME_ADJ = "adj";

const std::string CONV_ATTR_NAME_TARGET_SHAPE = "target_shape";

const std::string CONV_ATTR_NAME_BEFORE_PAD = "before_pad";

const std::string CONV_ATTR_NAME_HAS_BIAS = "has_bias";

const std::string NEED_INFER = "isNeedInfer";

// Pooling
const std::string POOLING_ATTR_MODE = "mode";
const std::string POOLING_ATTR_NAN_OPT = "nan_opt";
const std::string POOLING_ATTR_PAD_MODE = "pad_mode";
const std::string POOLING_ATTR_GLOBAL_POOLING = "global_pooling";
const std::string POOLING_ATTR_WINDOW = "window";
const std::string POOLING_ATTR_PAD = "pad";
const std::string POOLING_ATTR_STRIDE = "stride";
const std::string POOLING_ATTR_CEIL_MODE = "ceil_mode";
const std::string POOLING_ATTR_DATA_MODE = "data_mode";
const std::string POOLING_ATTR_BEFORE_PAD = "before_pad";
const std::string POOLING_ATTR_NAME_ALGO = "algo";

// Eltwise
const std::string ELTWISE_ATTR_MODE = "mode";
const std::string ELTWISE_ATTR_COEFF = "coeff";
const std::string ELTWISE_ATTR_WEIGHT = "weight";
const std::string ELTWISE_ATTR_RELU_FLAG = "relu_flag";
const std::string ELTWISE_ATTR_ALPHA = "alpha";
const std::string ELTWISE_ATTR_BETA = "beta";

// BatchNorm
const std::string BATCHNORM_ATTR_MODE = "mode";
const std::string BATCHNORM_ATTR_EPSILON = "epsilon";
const std::string BATCHNORM_ATTR_USE_GLOBAL_STATS = "use_global_stats";
const std::string BATCHNORM_ATTR_MOVING_AVERAGE_FRACTION = "moving_average_fraction";
const std::string BATCHNORM_ATTR_ESTIMATED_MEAN = "estimated_mean";
const std::string BATCHNORM_ATTR_ESTIMATED_VARIANCE = "estimated_variance";
const std::string BATCHNORM_ATTR_SCALE = "scale";
const std::string BATCHNORM_ATTR_BIAS = "bias";
const std::string BATCHNORM_ATTR_DATA_FORMAT = "data_format";
const std::string BATCHNORM_ATTR_IS_TRAINING = "is_training";
const std::string BATCHNORM_ATTR_IS_TRAINING_FUSION = "is_training_fusion";

// huberloss
const std::string HUBER_LOSS_ATTR_DELTA = "delta";

// SSDRealDivTileMul
const std::string SSD_REAL_DIV_TILE_MUL_ATTR_TILE_PARA = "tilepara";

// SSDSumMulRealDivMean
const std::string SSD_SUM_MUL_REALDIV_MEAN_ATTR_REDUCTION_INDICES = "reduction_indices";
const std::string SSD_SUM_MUL_REALDIV_MEAN_ATTR_AXIS = "axis";
const std::string SSD_SUM_MUL_REALDIV_MEAN_ATTR_MEAN_PARA = "mean_para";
const std::string SSD_SUM_MUL_REALDIV_MEAN_ATTR_HAS_SUM = "has_sum";

// ConcatFive2Four
// ConcatFour2Five
const std::string SSD_BOX_TYPE_NUM = "box_type_num";
const std::string SSD_CLASS_NUM = "class_num";
const std::string TRANS_FOR_LOSS_MODE = "trans_for_loss_mode";
const std::string SSD_FEATURE_MAP_SIZE = "feature_map_size";
const std::string SSD_FEATURE_MAP_HIGH = "feature_map_high";
const std::string SSD_FEATURE_MAP_WIDTH = "feature_map_width";

// Scale
const std::string SCALE_ATTR_SCALE = "scale";
const std::string SCALE_ATTR_BIAS = "bias";

// FullConnection
const std::string FULL_CONNECTION_ATTR_FILTER = "filter";
const std::string FULL_CONNECTION_ATTR_BIAS = "bias";
const std::string FULL_CONNECTION_ATTR_NUM_OUTPUT = "num_output";
const std::string FULL_CONNECTION_ATTR_RELU_FLAG = "relu_flag";
const std::string FULL_ATTR_NAME_ALGO = "algo";

// SoftmaxOpParams
const std::string SOFTMAX_ATTR_ALGO = "algo";
const std::string SOFTMAX_ATTR_MODE = "mode";

// SparseSoftmaxCrossEntropy
const std::string SPARSE_SOFTMAX_CROSS_ENTROPY_ATTR_MODE = "cross_entropy_mode";
const std::string SPARSE_SOFTMAX_CROSS_ENTROPY_IS_GRAD = "cross_entropy_is_grad";
// Attr labelSmoothing
const std::string SOFTMAX_CROSS_ENTROPY_LABELSMOOTHING = "labelSmoothing";

// ApplyMomentum
const std::string APPLYMENTUM_ATTR_IS_GRAPH_FUSION = "applymomentum_is_graph_fusion";

// Activation
const std::string ACTIVATION_ATTR_MODE = "mode";
const std::string ACTIVATION_ATTR_COEF = "coef";

// Concat
const std::string CONCAT_ATTR_NAME_AXIS = "axis";

// Const
const std::string CONST_ATTR_NAME_DATA_TRANSTYPE = "data_transtype";
const std::string CONST_ATTR_NAME_OUTPUT_FORMAT = "output_format";
const std::string CONST_ATTR_NAME_OUTPUT_TYPE = "output_type";
const std::string CONST_ATTR_NAME_INPUT = "is_const";

// Roipooling
const std::string ROIPOOLING_ATTR_NAME_POOLED_H = "pooled_h";
const std::string ROIPOOLING_ATTR_NAME_POOLED_W = "pooled_w";
const std::string ROIPOOLING_ATTR_NAME_SPATIAL_SCALE = "spatial_scale";
const std::string ROIPOOLING_ATTR_NAME_RIO_POOLING_MODE = "rio_pooling_mode";
const std::string ROIPOOLING_ATTR_NAME_POOLING_MODE = "pooling_mode";
const std::string ROIPOOLING_ATTR_NAME_SAMPLING_RATIO = "sampling_ratio";

// DetectionOutput
const std::string DETECTIONOUTPUT_ATTR_NUM_CLASSES = "num_classes";
const std::string DETECTIONOUTPUT_ATTR_OCR_NUM_CLASSES = "ocr_num_classes";
const std::string DETECTIONOUTPUT_ATTR_NMS_THRESHOLD = "nms_threshold";
const std::string DETECTIONOUTPUT_ATTR_TOP_K = "top_k";
const std::string DETECTIONOUTPUT_ATTR_CONFIDENCE_THRESHOLD = "confidence_threshold";
const std::string DETECTIONOUTPUT_ATTR_IMG_H = "img_h";
const std::string DETECTIONOUTPUT_ATTR_IMG_W = "img_w";
const std::string DETECTIONOUTPUT_ATTR_BATCH_SIZE = "batch_size";
// Ssd DetectionOutput
const std::string DETECTIONOUTPUT_ATTR_ETA = "eta";
const std::string DETECTIONOUTPUT_ATTR_SHARED_LOCATION = "shared_location";
const std::string DETECTIONOUTPUT_ATTR_BACKGROUND_LABEL_ID = "background_label_id";
const std::string DETECTIONOUTPUT_ATTR_CODE_TYPE = "code_type";
const std::string DETECTIONOUTPUT_ATTR_VARIANCE_ENCODED_IN_TARGET = "variance_encoded_in_target";
const std::string DETECTIONOUTPUT_ATTR_KEEP_TOP_K = "keep_top_k";
// Refinedet DetectionOutput
const std::string DETECTIONOUTPUT_ATTR_OBJECTNESS_SCORE = "objectness_score";
// yolo DetectionOutput
const std::string DETECTIONOUTPUT_ATTR_ClASSES = "classes";
const std::string DETECTIONOUTPUT_ATTR_BIASES = "biases";
const std::string DETECTIONOUTPUT_ATTR_RELATIVE = "relative";
const std::string DETECTIONOUTPUT_ATTR_OBJECTNESS_THRESHOLD = "objectness_threshold";
const std::string DETECTIONOUTPUT_ATTR_CLASS_THRESHOLD = "class_threshold";
const std::string DETECTIONOUTPUT_ATTR_POST_TOP_K = "post_top_k";
const std::string DETECTIONOUTPUT_ATTR_IOU_THRESHOLD_DECAY = "iou_threshold_decay";
const std::string DETECTIONOUTPUT_ATTR_COOR_SCALE_FACTOR = "coor_scale_factor";
const std::string DETECTIONOUTPUT_ATTR_YOLO_VERSION = "yolo_version";

// DetectionPostprocess
const std::string POSTPROCESS_ATTR_NAME_CLS_NUM = "cls_num";
const std::string POSTPROCESS_ATTR_NAME_CONF_THRESH = "conf_thresh";
const std::string POSTPROCESS_ATTR_NAME_NMS_THRESH = "nms_thresh";
const std::string POSTPROCESS_ATTR_POST_NMS_TOPN = "post_nms_topn";
const std::string POSTPROCESS_ATTR_NAME_BBOX_REG_WEIGHT = "bbox_reg_weights";

// Spatialtransfrom
const std::string SPTIALTF_ATTR_NAME_OUTPUT_H = "output_h";
const std::string SPTIALTF_ATTR_NAME_OUTPUT_W = "output_w";
const std::string SPTIALTF_ATTR_NAME_BORDER_VALUE = "border_value";
const std::string SPTIALTF_ATTR_NAME_AFFINE_TRANSFORM = "affine_transform";

// Proposa
const std::string PROPOSAL_ATTR_NAME_FEAT_STRIDE = "feat_stride";
const std::string PROPOSAL_ATTR_NAME_BASE_SIZE = "base_size";
const std::string PROPOSAL_ATTR_NAME_MIN_SIZE = "min_size";
const std::string PROPOSAL_ATTR_NAME_RATIO = "ratio";
const std::string PROPOSAL_ATTR_NAME_SCALE = "scale";
const std::string PROPOSAL_ATTR_NAME_PRE_NMS_TOPN = "pre_nms_topn";
const std::string PROPOSAL_ATTR_NAME_POST_NMS_TOPN = "post_nms_topn";
const std::string PROPOSAL_ATTR_NAME_NMS_THRESH = "nms_thresh";
const std::string PROPOSAL_ATTR_NAME_TOP_SIZE = "top_size";
const std::string PROPOSAL_ATTR_IMG_H = "img_h";
const std::string PROPOSAL_ATTR_IMG_W = "img_w";
// Softmax
const std::string SOFTMAX_ATTR_AXIS = "axis";

// Permute
const std::string PERMUTE_ATTR_ORDER = "order";
const std::string PERMUTE_ATTR_PERM = "perm";

// SSD Normalize
const std::string SSDNORMALIZE_ATTR_ACCROSS_SPATIAL = "across_spatial";
const std::string SSDNORMALIZE_ATTR_CHANNEL_SHARED = "channel_shared";
const std::string SSDNORMALIZE_ATTR_EPS = "eps";

// Flatten
const std::string FLATTEN_ATTR_AXIS = "axis";
const std::string FLATTEN_ATTR_END_AXIS = "end_axis";

// SsdPRIORBOX
const std::string SSD_PRIOR_BOX_ATTR_FLIP = "flip";
const std::string SSD_PRIOR_BOX_ATTR_CLIP = "clip";
const std::string SSD_PRIOR_BOX_ATTR_IMG_H = "img_h";
const std::string SSD_PRIOR_BOX_ATTR_IMG_W = "img_w";
const std::string SSD_PRIOR_BOX_ATTR_STEP_H = "step_h";
const std::string SSD_PRIOR_BOX_ATTR_STEP_W = "step_w";
const std::string SSD_PRIOR_BOX_ATTR_OFFSET = "offset";
const std::string SSD_PRIOR_BOX_ATTR_MIN_SIZE = "min_size";
const std::string SSD_PRIOR_BOX_ATTR_MAX_SIZE = "max_size";
const std::string SSD_PRIOR_BOX_ATTR_MIN_SIZE_NUM = "min_size_num";
const std::string SSD_PRIOR_BOX_ATTR_MAX_SIZE_NUM = "max_size_num";
const std::string SSD_PRIOR_BOX_ATTR_ASPECT_RATIO = "aspect_ratio";
const std::string SSD_PRIOR_BOX_ATTR_ASPECT_RATIO_NUM = "aspect_ratio_num";
const std::string SSD_PRIOR_BOX_ATTR_VARIANCE = "variance";
const std::string SSD_PRIOR_BOX_ATTR_VARIANCE_NUM = "variance_num";

//  RefinedetDetectionOutput
const std::string REFINEDET_PRIOR_BOX_ATTR_VARIANCE_NUM = "variance_num";
const std::string REFINEDET_PRIOR_BOX_ATTR_VARIANCE = "variance";

// PRelu
const std::string PRELU_ATTR_CHANNEL_SHARED = "channel_shared";

// Psroi pooling
const std::string PSROIPOOLING_ATTR_SPATIAL_SCALE = "spatial_scale";
const std::string PSROIPOOLING_ATTR_OUTPUT_DIM = "output_dim";
const std::string PSROIPOOLING_ATTR_GROUP_SIZE = "group_size";

// Power
const std::string POWER_ATTR_NAME_POWER = "power";
const std::string POWER_ATTR_NAME_SCALE = "scale";
const std::string POWER_ATTR_NAME_SHIFT = "shift";

// log
const std::string LOG_ATTR_NAME_SCALE = "scale";
const std::string LOG_ATTR_NAME_SHIFT = "shift";
const std::string LOG_ATTR_NAME_BASE = "base";
// Pack
const std::string PACK_ATTR_NAME_NUM = "N";

// Unpack
const std::string UNPACK_ATTR_NAME_NUM = "num";
const std::string DYNAMIC_STITCH_ATTR_NAME_NUM = "DynamicStitchN_";
// Gathernd
const std::string GATHERND_ATTR_NAME_TINDICES = "Tindices";
const std::string GATHERND_ATTR_NAME_TPARAMS = "Tparams";

// Argmax
const std::string ARGMAX_ATTR_NAME_TOPK = "topk";
const std::string ARGMAX_ATTR_NAME_REDUCESIZE = "reduce_size";
const std::string ARGMAX_ATTR_NAME_REDUCESTRIDE = "reduce_stride";
const std::string ARGMAX_ATTR_NAME_OUTMAX = "outmaxval";
const std::string ARGMAX_ATTR_NAME_AXIS = "axis";
const std::string ARGMAX_ATTR_NAME_AXISTYPE = "axis_type";
const std::string ARGMAX_ATTR_NAME_KEEPDIMS = "keep_dims";

// upsample
const std::string UPSAMPLE_ATTR_NAME_SCALE_H = "scale_h";
const std::string UPSAMPLE_ATTR_NAME_SCALE_W = "scale_w";

// Relu
const std::string ATTR_NAME_NEGATIVE_SLOPE = "negative_slope";

// FreeSpaceExtract
const std::string FREESPACEEXTRACT_ATTR_NAME_ORG_HEIGHT = "org_height";

// Split
const std::string SPLIT_ATTR_NAME_SLICE_POINT = "slice_point";
const std::string SPLIT_ATTR_NAME_SIZE_SPLIT = "size_split";
const std::string SPLIT_ATTR_NAME_NUM_SPLIT = "num_split";

// Tvm
const std::string TVM_ATTR_NAME_MAGIC = "tvm_magic";
const std::string TVM_ATTR_NAME_BLOCKDIM = "tvm_blockdim";
const std::string TVM_ATTR_NAME_METADATA = "tvm_metadata";
const std::string TVM_ATTR_NAME_WORKSPACE_TYPE = "tvm_workspace_type";

// Ffts Tvm
const std::string TVM_ATTR_NAME_THREAD_MAGIC = "_thread_tvm_magic";
const std::string TVM_ATTR_NAME_THREAD_BLOCKDIM = "_thread_tvm_blockdim";
const std::string TVM_ATTR_NAME_THREAD_METADATA = "_thread_tvm_metadata";
const std::string TVM_ATTR_NAME_THREAD_WORKSPACE_TYPE = "_thread_tvm_workspace_type";
const std::string TVM_ATTR_NAME_THREAD_N_BATCH_SPLIT = "_thread_is_n_batch_split";

const std::string ATTR_NAME_THREAD_TBE_KERNEL_BUFFER = "_thread_tbe_kernel_buffer";
const std::string ATTR_NAME_THREAD_TBE_KERNEL_NAME = "_thread_tbe_kernel_name";

// Squeeze
const std::string SQUEEZE_ATTR_AXIS = "axis";
const std::string SQUEEZE_ATTR_DIMS = "squeeze_dims";
const std::string SQUEEZE_OP_NAME = "Squeeze";

// Stride slice
const std::string STRIDE_SLICE_ATTR_BEGIN_MASK = "begin_mask";
const std::string STRIDE_SLICE_ATTR_END_MASK = "end_mask";
const std::string STRIDE_SLICE_ATTR_ELLIPSIS_MASK = "ellipsis_mask";
const std::string STRIDE_SLICE_ATTR_NEW_AXIS_MASK = "new_axis_mask";
const std::string STRIDE_SLICE_ATTR_SHRINK_AXIS_MASK = "shrink_axis_mask";

// Slice
const std::string SLICE_ATTR_NAME_BEGINS = "begins";
const std::string SLICE_ATTR_NAME_SIZES = "sizes";

// Roialign
const std::string ROIALIGN_ATTR_SPATIAL_SCALE = "spatial_scale";
const std::string ROIALIGN_ATTR_SAMPLING_RATIO = "sampling_ratio";
const std::string ROIALIGN_ATTR_NAME_POOLED_H = "pooled_h";
const std::string ROIALIGN_ATTR_NAME_POOLED_W = "pooled_w";

// Generate_rpn_proposal
const std::string GENERATE_RPN_PROPOSAL_ATTR_PRE_NMS_TOPK = "pre_nms_topk";
const std::string GENERATE_RPN_PROPOSAL_ATTR_POST_NMS_TOPK = "post_nms_topk";
const std::string GENERATE_RPN_PROPOSAL_ATTR_RPN_MINI_SIZE = "rpn_mini_size";
const std::string GENERATE_RPN_PROPOSAL_ATTR_RPN_PROPOSAL_NMS_THRESH = "rpn_proposal_nms_thresh";
const std::string GENERATE_RPN_PROPOSAL_ATTR_RPN_PROPOSAL_FILTER_THRESH = "rpn_proposal_filter_thresh";
// Decode_bbox
const std::string DECODE_BBOX_ATTR_DECODECLIP = "decodeClip";

// Cast
const std::string CAST_ATTR_DSTT = "DstT";
const std::string CAST_ATTR_SRCT = "SrcT";
const std::string CAST_ATTR_DST_TYPE = "dst_type";
const std::string CAST_ATTR_TRUNCATE = "truncate";

// Fastrcnnn predications
const std::string FASTRCNN_PREDICTIONS_ATTR_TOPK = "fsr_topk";
const std::string FASTRCNN_PREDICTIONS_ATTR_SCORE_THRESHOLD = "fsr_score_thres";
const std::string FASTRCNN_PREDICTIONS_ATTR_NMS_THRESHOLD = "fsr_nms_thres";
const std::string FASTRCNN_PREDICTIONS_ATTR_NUM_CLASSES = "fsr_num_classes";

// REORG
const std::string REORG_ATTR_STRIDE = "stride";
const std::string REORG_ATTR_REVERSE = "reverse";

// MERGE
const std::string MERGE_DEAD_INDEX = "merge_dead_index";
const std::string MERGE_PRENODE_FLAG = "merge_prenode_flag";
const std::string TO_BE_OUTPUT = "to_be_output";

// ENTER
const std::string ENTER_ATTR_FRAME_NAME = "frame_name";
const std::string ENTER_ATTR_CONSTANT_FLAG = "is_constant";

// Concatv2
const std::string CONCAT_V2_ATTR_TIDX = "Tidx";
const std::string CONCAT_V2_ATTR_N = "N";
// SUM
const std::string SUM_ATTR_TIDX = "Tidx";
const std::string SUM_ATTR_AXIS = "axis";
const std::string SUM_ATTR_KEEP_DIMS = "keep_dims";

// ResizeBilinear
const std::string RESIZE_BILINEAR_ATTR_MODE = "mode";
const std::string RESIZE_BILINEAR_ATTR_ALIGN_CORNERS = "align_corners";
const std::string RESIZE_BILINEAR_ATTR_HEIGHT = "height";
const std::string RESIZE_BILINEAR_ATTR_WIDTH = "width";
const std::string RESIZE_BILINEAR_ATTR_ZOOM_FACTOR = "zoom_factor";
const std::string RESIZE_BILINEAR_ATTR_SHRINK_FACTOR = "shrink_factor";
const std::string RESIZE_BILINEAR_ATTR_PAD_BEGIN = "pad_begin";
const std::string RESIZE_BILINEAR_ATTR_PAD_END = "pad_end";
const std::string RESIZE_BILINEAR_ATTR_ALPHA = "alpha";
const std::string RESIZE_BILINEAR_ATTR_BETA = "beta";

// RetinaNet
const std::string RETINANET_FILTER_BACKGROUND_TRUE = "retina_conv_filter_background";
const std::string RETINANET_ANCHOR_FUSION = "retina_anchor_fusion";

// MatMul
const std::string MATMUL_TRANSPOSE_X = "transposeX";
const std::string MATMUL_TRANSPOSE_W = "transposeW";
const std::string MATMUL_HAS_BIAS = "has_bias";
const std::string MATMUL_ATTR_IS_TRAINING = "matmul_is_training";

// Flatten
const std::string FLATTEN_START_AXIS = "start_axis";
const std::string FLATTEN_END_AXIS = "end_axis";

// Reshape
const std::string RESHAPE_ATTR_AXIS = "axis";
const std::string RESHAPE_ATTR_NUM_AXES = "num_axes";
const std::string RESHAPE_ATTR_FORMAT = "format";
const std::string RESHAPE_ATTR_SHAPE = "shape";
const std::string RESHAPE_ATTR_ALPHA = "alpha";
const std::string RESHAPE_ATTR_BETA = "beta";

// Frameoworkop
const std::string T_IN_DATATYPE = "t_in_datatype";
const std::string T_OUT_DATATYPE = "t_out_datatype";
const std::string ATTR_NAME_OUT_N = "out_n";
const std::string ATTR_NAME_OUT_C = "out_c";
const std::string ATTR_NAME_OUT_H = "out_h";
const std::string ATTR_NAME_OUT_W = "out_w";
const std::string ATTR_PAD_DEPTH_CONV = "pad_depth_conv";
const std::string ATTR_PAD_CONV = "pad_conv";

const std::string ATTR_NAME_BEFORE_PAD = "before_pad";
const std::string ANN_MEAN_KEEPDIMS = "AnnMeanKeepDims";
const std::string PAD_ATTR_PADDINGDS = "paddings";
const std::string PAD_ATTR_CONSTANT_VALUE = "padvalue";

// ConvGradFilter
const std::string CONV_GRAD_FILTER_OUTPUT_SHAPE = "conv_grad_filter_output_shape";
// ConvGradInput
const std::string CONV_GRAD_INPUT_OUTPUT_SHAPE = "conv_grad_input_output_shape";

// Rnn
const std::string RNN_MODE_STATIC = "rnn_static";
const std::string MUTI_RNN = "multi_rnn";
const std::string CNN_RNN = "cnn_rnn";
const std::string RNN_MODE_ = "rnn_";


const std::string CELL_MODE = "mode";
const std::string LSTM_CELL = "lstm_cell";
const std::string GRU_CELL = "gru_cell";
const std::string RNN_HT = "ht";
const std::string RNN_XT_HT = "xt_ht";
const std::string RNN_BATCH_SIZE = "batch_size";
const std::string LSTM_CELL_CLIP = "lstm_cell_clip";
const std::string LSTM_PROJ_CLIP = "lstm_proj_clip";
const std::string LSTM_ACTIVATE = "lstm_activate";
const std::string LSTM_OUT_MAP = "lstm_out_map";
const std::string LSTM_OUT_MODE = "lstm_out_mode";
const std::string LSTM_STATE_OUT_MODE = "lstm_state_out_mode";
const std::string LSTM_TIME_MAJOR = "lstm_time_major";
const std::string LSTM_IS_INPUT_PRE_PROCESS = "lstm_is_input_pre_process";

// Upsample
const std::string UPSAMPLE_ATTR_NAME_SCALE = "scale";

// PadV2
const std::string PADV2_ATTR_NAME_MODE = "mode";
const std::string PADV2_ATTR_NAME_PADS = "paddings";
const std::string PADV2_ATTR_NAME_T = "T";
const std::string PADV2_ATTR_NAME_PAD_FORMAT = "pad_format";
const std::string PADV2_ATTR_NAME_CONST_VALUE = "const_value";

// MirrorPad
const std::string MIRRORPAD_ATTR_NAME_MODE = "mode";
const std::string MIRRORPAD_ATTR_NAME_PADS = "paddings";
const std::string MIRRORPAD_ATTR_NAME_PAD_FORMAT = "pad_format";
const std::string MIRRORPAD_ATTR_NAME_CONST_VALUE = "const_value";
// Filler
const std::string FILLER_TYPE = "filler_type";
const std::string FILLER_VALUE = "filler_value";

// Shufflechannel
const std::string SHUFFLE_CHANNEL_GROUP = "group";

// TopKV2
const std::string TOPKV2_ATTR_K = "k";

// Calibaration
const std::string STRIDE_H_INDEX = "STRIDE_H_INDEX";
const std::string STRIDE_W_INDEX = "STRIDE_W_INDEX";
const std::string PAD_TOP_INDEX = "PAD_TOP_INDEX";
const std::string PAD_BOTTOM_INDEX = "PAD_BOTTOM_INDEX";
const std::string PAD_RIGHT_INDEX = "PAD_RIGHT_INDEX";
const std::string PAD_LEFT_INDEX = "PAD_LEFT_INDEX";
const std::string QUANTIZE_ALGO_ATTR = "quantize_algo";
const std::string SCALE_TYPE_ATTR = "scale_type";

const std::string QUANTIZE_SCALE_MODE = "quantize_scale_mode";
const std::string QUANTIZE_SCALE_VALUE = "quantize_scale_value";
const std::string QUANTIZE_SCALE_OFFSET = "quantize_scale_offset";
const std::string QUANTIZE_OFFSET_DATA_VALUE = "quantize_offset_data_value";
const std::string QUANTIZE_OFFSET_DATA_OFFSET = "quantize_offset_data_offset";
const std::string QUANTIZE_OFFSET_WEIGHT_VALUE = "quantize_offset_weight_value";
const std::string QUANTIZE_OFFSET_WEIGHT_OFFSET = "quantize_offset_weight_offset";
const std::string QUANTIZE_OFFSET_PAD_VALUE = "quantize_offset_pad_value";
const std::string QUANTIZE_OFFSET_PAD_OFFSET = "quantize_offset_pad_offset";

const std::string DEQUANTIZE_SCALE_MODE = "dequantize_scale_mode";
const std::string DEQUANTIZE_SCALE_VALUE = "dequantize_scale_value";
const std::string DEQUANTIZE_SCALE_OFFSET = "dequantize_scale_offset";
const std::string DEQUANTIZE_OFFSET_DATA_TYPE = "dequantize_offset_data_value";
const std::string DEQUANTIZE_OFFSET_DATA_OFFSET = "dequantize_offset_data_offset";
const std::string DEQUANTIZE_OFFSET_WEIGHT_VALUE = "dequantize_offset_weight_value";
const std::string DEQUANTIZE_OFFSET_WEIGHT_OFFSET = "dequantize_offset_weight_offset";
const std::string DEQUANTIZE_OFFSET_PAD_VALUE = "dequantize_offset_pad_value";
const std::string DEQUANTIZE_OFFSET_PAD_OFFSET = "dequantize_offset_pad_offset";

const std::string REQUANTIZE_SCALE_MODE = "requantize_scale_mode";
const std::string REQUANTIZE_SCALE_VALUE = "requantize_scale_value";
const std::string REQUANTIZE_SCALE_OFFSET = "requantize_scale_offset";
const std::string REQUANTIZE_OFFSET_DATA_VALUE = "requantize_offset_data_value";
const std::string REQUANTIZE_OFFSET_DATA_OFFSET = "requantize_offset_data_offset";
const std::string REQUANTIZE_OFFSET_WEIGHT_VALUE = "requantize_offset_weight_value";
const std::string REQUANTIZE_OFFSET_WEIGHT_OFFSET = "requantize_offset_weight_offset";
const std::string REQUANTIZE_OFFSET_PAD_VALUE = "requantize_offset_pad_value";
const std::string REQUANTIZE_OFFSET_PAD_OFFSET = "requantize_offset_pad_offset";

const std::string ATTR_NAME_IS_CONST = "attr_name_is_const";

const std::string ATTR_NAME_GROUP = "group";
const std::string ATTR_NAME_DILATION_SIZE = "dilation_size";
const std::string ATTR_NAME_EPSILON = "epsilon";
const std::string ATTR_NAME_POOLING_MODE = "mode";
const std::string ATTR_NAME_CLASS_NUM = "class_num";
// model
const std::string ATTR_MODEL_TARGET_TYPE = "target_type";

const std::string ATTR_MODEL_STREAM_NUM = "stream_num";

const std::string ATTR_MODEL_EVENT_NUM = "event_num";

const std::string ATTR_MODEL_HUGE_STREAM_LIST = "huge_stream_list";

const std::string ATTR_MODEL_LABEL_NUM = "label_num";

const std::string ATTR_MODEL_MEMORY_SIZE = "memory_size";

const std::string ATTR_MODEL_ZERO_COPY_MEMORY_SIZE = "zero_copy_memory_size";

const std::string ATTR_MODEL_P2P_MEMORY_SIZE = "p2p_memory_size";

const std::string ATTR_MODEL_OUT_NODES_NAME  = "attr_model_out_nodes_name";

const std::string ATTR_MODEL_WEIGHT_SIZE = "weight_size";

const std::string ATTR_MODEL_TASK_GEN_BASE_ADDR = "task_gen_base_addr";

const std::string ATTR_MODEL_TASK_GEN_WEIGHT_ADDR = "task_gen_weight_addr";

const std::string ATTR_MODEL_TASK_GEN_VAR_ADDR = "task_gen_variable_addr";

const std::string ATTR_MODEL_VAR_SIZE = "variable_size";

const std::string ATTR_MODEL_TASK_INDEX_OP_NAME = "task_index_op_name";

const std::string ATTR_MODEL_CORE_TYPE = "core_type";

const std::string ATTR_MODEL_ATC_VERSION = "atc_version";

const std::string ATTR_MODEL_ATC_CMDLINE = "atc_cmdline";

const std::string ATTR_MODEL_OPP_VERSION = "opp_version";

const std::string ATTR_MODEL_SESSION_SCOPE_MEMORY_SIZE = "session_scope_memory_size";

const std::string ATTR_NAME_FILE_CONSTANT_ID = "file_id";

// Public attribute
const std::string ATTR_NAME_IMPLY_TYPE = "imply_type";

const std::string ATTR_NAME_BYTE_SIZE = "op_byte_size";

const std::string ATTR_NAME_FUSION_INFERENCE_ID = "fusion_inference_id";

const std::string ATTR_NAME_FUSION_OPDEF = "fusion_opdef";

const std::string ATTR_NAME_IO_OP = "io_op";

const std::string ATTR_NAME_FUSION_SCOPE = "fusion_scope";

const std::string ATTR_NAME_OPATTR = "opattr";

const std::string ATTR_NAME_SEQLEN_INDEX = "seqlen_index";

const std::string ATTR_NAME_X_INDEX = "x_index";

const std::string ATTR_NAME_CONT_INDEX = "cont_index";

const std::string ATTR_NAME_XSTATIC_INDEX = "xstatic_index";

const std::string TARGET_TYPE_MINI = "MINI";

const std::string TARGET_TYPE_TINY = "TINY";

const std::string TARGET_TYPE_LITE = "LITE";

// l2_normalize
const std::string L2_NORMALIZE_ATTR_AXIS = "axis";
const std::string L2_NORMALIZE_ATTR_EPS = "eps";

const std::string POOL_PARAMA_ATTR_WINDOW = "window";
const std::string POOL_PARAMA_ATTR_CEIL_MODE = "ceil_mode";
const std::string POOL_PARAMA_ATTR_DATA_MODE = "data_mode";
const std::string POOL_PARAMA_ATTR_GLOBAL_POOLING = "global_pooling";
const std::string POOL_PARAMA_ATTR_NAN_OP = "nan_opt";
const std::string POOL_PARAMA_ATTR_PAD_MOD = "pad_mode";

// HCOM
const std::string HCOM_ATTR_ROOT_RANK = "root_rank";
const std::string HCOM_ATTR_RANK_SIZE = "rank_size";

const std::string HCOM_ATTR_REDUCE_TYPE = "reduction";
const std::string HCOM_ATTR_GROUP = "group";
const std::string HCOM_ATTR_SR_TAG = "sr_tag";
const std::string HCOM_ATTR_SRC_RANK = "src_rank";
const std::string HCOM_ATTR_DEST_RANK = "dest_rank";
const std::string HCOM_ATTR_FUSION = "fusion";
const std::string HCOM_ATTR_SHAPE = "shape";
const std::string HCOM_ATTR_DATA_TYPE = "dtype";

// SpaceToDepth/DepthToSpace
const std::string ATTR_NAME_BLOCK_SIZE = "block_size";

// SparseSoftmaxCrossEntropyWithLogits
const std::string SPARSE_SOFT_MAX_ATTR_TLABLES = "Tlabels";

// MaxPoolGradWithArgmax
const std::string MAX_POOL_GRAD_OUTPUT_SHAPE = "max_pool_grad_output_shape";

// AvgPoolGrad
const std::string AVG_POOL_GRAD_OUTPUT_SHAPE = "avg_pool_grad_output_shape";

// Pad
const std::string ATTR_PAD_FORMAT = "attr_pad_format";

// Varible
const std::string VAR_ATTR_FORMAT = "_var_format";
const std::string VAR_ATTR_NAME = "var_name";
const std::string VAR_ATTR_FRACTALZ_FORMAT = "FZ";
const std::string VAR_ATTR_4D_FORMAT = "4D";
const std::string VAR_ATTR_5D_FORMAT = "5D";
const std::string VAR_ATTR_DATA_TYPE = "data_format";
const std::string VAR_ATTR_VAR_IN_NAME = "var_in_name";
const std::string VAR_ATTR_VAR_IN_INDEX = "var_in_index";
const std::string VAR_ATTR_VAR_OUT_INDEX = "var_out_index";
const std::string VAR_ATTR_SHAPE = "shape";
const std::string HALF_VAR_NAME_END = "_fp16";
const std::string VAR_ATTR_INITED = "var_is_inited";

const std::string VAR_ATTR_CONTAINER = "container";
const std::string VAR_ATTR_SHARED_NAME = "shared_name";
const std::string VAR_ATTR_DTYPE = "dtype";

const std::string VAR_ATTR_SRC_VAR_NAME = "_src_var_name";
const std::string VAR_ATTR_VAR_IS_SAVE = "_var_is_save";
const std::string VAR_ATTR_VAR_IS_RESTORE = "_var_is_restore";
const std::string VAR_ATTR_VAR_IS_BROADCAST = "_var_is_broadcast";
const std::string REF_VAR_SRC_VAR_NAME = "ref_var_src_var_name";
const std::string REF_VAR_PRE_PEER_OUT_INDEX = "ref_var_pre_peer_out_index";

// Assign
const std::string ASSIGN_VALIDATE_SHAPE = "validate_shape";
const std::string ASSIGN_VAR_NAME = "_assign_var_name";

// Inplace support
const std::string INPLACE_SUPPORT_INPUT_INDEX = "_inplace_support_input_index";

//  space2bacth batch2space
const std::string BATCH_SPACE_ATTR_BLOCK = "block";
const std::string BATCH_SPACE_ATTR_PADDING = "padding";

// depth_to_space space_to_depth
const std::string DEPTH_SPACE_ATTR_BLOCK_SIZE = "block_size";

// FakeQuantWithMinMaxVars
const std::string FakeQuantWithMinMaxVars_ATTR_MAX = "max";
const std::string FakeQuantWithMinMaxVars_ATTR_MIN = "min";

// mobilenet_ssd_conv_fusion
const std::string SSD_BOXPREDICTOR_BOXES_FUSION = "ssd_boxpredictor_boxes_fusion";
const std::string SSD_BOXPREDICTOR_SCORES_FUSION = "ssd_boxpredictor_scores_fusion";
const std::string SSD_BOXPREDICTOR_FUSION_BOX_TYPE_NUM = "ssd_boxpredictor_fusion_box_type_num";

// lsh project
const std::string LSH_PROJ_TYPE = "lsh_project_type";

//  log time stamp
const std::string LOG_TIME_STAMP_LOGID = "logid";
const std::string LOG_TIME_STAMP_NOTIFY = "notify";

// ShapeN
const std::string SHAPEN_ATTR_N = "N";
const std::string SHAPEN_ATTR_IN_TYPE = "in_type";
const std::string SHAPEN_ATTR_OUT_TYPE = "dtype";

//  GatherV2 attr def
const std::string GATHERV2_ATTR_NAME_TAXIS = "Taxis";
const std::string GATHERV2_ATTR_NAME_TINDICES = "Tindices";
const std::string GATHERV2_ATTR_NAME_TPARAMS = "Tparams";

//  Reshape attr def
const std::string RESHAPE_ATTR_NAME_INPUT_DESC = "input_desc_reshape";
const std::string RESHAPE_ATTR_NAME_OUTPUT_DESC = "output_desc_reshape";

//  axis attr def
const std::string ATTR_NAME_AXIS_ORG_OP = "axis_org_op";

const std::string ATTR_NAME_LINK_WITH_SPARE = "link_with_sparse";

const std::string ATTR_NAME_NET_OUTPUT_FORMAT = "net_output_format";
const std::string ATTR_NAME_NET_OUTPUT_DATATYPE = "net_output_datatype";

//  For constant folding
const std::string ATTR_NO_NEED_CONSTANT_FOLDING = "no_need_constant_folding";

const std::string ATTR_NAME_CONTINUOUS_INPUT = "continuous_input";

const std::string ATTR_NAME_CONTINUOUS_INPUT_ALLOC = "continuous_input_alloc";

const std::string ATTR_NAME_CONTINUOUS_OUTPUT = "continuous_output";

// For AscendWeightQuant+Enter
const std::string ATTR_NAME_FINAL_CONST_NODE = "_final_const_node";

// attr _input_mutable = true means node will modify its input in runtime
const std::string ATTR_NAME_MODIFY_INPUT = "_input_mutable";

const std::string ATTR_NAME_REFERENCE = "reference";

const std::string ATTR_NAME_NOTASK = "_no_task";

const std::string ATTR_NAME_OUTPUT_REUSE_INPUT = "_output_reuse_input";

const std::string ATTR_NAME_REUSE_INPUT_ON_DIM_INDEX = "_reuse_input_on_dim_index";

const std::string ATTR_NAME_NOPADDING_CONTINUOUS_INPUT = "_no_padding_continuous_input";

const std::string ATTR_NAME_NOPADDING_CONTINUOUS_OUTPUT = "_no_padding_continuous_output";

const std::string ATTR_NAME_ATOMIC_INDEX = "atomic_index";

// Used for mark the active label list stream of activated node
const std::string ATTR_NAME_ACTIVE_LABEL_LIST = "_active_label_list";

// Used for l2cache, true: the memory of all inputs is used for the last time.
const std::string ATTR_NAME_IS_END_OF_INPUTMEM_LIFECYCLE = "is_end_of_inputmem_lifecycle";

const std::string ATTR_NAME_DATA_VISIT_DISTANCE = "_data_visit_distance";

// Multi batch
const std::string ATTR_NAME_PRED_VALUE = "_pred_value";
const std::string ATTR_NAME_BATCH_NUM = "_batch_num";
const std::string ATTR_NAME_BATCH_LABEL = "_batch_label";
const std::string ATTR_NAME_COMBINED_BATCH = "_combined_batch";

// Control flow
const std::string ATTR_NAME_STREAM_SWITCH_COND = "switch_condition";
const std::string ATTR_NAME_TRUE_BRANCH_STREAM = "true_branch_stream";
const std::string ATTR_NAME_ACTIVE_STREAM_LIST = "active_stream_list";
const std::string ATTR_NAME_SWITCHN_PRED_VALUE = "switch_pred_value";
const std::string ATTR_NAME_ITERATORS_PER_LOOP = "iterations_per_loop";
const std::string ATTR_NAME_FLOW_CTRL_NODE_FLAG = "is_flow_ctrl_node";
const std::string ATTR_NAME_SUBGRAPH_FIRST_ACTIVE = "subgraph_first_active";
const std::string ATTR_NAME_COMBINED_DYNAMIC_DIMS = "combined_dynamic_dims";

const std::string ATTR_NAME_SWITCH_BRANCH_NODE_LABEL = "_switch_branch_node_label";
const std::string ATTR_NAME_SWITCH_TRUE_BRANCH_FLAG = "_switch_true_branch_flag";
const std::string ATTR_NAME_SWITCH_DATA_TYPE = "_switch_data_type";
const std::string ATTR_NAME_ORIG_NODE_NAME = "_original_node_name";
const std::string ATTR_NAME_CYCLIC_DEPENDENCE_FLAG = "_cyclic_dependence_flag";
const std::string ATTR_NAME_STREAM_SWITCH_TYPE = "_stream_switch_type";

const std::string ATTR_NAME_NEXT_ITERATION = "_next_iteration_node";

// Function Op
const std::string ATTR_NAME_PARENT_NODE_INDEX = "_parent_node_index";

const std::string ATTR_NAME_NEED_INFER_AGAIN = "_need_infer_again";

const std::string ATTR_NAME_MERGE_INPUT_INDEX = "_merge_input_index";
const std::string ATTR_NAME_CONTROL_FLOW_GROUP = "_control_flow_group";

// Used for mark the active node is for loop, type:bool
const std::string ATTR_NAME_IS_LOOP_ACTIVE = "is_loop_active";

const std::string ATTR_NAME_MEMORY_TYPE_INPUT = "memory_type_input";

const std::string ATTR_NAME_MEMORY_TYPE_OUTPUT = "memory_type_output";

const std::string ATTR_NAME_MEMORY_TYPE_WORKSPACE = "memory_type_workspace";

const std::string ATTR_NAME_MEMORY_TYPE_RANGE = "_memory_type_range";

const std::string MODEL_ATTR_SESSION_ID = "session_id";

// lx fusion
const std::string ATTR_NAME_L1_FUSION_GROUP_ID = "_l1_fusion_group_id";
const std::string ATTR_NAME_FUSION_GROUP_KEY = "_fusion_group_key";
const std::string ATTR_NAME_L1_FUSION_GROUP_KEY = "_l1_fusion_group_key";
const std::string ATTR_NAME_FUSION_VIRTUAL_OP = "_fusion_virtual_op";
const std::string ATTR_NAME_FUSION_GROUP_TYPE = "_fusion_group_type";
const std::string ATTR_NAME_L1_FUSION_EXTEND_PTR = "_l1_fusion_extend_content";
const std::string ATTR_NAME_GET_TENSOR_ACTUAL_SIZE = "_tensor_actual_size";
const std::string ATTR_NAME_OUTPUT_OFFSET_FOR_L1_FUSION = "_output_offset_for_l1_fuison";
const std::string ATTR_NAME_SWITCH_FOR_L1_FUSION = "_enable_l1_fusion";
const std::string ATTR_N_BATCH_SPILT = "_is_n_batch_split";
const std::string ATTR_NO_TASK_AND_DUMP_NEEDED = "_no_task_and_dump_needed";
const std::string ATTR_DATA_DUMP_REF = "_datadump_ref";
const std::string ATTR_NAME_OUTPUT_OFFSET_FOR_BUFFER_FUSION = "_output_offset_for_buffer_fusion";
const std::string ATTR_NAME_L2_FUSION_GROUP_ID = "_l2_fusion_group_id";
const std::string ATTR_NAME_SWITCH_FOR_L2_FUSION = "_enable_l2_fusion";
const std::string ATTR_NAME_OP_INPUT_L1_FLAG = "_op_input_l1_flag";
const std::string ATTR_NAME_OP_INPUT_L1_ADDR = "_op_input_l1_addr";
const std::string ATTR_NAME_OP_INPUT_L1_VALID_SIZE = "_op_input_l1_valid_size";
const std::string ATTR_NAME_ENGINE_NAME_FOR_LX = "_lxfusion_engine_name";
const std::string ATTR_NAME_KKERNEL_LIB_NAME_FOR_LX = "_lxfusion_op_kernel_lib_name";
const std::string ATTR_NAME_NEED_LX_FUSION = "_lx_fusion";
const std::string ATTR_NAME_OPTIMIZE_GROUP = "_optimize_group";
const std::string ATTR_NAME_OP_COMPILE_STRATEGY = "_op_compile_strategy";
const std::string ATTR_NAME_TBE_KERNEL_NAME = "_tbe_kernel_name";
const std::string ATTR_NAME_TBE_KERNEL_NAME_FOR_LOAD = "_tbe_kernel_name_for_load";
const std::string ATTR_NAME_TBE_KERNEL_BUFFER = "_tbe_kernel_buffer";
const std::string ATTR_NAME_DATA_SLICE = "_data_slice";
const std::string ATTR_NAME_NEED_RECOVER_ATTR = "_need_recover_attr";
const std::string ATTR_NAME_OFF_SUPERKERNEL_ATTR = "_off_superkernel";

// merge subgraph with output anchor map
const std::string ATTR_NAME_FUSION_ORIGIN_NAME = "_fusion_origin_name";
const std::string ATTR_NAME_FUSION_ORIGIN_OUTPUT_INDEX = "_fusion_origin_output_index";

// read var offset
const std::string ATTR_NAME_INNER_OFFSET = "_inner_offset";

// used for memory allocate
const std::string ATTR_NAME_INPUT_MEM_TYPE_LIST = "_input_memory_type";
const std::string ATTR_NAME_OUTPUT_MEM_TYPE_LIST = "_output_memory_type";
const std::string ATTR_NAME_WORKSPACE_TYPE_LIST = "_workspace_type";
const std::string ATTR_NAME_TENSOR_MEM_TYPE = "_tensor_memory_type";

// Op debug attrs
const std::string ATTR_OP_DEBUG_FLAG = "_op_debug_flag";
const std::string ATTR_OP_DEBUG_MODE = "_op_debug_mode";

// Atomic addr clean attrs
const std::string ATOMIC_ATTR_INPUT_INDEX = "atomic_input_index";
const std::string ATOMIC_ATTR_OUTPUT_INDEX = "atomic_output_index";
const std::string ATOMIC_ATTR_IS_FUSION_NODE = "is_fusion_node";
const std::string EXT_ATTR_ATOMIC_WORKSPACE_INFO = "sub_node_workspace_info";
const std::string EXT_ATTR_ATOMIC_WORKSPACE_OFFSET = "sub_node_workspace_offset";
const std::string ATOMIC_ATTR_IS_ATOMIC_NODE = "is_atomic_node";
const std::string ATOMIC_ATTR_TVM_MAGIC = "_atomic_tvm_magic";
const std::string ATOMIC_ATTR_TVM_METADATA = "_atomic_tvm_metadata";
const std::string ATOMIC_ATTR_TBE_KERNEL_NAME = "_atomic_tbe_kernel_name";
const std::string EXT_ATTR_ATOMIC_TBE_KERNEL = "_atomic_tbe_kernel";

// Source/dst format for Op FormatTransfer
const std::string FORMAT_TRANSFER_SRC_FORMAT = "src_format";
const std::string FORMAT_TRANSFER_DST_FORMAT = "dst_format";

// For compile op by ge call
const std::string ATTR_NEED_COMPILE = "_node_need_compile";

const std::string ATTR_INSERT_BY_MBATCH = "mbatch-inserted-node";

const std::string ATTR_MBATCH_ORIGIN_INPUT_DIMS = "_mbatch_origin_input_dims";

const std::string ATTR_DYNAMIC_TYPE = "mbatch_dynamic_type";

const std::string ATTR_USER_DESIGNEATE_SHAPE_ORDER = "user_designate_shape_order";

// For inserted op
const std::string ATTR_INSERTED_BY_GE = "_inserted_by_ge";

// For compress weight
const std::string ATTR_NAME_COMPRESS_WEIGHT = "_is_compress_weight";

// For data dump
const std::string ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES = "_datadump_original_op_names";
const std::string ATTR_NAME_DATA_DUMP_IS_MULTIOP = "_datadump_is_multiop";
const std::string ATTR_NAME_DATA_DUMP_SUB_SPLITER_INDEX = "_datadump_sub_spliter_index";
const std::string ATTR_NAME_DATA_DUMP_GROUP_OP_NAME = "_datadump_group_op_name";
const std::string ATTR_NAME_DATA_DUMP_ORIGIN_NAME = "_datadump_origin_name";
const std::string ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX = "_datadump_origin_output_index";
const std::string ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT = "_datadump_origin_format";
const std::string ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE = "_datadump_origin_data_type";

// functional ops attr
const std::string ATTR_NAME_IF_THEN_BRANCH = "then_branch";
const std::string ATTR_NAME_IF_ELSE_BRANCH = "else_branch";
const std::string ATTR_NAME_WHILE_COND = "cond";
const std::string ATTR_NAME_WHILE_BODY = "body";

// used for label switch
const std::string ATTR_NAME_LABEL_SWITCH_INDEX = "_label_switch_index";
const std::string ATTR_NAME_LABEL_SWITCH_LIST = "_label_switch_list";
const std::string ATTR_NAME_SUBGRAPH_END_NODE = "_subgraph_end_node";

const std::string ATTR_NAME_INPUT_DATATYPE = "input_datatype";
const std::string ATTR_NAME_OUTPUT_DATATYPE = "output_datatype";

// used for LX tiling
const std::string ATTR_NAME_OP_L1_SPACE = "_l1_space";
const std::string ATTR_NAME_FUSION_TYPE_LIST = "_fusion_type_list";
const std::string ATTR_NAME_VALID_INPUT_SHAPE_LIST_LIST = "_valid_input_shape_list_list";
const std::string ATTR_NAME_VALID_OUTPUT_SHAPE_LIST_LIST = "_valid_output_shape_list_list";
const std::string ATTR_NAME_SLICE_INPUT_OFFSET_LIST_LIST = "_input_offset_list_list";
const std::string ATTR_NAME_SLICE_OUTPUT_OFFSET_LIST_LIST = "_output_offset_list_list";

// for unregistered op
const std::string ATTR_NAME_UNREGST_OPPATH = "_unregst_oppath";
const std::string ATTR_NAME_UNREGST_ATTRLIST = "_unregst_attrlist";

// used for Horovod
const std::string ATTR_INTER_EVENT_IDENTIFY = "event_id";
const std::string ATTR_HOROVOD_ATTR_REDUCE_TYPE = "reduce_op";
// used for allreduce tailing optimization
const std::string ATTR_NAME_HCCL_FUSED_GROUP = "_hccl_fused_group";
const std::string ATTR_NAME_HCCL_FUSED_FLAG = "_hccl_fused_node";

// used for parallel group
const std::string ATTR_NAME_PARALLEL_GROUP = "_parallel_group";

// dynamic shape attr
const std::string ATTR_DYNAMIC_SHAPE_FIXED_ADDR = "_alloc_fixed_addr";
const std::string ATTR_DYNAMIC_SHAPE_FIXED_ADDR_INDEX = "_alloc_fixed_addr_index";
const std::string ATTR_DYNAMIC_SHAPE_SINGLE_AICPU = "_single_aicpu_dynamic";

// op dynamic input
const std::string ATTR_NAME_DYNAMIC_INPUT_START = "_dynamic_input_index_start";
const std::string ATTR_NAME_DYNAMIC_INPUT_END = "_dynamic_input_index_end";

// atc user def dtype&format
const std::string ATTR_ATC_USER_DEFINE_DATATYPE = "_user_defined_data_type";
const std::string ATTR_ATC_USER_DEFINE_FORMAT = "_user_defined_format";

// atc user def dtype&format
const std::string ATTR_ATC_USER_DEFINE_OUTPUT_NODES = "_user_defined_output_nodes";

// for fusion op plugin
const std::string ATTR_NAME_FUSIONOP_ORIGINAL_TYPE = "_fusionop_original_type";

// graph partition for aicpu
const std::string ATTR_NAME_PLD_FRONT_NODE_ENGINE_NAME = "pld_front_node_engine_name";
const std::string ATTR_NAME_END_REAR_NODE_ENGINE_NAME = "end_rear_node_engine_name";

// input and output memory type
const std::string ATTR_VARIABLE_PLACEMENT = "_variable_placement";
const std::string ATTR_INPUT_MEMORY_TYPE = "_input_memory_type";
const std::string ATTR_OUTPUT_MEMORY_TYPE = "_output_memory_type";
const std::string ATTR_NAME_SPECIAL_OUTPUT_SIZE = "_special_output_size";
const std::string ATTR_NAME_SPECIAL_INPUT_SIZE = "_special_input_size";

// stage
const std::string ATTR_STAGE_LEVEL = "_stage_level";

// input_output_offset
const std::string ATTR_ZERO_COPY_BASIC_OFFSET = "_zero_copy_basic_offset";
const std::string ATTR_ZERO_COPY_RELATIVE_OFFSET = "_zero_copy_relative_offset";

// mark node cannot be deleted
const std::string ATTR_NAME_CANNOT_BE_DELETED = "_cannot_be_deleted";

// The processing mode of INF and NAN during floating-point number calculation.
const std::string ATTR_FP_CEILING_MODE = "_fp_ceiling_mode";
// count of data from getnext_sink
const std::string ATTR_GETNEXT_SINK_DATA_COUNT = "N";
const std::string ATTR_GETNEXT_SINK_SHAPE_INFO = "shape_info";

// getnext_sink marked on NetOutput
const std::string ATTR_GETNEXT_SINK_DYNMAIC = "getnext_sink_dynamic";
const std::string ATTR_ALL_GEARS_INFO = "all_gears_info";

// Calculate the operator output memory
const std::string ATTR_NAME_MEMORY_SIZE_CALC_TYPE = "_memory_size_calc_type";
// Indicates which operators keep the precision unchanged
const std::string ATTR_NAME_KEEP_DTYPE = "_keep_dtype";

// profiling task mark on fp bp
const std::string ATTR_NAME_INSERT_FP_PROFILILNG_TASK = "_fp_profiling_task";
const std::string ATTR_NAME_INSERT_BP_PROFILILNG_TASK = "_bp_profiling_task";
const std::string ATTR_NAME_INSERT_END_PROFILILNG_TASK = "_end_profiling_task";
const std::string ATTR_NAME_INSERT_PROFILILNG_TASK_LOG_ID = "_profiling_log_id";
// padding dimension type (FE set and ge get)
const std::string ATTR_NAME_RESHAPE_INFER_TYPE = "_infer_reshape_type";

// mark single op scene
const std::string ATTR_SINGLE_OP_SCENE = "_single_op_scene";

// for fe judge whether trans/cast op is inserted
const std::string ATTR_NAME_FORMAT_CONTINUOUS = "_format_continuous";
const std::string ATTR_NAME_REFRESH_CONTINUOUS_FLAG = "_refresh_continuous_flag";
const std::string ATTR_NAME_FORMAT_AGNOSTIC = "_format_agnostic";
const std::string ATTR_NAME_FORMAT_AGNOSTIC_EXCEPT_OUTPUT = "_format_agnostic_except_output";
const std::string ATTR_NAME_FORMAT_AGNOSTIC_EXCEPT_INPUT = "_format_agnostic_except_input";

// for ffts/ffts_plus
const std::string ATTR_NAME_FFTS_SUB_GRAPH = "_ffts_sub_graph";
const std::string ATTR_NAME_THREAD_SCOPE_ID = "_thread_scope_id";
const std::string ATTR_NAME_THREAD_MODE = "_thread_mode";
const std::string ATTR_NAME_FFTS_PLUS_SUB_GRAPH = "_ffts_plus_sub_graph";
const std::string ATTR_NAME_COMPOSITE_ENGINE_NAME = "_composite_engine_name";
const std::string ATTR_NAME_COMPOSITE_ENGINE_KERNEL_LIB_NAME = "_composite_engine_kernel_lib_name";
const std::string ATTR_NAME_CUBE_VECTOR_CORE_TYPE = "_cube_vector_core_type";

// mark fuzz build scene
const std::string ATTR_NAME_FUZZ_BUILD = "_fuzz_build";
const std::string ATTR_NAME_PLACEMENT = "_mem_type";
const std::string ATTR_NAME_VALUE = "_value";
const std::string ATTR_NAME_VALUE_RANGE = "_value_range";
const std::string ATTR_NAME_BUILD_MODE = "_build_mode";
const std::string ATTR_NAME_FUZZ_BUILD_RES_ATTRS = "_fuzz_build_res";
const std::string ATTR_NAME_FUZZ_INPUTS_SUPPORTED_ATTRS = "_inputs_support_info";
const std::string ATTR_NAME_FUZZ_OUTPUTS_SUPPORTED_ATTRS = "_outputs_support_info";
const std::string ATTR_NAME_FUZZ_IS_HIGH_PERFORMANCE_ATTRS = "_is_high_performance";
const std::string ATTR_NAME_IS_ORIGINAL_INPUT = "_is_original_input";

// buffer pool allocator
const std::string ATTR_NAME_BUFFER_POOL_ID = "_buffer_pool_id";
const std::string ATTR_NAME_BUFFER_POOL_SIZE = "_buffer_pool_size";
const std::string ATTR_NAME_EVENT_MULTIPLEXING = "_event_multiplexing";
const std::string ATTR_NAME_BUFFER_POOL_NODE_SIZE_AND_OFFSET = "_buffer_pool_node_size_and_offset";

// session scope memory
const std::string ATTR_NAME_WORKSPACE_MEMORY_NO_REUSE_SCOPE = "_workspace_memory_no_reuse_scope";

// for blocking op
const std::string ATTR_NAME_IS_BLOCKING_OP = "_is_blocking_op";

// for op specified engine
const std::string ATTR_NAME_OP_SPECIFIED_ENGINE_NAME = "_specified_engine_name";
const std::string ATTR_NAME_OP_SPECIFIED_KERNEL_LIB_NAME = "_specified_kernel_lib_name";

// for pipeline partition
const std::string ATTR_NAME_PIPELINE_PARTITIONED = "_pipeline_partitioned";
const std::string ATTR_NAME_OUTPUT_PIPELINE = "_output_pipeline";

// model deploy scheduler(mds)
const std::string ATTR_NAME_GRADIENT_NODE = "_gradient_node";
const std::string ATTR_NAME_TRAINABLE_VAR = "_trainable_var";
const std::string ATTR_NAME_FISSION_FACTOR = "_fission_factor";
const std::string ATTR_NAME_DEPLOY_INFO = "_deploy_info";
const std::string ATTR_NAME_CUT_INFO = "_cut_info";
const std::string ATTR_NAME_DEPLOY_DEVICE_TYPE = "_device_type";
const std::string ATTR_NAME_DEPLOY_DEVICE_ID = "_device_id";
const std::string ATTR_NAME_DEPLOY_GRAPH_INPUTS = "_graph_inputs";
const std::string ATTR_NAME_DEPLOY_NEED_RETURN_RESULT = "_need_return_result";

// for qos
const std::string ATTR_NAME_QOS_SERVICE_LABEL = "_qos_service_label";

// for constant folding, mark potential const
const std::string ATTR_NAME_POTENTIAL_CONST = "_is_potential_const";
const std::string ATTR_NAME_POTENTIAL_WEIGHT = "_potential_weight";
const std::string ATTR_NAME_POTENTIAL_WEIGHT_INDICES = "_potential_weight_indices";

// name of network output tensor
const std::string ATTR_NAME_ORIGIN_OUTPUT_TENSOR_NAME = "_origin_output_tensor_name";

// for scope op to record the input and output information of the original graph node
const std::string ATTR_NAME_ORIGIN_GRAPH_NODE_INPUTS = "_origin_graph_node_inputs";
const std::string ATTR_NAME_ORIGIN_GRAPH_NODE_OUTPUTS = "_origin_graph_node_outputs";

// for operator resource list(e.g. queues, channels)
const std::string ATTR_NAME_RESOURCE_LIST = "_resource_list";

// for no tiling
const std::string ATTR_NAME_OP_TILING_INLINE_ENGINE = "_op_tiling_inline_engine";
const std::string ATTR_NAME_OP_EXPORT_SHAPE_ENGINE = "_op_export_shape_engine";
const std::string ATTR_NAME_OP_MAX_SHAPE = "_op_max_shape";
const std::string ATTR_NAME_TENSOR_MAX_SHAPE = "_tensor_max_shape";
const std::string ATTR_NAME_OP_NO_TILING = "_op_no_tiling";
const std::string ATTR_NAME_TENSOR_DESC_MEM_OFFSET = "_tensor_desc_mem_offset";
const std::string ATTR_NAME_TENSOR_NO_TILING_MEM_TYPE = "_tensor_no_tiling_mem_type";

// for subgraph multi dims
const std::string ATTR_NAME_SUBGRAPH_MULTI_DIMS_INDEX = "_subgraph_multi_dims_index";
const std::string ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_SHAPE = "_subgraph_multi_dims_input_shape";
const std::string ATTR_NAME_SUBGRAPH_MULTI_DIMS_INPUT_DIMS = "_subgraph_multi_dims_input_dims";
const std::string ATTR_NAME_SUBGRAPH_IS_MULTI_DIMS = "_subgraph_is_multi_dims";
const std::string ATTR_NAME_OP_MULTI_DIMS_INPUT_DIMS = "_op_multi_dims_input_dims";
}  // namespace ge
