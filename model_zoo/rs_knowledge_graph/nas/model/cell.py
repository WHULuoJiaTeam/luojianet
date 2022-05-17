import luojianet_ms as luojia
from luojianet_ms import nn
from luojianet_ms import ops
import math
import numpy
import luojianet_ms.common.initializer as weight_init
from collections import OrderedDict
from model.ops import OPS, OPS_mini
from model.ops import conv3x3
class ReLUConvBN(nn.Module):

    def __init__(self, C_in, C_out):
        super(ReLUConvBN, self).__init__()
        kernel_size = 3
        padding = 1
        self.scale = 1
        self.op = nn.SequentialCell(
            nn.ReLU(),
            nn.Conv2d(C_in, C_out, kernel_size, padding=0),
            nn.BatchNorm2d(C_out)
        )

        self.scale = C_in/C_out
        self._initialize_weights()

    def call(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = nn.ResizeBilinear()(x, [feature_size_h, feature_size_w], align_corners=True)
        return self.op(x)

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.HeUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.has_bias:
                    cell.bias.set_data(get_conv_bias(cell))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(weight_init.initializer(1,
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(0,
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

class ConvBNReLU(nn.Module):

    def __init__(self, C_in, C_out):
        super(ConvBNReLU, self).__init__()
        kernel_size = 3
        padding = 1
        self.scale = 1
        self.op = nn.SequentialCell(
            nn.Conv2d(C_in, C_out, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(C_out),
            nn.ReLU()
        )

        self.scale = C_in/C_out
        self._initialize_weights()

    def call(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = nn.ResizeBilinear()(x, [feature_size_h, feature_size_w], align_corners=True)
        return self.op(x)

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.HeUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.has_bias:
                    cell.bias.set_data(get_conv_bias(cell))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(weight_init.initializer(1,
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(0,
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))


class MixedCell(nn.Module):

    def __init__(self, C_in, C_out):
        super(MixedCell, self).__init__()
        kernel_size = 5
        padding = 2
        self.scale = 1
        self._ops = nn.CellList()
        self._ops_index = OrderedDict()
        for op_name in OPS:
            op = OPS[op_name](C_in, C_out, 1, True)
            self._ops.append(op)
            self._ops_index[op_name] = int(len(self._ops) - 1)
        self.ops_num = len(self._ops)
        self.scale = C_in/C_out
        self._initialize_weights()

    def call(self, x, cell_alphas):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = nn.ResizeBilinear()(x, [feature_size_h, feature_size_w], align_corners=True)
        return sum(w * self._ops[op](x) for w, op in zip(cell_alphas, self._ops))

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.HeUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.has_bias:
                    cell.bias.set_data(get_conv_bias(cell))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(weight_init.initializer(1,
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(0,
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))

class MixedRetrainCell(nn.Module):

    def __init__(self, C_in, C_out, arch):
        super(MixedRetrainCell, self).__init__()
        self.scale = 1
        self._ops = nn.CellList()
        self._ops_index = OrderedDict()
        for i, op_name in enumerate(OPS):
            if arch[i] == 1:
                op = OPS[op_name](C_in, C_out, 1, True)
                self._ops.append(op)
                self._ops_index[op_name] = int(len(self._ops) - 1)
        self.ops_num = len(self._ops)
        self.scale = C_in/C_out
        self._initialize_weights()

    def call(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = nn.ResizeBilinear()(x, [feature_size_h, feature_size_w], align_corners=True)
        # sum = (1e-3 * ops.StandardNormal()((4, int(x.shape[1] / self.scale), feature_size_h, feature_size_w)))
        # for op in self._ops_index:
        #     op_rs = self._ops[self._ops_index[op]](x)
        #     sum += op_rs
        # return sum
        return sum(self._ops[self._ops_index[op]](x) for op in self._ops_index)

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.HeUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.has_bias:
                    cell.bias.set_data(get_conv_bias(cell))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(weight_init.initializer(1,
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(0,
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))


class Fusion(nn.Module):

    def __init__(self, C_in, C_out):
        super(Fusion, self).__init__()

        self.scale = 1

        self.conv = nn.SequentialCell(
        conv3x3(C_in, C_out, 1),
        nn.BatchNorm2d(C_out, 1),
        nn.ReLU())
        self.scale = C_in / C_out
        self._initialize_weights()

    def call(self, x):
        if self.scale != 0:
            feature_size_h = self.scale_dimension(x.shape[2], self.scale)
            feature_size_w = self.scale_dimension(x.shape[3], self.scale)
            x = nn.ResizeBilinear()(x, [feature_size_h, feature_size_w], align_corners=True)
        return self.conv(x)

    def _initialize_weights(self):
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(weight_init.initializer(weight_init.HeUniform(),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.has_bias:
                    cell.bias.set_data(get_conv_bias(cell))
            elif isinstance(cell, nn.BatchNorm2d):
                cell.gamma.set_data(weight_init.initializer(1,
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(0,
                                                           cell.beta.shape,
                                                           cell.beta.dtype))

    def scale_dimension(self, dim, scale):
        return (int((float(dim) - 1.0) * scale + 1.0) if dim % 2 == 1 else int((float(dim) * scale)))




def calculate_fan_in_and_fan_out(shape):
    """
    calculate fan_in and fan_out

    Args:
        shape (tuple): input shape.

    Returns:
        Tuple, a tuple with two elements, the first element is `n_in` and the second element is `n_out`.
    """
    dimensions = len(shape)
    if dimensions < 2:
        raise ValueError("Fan in and fan out can not be computed for tensor with fewer than 2 dimensions")
    if dimensions == 2:
        fan_in = shape[1]
        fan_out = shape[0]
    else:
        num_input_fmaps = shape[1]
        num_output_fmaps = shape[0]
        receptive_field_size = 1
        if dimensions > 2:
            receptive_field_size = shape[2] * shape[3]
        fan_in = num_input_fmaps * receptive_field_size
        fan_out = num_output_fmaps * receptive_field_size
    return fan_in, fan_out

def get_conv_bias(cell):
    """Bias initializer for conv."""
    weight = weight_init.initializer(weight_init.HeUniform(negative_slope=math.sqrt(5)),
                                     cell.weight.shape, cell.weight.dtype).to_tensor()
    fan_in, _ = calculate_fan_in_and_fan_out(weight.shape)
    bound = 1 / math.sqrt(fan_in)
    return weight_init.initializer(weight_init.Uniform(scale=bound),
                                   cell.bias.shape, cell.bias.dtype)