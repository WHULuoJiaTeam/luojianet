
from luojianet_ms import nn, ops
from luojianet_ms import dtype as mstype


class ConvGRUCell2(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        super(ConvGRUCell2, self).__init__()

        # filters used for gates
        gru_input_channel = input_channel + output_channel
        self.output_channel = output_channel

        self.gate_conv = nn.Conv2d(gru_input_channel, output_channel * 2, kernel_size)
        self.reset_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)
        self.update_gate_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        # filters used for outputs
        self.output_conv = nn.Conv2d(gru_input_channel, output_channel, kernel_size)
        self.output_norm = nn.GroupNorm(1, output_channel, 1e-5, True)

        self.activation = nn.Tanh()
        self.cat_axis_1 = ops.Concat(axis=1)
        self.sigmoid = nn.Sigmoid()

    def gates(self, x, h):
        # x = N x C x H x W
        # h = N x C x H x W

        # c = N x C*2 x H x W
        c = self.cat_axis_1((x, h))
        f = self.gate_conv(c)

        # r = reset gate, u = update gate
        # both are N x O x H x W
        split = ops.Split(1, 2)
        r, u = split(f)

        rn = self.reset_gate_norm(r)
        un = self.update_gate_norm(u)
        rns = self.sigmoid(rn)
        uns = self.sigmoid(un)

        return rns, uns

    def output(self, x, h, r, u):
        f = self.cat_axis_1((x, r * h))
        o = self.output_conv(f)
        on = self.output_norm(o)

        return on

    def forward(self, x, h=None):
        N, C, H, W = x.shape
        HC = self.output_channel
        if h is None:
            zeros = ops.Zeros()
            h = zeros((N, HC, H, W), mstype.float32)

        r, u = self.gates(x, h)
        o = self.output(x, h, r, u)
        y = self.activation(o)
        output = u * h + (1 - u) * y

        return output, output


class ConvGRUCell(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size=3):
        super(ConvGRUCell, self).__init__()

        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size

        self.conv_gates = nn.Conv2d(self.input_channels + self.hidden_channels, 2 * self.hidden_channels,
                                    kernel_size=self.kernel_size, stride=1, has_bias=True)
        self.convc = nn.Conv2d(self.input_channels + self.hidden_channels, self.hidden_channels,
                               kernel_size=self.kernel_size, stride=1, has_bias=True)
        self.cat_axis_1 = ops.Concat(axis=1)
        self.sigmoid = nn.Sigmoid()
        self.activation = nn.Tanh()

    def forward(self, x, h):
        N, C, H, W = x.shape[0], x.shape[1], x.shape[2], x.shape[3]
        HC = self.hidden_channels
        if h is None:
            zeros = ops.Zeros()
            h = zeros((N, HC, H, W), mstype.float32)

        input = self.cat_axis_1((x, h))
        gates = self.conv_gates(input)

        split = ops.Split(1, 2)
        reset_gate, update_gate = split(gates)

        # activation
        reset_gate = self.sigmoid(reset_gate)
        update_gate = self.sigmoid(update_gate)

        # print(reset_gate)
        # concatenation
        input = self.cat_axis_1((x, reset_gate * h))

        # convolution
        conv = self.convc(input)

        # activation
        conv = self.activation(conv)

        # soft update
        output = update_gate * h + (1 - update_gate) * conv

        return output, output


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, has_bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class Conv2DReLuBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(Conv2DReLuBN, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, has_bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, has_bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBn, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvTransBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvTransBnReLU, self).__init__()
        self.conv = nn.Conv2dTranspose(in_channels, out_channels, kernel_size, stride=stride, has_bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvTransReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvTransReLU, self).__init__()
        self.conv = nn.Conv2dTranspose(in_channels, out_channels, kernel_size, stride=stride, has_bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvBnReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBnReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, has_bias=False)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class ConvReLU3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvReLU3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, has_bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class ConvBn3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvBn3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, has_bias=False)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        return self.bn(self.conv(x))


class ConvGnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvGnReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, has_bias=False)
        G = max(1, out_channels // 8)
        self.gn = nn.GroupNorm(G, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))


class ConvGn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvGn, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, has_bias=False)
        G = max(1, out_channels // 8)
        self.gn = nn.GroupNorm(G, out_channels)

    def forward(self, x):
        return self.gn(self.conv(x))


class ConvTransGnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super(ConvTransGnReLU, self).__init__()
        self.conv = nn.Conv2dTranspose(in_channels, out_channels, kernel_size, stride=stride, has_bias=False)
        G = max(1, out_channels // 8)
        self.gn = nn.GroupNorm(G, out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample=None):
        super(BasicBlock, self).__init__()

        self.conv1 = ConvBnReLU(in_channels, out_channels, kernel_size=3, stride=stride)
        self.conv2 = ConvBn(out_channels, out_channels, kernel_size=3, stride=1)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            x = self.downsample(x)
        out += x

        return out


class Hourglass3d(nn.Module):
    def __init__(self, channels):
        super(Hourglass3d, self).__init__()

        self.conv1a = ConvBnReLU3D(channels, channels * 2, kernel_size=3, stride=2)
        self.conv1b = ConvBnReLU3D(channels * 2, channels * 2, kernel_size=3, stride=1)

        self.conv2a = ConvBnReLU3D(channels * 2, channels * 4, kernel_size=3, stride=2)
        self.conv2b = ConvBnReLU3D(channels * 4, channels * 4, kernel_size=3, stride=1)

        self.dconv2 = nn.SequentialCell(
            nn.Conv3dTranspose(channels * 4, channels * 2, kernel_size=3, stride=2, has_bias=False),
            nn.BatchNorm3d(channels * 2))

        self.dconv1 = nn.SequentialCell(
            nn.Conv3dTranspose(channels * 2, channels, kernel_size=3, stride=2, has_bias=False),
            nn.BatchNorm3d(channels))

        self.redir1 = ConvBn3D(channels, channels, kernel_size=1, stride=1)
        self.redir2 = ConvBn3D(channels * 2, channels * 2, kernel_size=1, stride=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        conv1 = self.conv1b(self.conv1a(x))
        conv2 = self.conv2b(self.conv2a(conv1))
        dconv2 = self.relu(self.dconv2(conv2) + self.redir2(conv1))
        dconv1 = self.relu(self.dconv1(dconv2) + self.redir1(x))

        return dconv1


if __name__ == "__main__":
    pass
