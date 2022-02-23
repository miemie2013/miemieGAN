# code was heavily based on https://github.com/clovaai/stargan-v2
# Users should be careful about adopting these functions in any commercial matters.
# https://github.com/clovaai/stargan-v2#license
import paddle
from paddle import nn
import paddle.nn.functional as F

import numpy as np
import math
import scipy



def _get_filter_size(filter):
    if filter is None:
        return 1, 1
    fw = filter.shape[-1]
    fh = filter.shape[0]
    return fw, fh

def upfirdn2d_setup_filter(shape, normalize=True, flip_filter=False, gain=1, separable=None):
    r"""Convenience function to setup 2D FIR filter for `upfirdn2d()`.

    Args:
        shape:       Torch tensor, numpy array, or python list of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable),
                     `[]` (impulse), or
                     `None` (identity).
        normalize:   Normalize the filter so that it retains the magnitude
                     for constant input signal (DC)? (default: True).
        flip_filter: Flip the filter? (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).
        separable:   Return a separable filter? (default: select automatically).

    Returns:
        Float32 tensor of the shape
        `[filter_height, filter_width]` (non-separable) or
        `[filter_taps]` (separable).
    """
    # Validate.
    if shape is None:
        shape = 1
    shape = paddle.to_tensor(shape, dtype='float32')
    assert shape.ndim in [0, 1, 2]
    assert shape.numel() > 0
    if shape.ndim == 0:
        shape = shape[np.newaxis]

    # Separable?
    if separable is None:
        separable = (shape.ndim == 1 and shape.numel() >= 8)
    if shape.ndim == 1 and not separable:
        # ger()相当于向量自乘
        shape = paddle.unsqueeze(shape, 1)  # [n, 1]
        shape = paddle.matmul(shape, shape.transpose((1, 0)))  # [n, n]
    assert shape.ndim == (1 if separable else 2)

    # Apply normalize, flip, gain, and device.
    if normalize:
        shape /= shape.sum()
    if flip_filter:
        shape = shape.flip(list(range(shape.ndim)))
    shape = shape * (gain ** (shape.ndim / 2))
    return shape


def bias_act(x, b=None, dim=1, act='linear', alpha=None, gain=None, clamp=None):
    assert clamp is None or clamp >= 0
    # activation_funcs = {
    #     'linear':   dnnlib.EasyDict(func=lambda x, **_:         x,                                          def_alpha=0,    def_gain=1,             cuda_idx=1, ref='',  has_2nd_grad=False),
    #     'relu':     dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.relu(x),                def_alpha=0,    def_gain=np.sqrt(2),    cuda_idx=2, ref='y', has_2nd_grad=False),
    #     'lrelu':    dnnlib.EasyDict(func=lambda x, alpha, **_:  torch.nn.functional.leaky_relu(x, alpha),   def_alpha=0.2,  def_gain=np.sqrt(2),    cuda_idx=3, ref='y', has_2nd_grad=False),
    #     'tanh':     dnnlib.EasyDict(func=lambda x, **_:         torch.tanh(x),                              def_alpha=0,    def_gain=1,             cuda_idx=4, ref='y', has_2nd_grad=True),
    #     'sigmoid':  dnnlib.EasyDict(func=lambda x, **_:         torch.sigmoid(x),                           def_alpha=0,    def_gain=1,             cuda_idx=5, ref='y', has_2nd_grad=True),
    #     'elu':      dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.elu(x),                 def_alpha=0,    def_gain=1,             cuda_idx=6, ref='y', has_2nd_grad=True),
    #     'selu':     dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.selu(x),                def_alpha=0,    def_gain=1,             cuda_idx=7, ref='y', has_2nd_grad=True),
    #     'softplus': dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.softplus(x),            def_alpha=0,    def_gain=1,             cuda_idx=8, ref='y', has_2nd_grad=True),
    #     'swish':    dnnlib.EasyDict(func=lambda x, **_:         torch.sigmoid(x) * x,                       def_alpha=0,    def_gain=np.sqrt(2),    cuda_idx=9, ref='x', has_2nd_grad=True),
    # }
    def_gain = 1.0
    if act in ['relu', 'lrelu', 'swish']:  # 除了这些激活函数的def_gain = np.sqrt(2)，其余激活函数的def_gain = 1.0
        def_gain = np.sqrt(2)
    def_alpha = 0.0
    if act in ['lrelu']:  # 除了这些激活函数的def_alpha = 0.2，其余激活函数的def_alpha = 0.0
        def_alpha = 0.2

    alpha = float(alpha if alpha is not None else def_alpha)
    gain = float(gain if gain is not None else def_gain)
    clamp = float(clamp if clamp is not None else -1)

    # 加上偏移
    if b is not None:
        new_shape = [-1 if i == dim else 1 for i in range(x.ndim)]
        b_ = paddle.reshape(b, new_shape)
        x = x + b_

    # 经过激活函数
    alpha = float(alpha)  # 只有leaky_relu需要
    if act == 'linear':
        pass
    elif act == 'relu':
        x = F.relu(x)
    elif act == 'lrelu':
        x = F.leaky_relu(x, alpha)
    elif act == 'tanh':
        x = paddle.tanh(x)
    elif act == 'sigmoid':
        x = F.sigmoid(x)
    elif act == 'elu':
        x = F.elu(x)
    elif act == 'selu':
        x = F.selu(x)
    elif act == 'softplus':
        x = F.softplus(x)
    elif act == 'swish':
        x = F.sigmoid(x) * x
    else:
        raise NotImplementedError("activation \'{}\' is not implemented.".format(act))


    # 乘以缩放因子
    gain = float(gain)
    if gain != 1:
        x = x * gain

    # 限制范围
    if clamp >= 0:
        x = paddle.clip(x, -clamp, clamp)
    return x

def _parse_padding(padding):
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all(isinstance(x, int) for x in padding)
    if len(padding) == 2:
        padx, pady = padding
        padding = [padx, padx, pady, pady]
    padx0, padx1, pady0, pady1 = padding
    return padx0, padx1, pady0, pady1


def _conv2d_wrapper(x, w, stride=1, padding=0, groups=1, transpose=False, flip_weight=True):
    """Wrapper for the underlying `conv2d()` and `conv_transpose2d()` implementations.
    """
    out_channels, in_channels_per_group, kh, kw = w.shape

    # Flip weight if requested.
    if not flip_weight: # conv2d() actually performs correlation (flip_weight=True) not convolution (flip_weight=False).
        w = w.flip([2, 3])

    # Workaround performance pitfall in cuDNN 8.0.5, triggered when using
    # 1x1 kernel + memory_format=channels_last + less than 64 channels.
    if kw == 1 and kh == 1 and stride == 1 and padding in [0, [0, 0], (0, 0)] and not transpose:
        if x.shape[2] * x.shape[3] == 1 and min(out_channels, in_channels_per_group) < 64:
            if out_channels <= 4 and groups == 1:
                in_shape = x.shape
                x = w.squeeze(3).squeeze(2) @ x.reshape([in_shape[0], in_channels_per_group, -1])
                x = x.reshape([in_shape[0], out_channels, in_shape[2], in_shape[3]])
            else:
                # x = x.to(memory_format=torch.contiguous_format)
                # w = w.to(memory_format=torch.contiguous_format)
                x = F.conv2d(x, w, groups=groups)
            # return x.to(memory_format=torch.channels_last)
            return x

    # Otherwise => execute using conv2d_gradfix.
    if transpose:
        return F.conv2d_transpose(x, weight=w, bias=None, stride=stride, padding=padding, output_padding=0, groups=groups, dilation=1)
    else:
        return F.conv2d(x, weight=w, bias=None, stride=stride, padding=padding, dilation=1, groups=groups)

def _parse_scaling(scaling):
    # scaling 一变二
    if isinstance(scaling, int):
        scaling = [scaling, scaling]
    assert isinstance(scaling, (list, tuple))
    assert all(isinstance(x, int) for x in scaling)
    sx, sy = scaling
    assert sx >= 1 and sy >= 1
    return sx, sy

def upfirdn2d(x, filter, up=1, down=1, padding=0, flip_filter=False, gain=1):
    if filter is None:
        filter = paddle.ones([1, 1], dtype=paddle.float32)
    batch_size, num_channels, in_height, in_width = x.shape
    upx, upy = _parse_scaling(up)        # scaling 一变二
    downx, downy = _parse_scaling(down)  # scaling 一变二
    padx0, padx1, pady0, pady1 = _parse_padding(padding)

    # Upsample by inserting zeros.
    # paddle最多支持5维张量，所以分开2次pad。
    # 根据data_format指定的意义填充(pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back)
    x = x.reshape([batch_size, num_channels, in_height, 1, in_width])
    x = paddle.nn.functional.pad(x, [0, 0, 0, upy - 1, 0, 0], data_format="NCDHW")
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width, 1])
    x = paddle.nn.functional.pad(x, [0, upx - 1, 0, 0, 0, 0], data_format="NCDHW")
    x = x.reshape([batch_size, num_channels, in_height * upy, in_width * upx])

    # Pad or crop.
    x = F.pad(x, [max(padx0, 0), max(padx1, 0), max(pady0, 0), max(pady1, 0)])
    x = x[:, :, max(-pady0, 0) : x.shape[2] - max(-pady1, 0), max(-padx0, 0) : x.shape[3] - max(-padx1, 0)]

    # Setup filter.
    filter = filter * (gain ** (filter.ndim / 2))
    assert filter.dtype == x.dtype
    # filter = paddle.cast(filter, dtype=x.dtype)
    if not flip_filter:
        filter = filter.flip(list(range(filter.ndim)))

    # Convolve with the filter.
    filter = paddle.unsqueeze(filter, [0, 1]).tile([num_channels, 1] + [1] * filter.ndim)
    if filter.ndim == 4:
        x = F.conv2d(x, weight=filter, groups=num_channels)
    else:
        x = F.conv2d(x, weight=filter.unsqueeze(2), groups=num_channels)
        x = F.conv2d(x, weight=filter.unsqueeze(3), groups=num_channels)

    # Downsample by throwing away pixels.
    # 因为:: （paddle.strided_slice()）没有实现二阶梯度，所以用其它等价实现。
    x222 = x[:, :, ::downy, ::downx]  # RuntimeError: (NotFound) The Op strided_slice_grad doesn't have any grad op.
    assert downy == downx
    # if downy == 1:
    #     pass
    # elif downy == 2:
    #     N, C, H, W = x.shape
    #     # print('rrrrrrrrrrrrrrrrrrr')
    #     # print(N, C, H, W)
    #     assert H == W
    #     pad_height_bottom = 0
    #     pad_width_right = 0
    #     if H % 2 == 1:
    #         pad_height_bottom = 1
    #         pad_width_right = 1
    #     stride2_kernel = np.zeros((C, 1, 2, 2), dtype=np.float32)
    #     stride2_kernel[:, :, 0, 0] = 1.0
    #     stride2_kernel = paddle.to_tensor(stride2_kernel)
    #     stride2_kernel.stop_gradient = True
    #     x = F.conv2d(x, stride2_kernel, bias=None, stride=2, groups=C,
    #                  padding=[[0, 0], [0, 0], [0, pad_height_bottom], [0, pad_width_right]])
    # else:
    #     raise NotImplementedError("downy \'{}\' is not implemented.".format(downy))
    return x222


def downsample2d(x, f, down=2, padding=0, flip_filter=False, gain=1):
    downx, downy = _parse_scaling(down)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw - downx + 1) // 2,
        padx1 + (fw - downx) // 2,
        pady0 + (fh - downy + 1) // 2,
        pady1 + (fh - downy) // 2,
    ]
    return upfirdn2d(x, f, down=down, padding=p, flip_filter=flip_filter, gain=gain)


def conv2d_resample(x, w, filter=None, up=1, down=1, padding=0, groups=1, flip_weight=True, flip_filter=False):
    r""" 2D卷积（带有上采样或者下采样）
    Padding只在最开始执行一次.

    Args:
        x:              Input tensor of shape
                        `[batch_size, in_channels, in_height, in_width]`.
        w:              Weight tensor of shape
                        `[out_channels, in_channels//groups, kernel_height, kernel_width]`.
        filter:         Low-pass filter for up/downsampling. Must be prepared beforehand by
                        calling upfirdn2d.setup_filter(). None = identity (default).
        up:             Integer upsampling factor (default: 1).
        down:           Integer downsampling factor (default: 1).
        padding:        Padding with respect to the upsampled image. Can be a single number
                        or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                        (default: 0).
        groups:         Split input channels into N groups (default: 1).
        flip_weight:    False = convolution, True = correlation (default: True).
        flip_filter:    False = convolution, True = correlation (default: False).
    """
    assert isinstance(up, int) and (up >= 1)
    assert isinstance(down, int) and (down >= 1)
    assert isinstance(groups, int) and (groups >= 1)
    out_channels, in_channels_per_group, kh, kw = w.shape
    fw, fh = _get_filter_size(filter)
    # 图片4条边上的padding
    px0, px1, py0, py1 = _parse_padding(padding)

    # Adjust padding to account for up/downsampling.
    if up > 1:
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2

    # Fast path: 1x1 convolution with downsampling only => downsample first, then convolve.
    if kw == 1 and kh == 1 and (down > 1 and up == 1):
        x = upfirdn2d(x, filter, down=down, padding=[px0, px1, py0, py1], flip_filter=flip_filter)
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        return x

    # Fast path: 1x1 convolution with upsampling only => convolve first, then upsample.
    if kw == 1 and kh == 1 and (up > 1 and down == 1):
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        x = upfirdn2d(x, filter, up=up, padding=[px0, px1, py0, py1], gain=up ** 2, flip_filter=flip_filter)
        return x

    # Fast path: downsampling only => use strided convolution.
    if down > 1 and up == 1:
        x = upfirdn2d(x, filter, padding=[px0, px1, py0, py1], flip_filter=flip_filter)
        x = _conv2d_wrapper(x=x, w=w, stride=down, groups=groups, flip_weight=flip_weight)
        return x

    # Fast path: upsampling with optional downsampling => use transpose strided convolution.
    if up > 1:
        if groups == 1:
            w = w.transpose((1, 0, 2, 3))
        else:
            w = w.reshape((groups, out_channels // groups, in_channels_per_group, kh, kw))
            w = w.transpose((0, 2, 1, 3, 4))
            w = w.reshape((groups * in_channels_per_group, out_channels // groups, kh, kw))
        px0 -= kw - 1
        px1 -= kw - up
        py0 -= kh - 1
        py1 -= kh - up
        pxt = max(min(-px0, -px1), 0)
        pyt = max(min(-py0, -py1), 0)
        x = _conv2d_wrapper(x=x, w=w, stride=up, padding=[pyt,pxt], groups=groups, transpose=True, flip_weight=(not flip_weight))
        x = upfirdn2d(x, filter, padding=[px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt], gain=up ** 2, flip_filter=flip_filter)
        if down > 1:
            x = upfirdn2d(x, filter, down=down, flip_filter=flip_filter)
        return x

    # Fast path: no up/downsampling, padding supported by the underlying implementation => use plain conv2d.
    if up == 1 and down == 1:
        if px0 == px1 and py0 == py1 and px0 >= 0 and py0 >= 0:
            return _conv2d_wrapper(x=x, w=w, padding=[py0,px0], groups=groups, flip_weight=flip_weight)

    # Fallback: Generic reference implementation.
    x = upfirdn2d(x, (filter if up > 1 else None), up=up, padding=[px0, px1, py0, py1], gain=up ** 2, flip_filter=flip_filter)
    x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
    if down > 1:
        x = upfirdn2d(x, filter, down=down, flip_filter=flip_filter)
    return x


def upsample2d(x, f, up=2, padding=0, flip_filter=False, gain=1):
    r"""Upsample a batch of 2D images using the given 2D FIR filter.

    By default, the result is padded so that its shape is a multiple of the input.
    User-specified padding is applied on top of that, with negative values
    indicating cropping. Pixels outside the image are assumed to be zero.

    Args:
        x:           Float32/float64/float16 input tensor of the shape
                     `[batch_size, num_channels, in_height, in_width]`.
        f:           Float32 FIR filter of the shape
                     `[filter_height, filter_width]` (non-separable),
                     `[filter_taps]` (separable), or
                     `None` (identity).
        up:          Integer upsampling factor. Can be a single int or a list/tuple
                     `[x, y]` (default: 1).
        padding:     Padding with respect to the output. Can be a single number or a
                     list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`
                     (default: 0).
        flip_filter: False = convolution, True = correlation (default: False).
        gain:        Overall scaling factor for signal magnitude (default: 1).

    Returns:
        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.
    """
    upx, upy = _parse_scaling(up)
    padx0, padx1, pady0, pady1 = _parse_padding(padding)
    fw, fh = _get_filter_size(f)
    p = [
        padx0 + (fw + upx - 1) // 2,
        padx1 + (fw - upx) // 2,
        pady0 + (fh + upy - 1) // 2,
        pady1 + (fh - upy) // 2,
    ]
    return upfirdn2d(x, f, up=up, padding=p, flip_filter=flip_filter, gain=gain*upx*upy)


class Conv2dLayer(nn.Layer):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d_setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

        # activation_funcs = {
        #     'linear':   dnnlib.EasyDict(func=lambda x, **_:         x,                                          def_alpha=0,    def_gain=1,             cuda_idx=1, ref='',  has_2nd_grad=False),
        #     'relu':     dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.relu(x),                def_alpha=0,    def_gain=np.sqrt(2),    cuda_idx=2, ref='y', has_2nd_grad=False),
        #     'lrelu':    dnnlib.EasyDict(func=lambda x, alpha, **_:  torch.nn.functional.leaky_relu(x, alpha),   def_alpha=0.2,  def_gain=np.sqrt(2),    cuda_idx=3, ref='y', has_2nd_grad=False),
        #     'tanh':     dnnlib.EasyDict(func=lambda x, **_:         torch.tanh(x),                              def_alpha=0,    def_gain=1,             cuda_idx=4, ref='y', has_2nd_grad=True),
        #     'sigmoid':  dnnlib.EasyDict(func=lambda x, **_:         torch.sigmoid(x),                           def_alpha=0,    def_gain=1,             cuda_idx=5, ref='y', has_2nd_grad=True),
        #     'elu':      dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.elu(x),                 def_alpha=0,    def_gain=1,             cuda_idx=6, ref='y', has_2nd_grad=True),
        #     'selu':     dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.selu(x),                def_alpha=0,    def_gain=1,             cuda_idx=7, ref='y', has_2nd_grad=True),
        #     'softplus': dnnlib.EasyDict(func=lambda x, **_:         torch.nn.functional.softplus(x),            def_alpha=0,    def_gain=1,             cuda_idx=8, ref='y', has_2nd_grad=True),
        #     'swish':    dnnlib.EasyDict(func=lambda x, **_:         torch.sigmoid(x) * x,                       def_alpha=0,    def_gain=np.sqrt(2),    cuda_idx=9, ref='x', has_2nd_grad=True),
        # }
        def_gain = 1.0
        if activation in ['relu', 'lrelu', 'swish']:  # 除了这些激活函数的def_gain = np.sqrt(2)，其余激活函数的def_gain = 1.0
            def_gain = np.sqrt(2)
        def_alpha = 0.0
        if activation in ['lrelu']:  # 除了这些激活函数的def_alpha = 0.2，其余激活函数的def_alpha = 0.0
            def_alpha = 0.2


        self.act_gain = def_gain

        if trainable:
            self.weight = self.create_parameter([out_channels, in_channels, kernel_size, kernel_size],
                                                default_initializer=paddle.nn.initializer.Normal())
            self.bias = self.create_parameter([out_channels], is_bias=True,
                                              default_initializer=paddle.nn.initializer.Constant(0.0)) if bias else None
        else:
            self.weight = self.create_parameter([out_channels, in_channels, kernel_size, kernel_size],
                                                default_initializer=paddle.nn.initializer.Constant(1.0))
            self.weight.stop_gradient = True
            self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * self.weight_gain
        b = paddle.cast(self.bias, dtype=x.dtype) if self.bias is not None else None
        flip_weight = (self.up == 1)  # slightly faster
        x = conv2d_resample(x=x, w=paddle.cast(w, dtype=x.dtype), filter=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)
        return x


class FullyConnectedLayer(nn.Layer):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = self.create_parameter([out_features, in_features],
                                            default_initializer=paddle.nn.initializer.Normal(mean=0.0, std=1.0 / lr_multiplier))
        self.bias = self.create_parameter([out_features], is_bias=True,
                                          default_initializer=paddle.nn.initializer.Constant(bias_init)) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x, dic2=None, pre_name=None):
        assert self.weight.dtype == x.dtype
        # w = paddle.cast(self.weight, dtype=x.dtype) * self.weight_gain
        w = self.weight * self.weight_gain
        b = self.bias
        if b is not None:
            assert b.dtype == x.dtype
            # b = paddle.cast(b, dtype=x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            # out = paddle.addmm(b.unsqueeze(0), x, w.t())   # 因为paddle.addmm()没有实现二阶梯度，所以用其它等价实现。
            out = paddle.matmul(x, w, transpose_y=True) + b.unsqueeze(0)
            if dic2 is not None:
                dout_dx = paddle.grad(
                    outputs=[out.sum()],
                    inputs=[x],
                    create_graph=False,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
                    retain_graph=True)[0]
                aaaaaaaaaaa1 = dic2[pre_name + '_dout_dx']
                aaaaaaaaaaa2 = dout_dx.numpy()
                ddd = np.sum((dic2[pre_name + '_dout_dx'] - dout_dx.numpy()) ** 2)
                print('ddd=%.6f' % ddd)
        else:
            r = x.matmul(w.t())
            out = bias_act(r, b, act=self.activation)
            if dic2 is not None:
                dr_dx = paddle.grad(
                    outputs=[r.sum()],
                    inputs=[x],
                    create_graph=False,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
                    retain_graph=True)[0]
                aaaaaaaaaaa1 = dic2[pre_name + '_dr_dx']
                aaaaaaaaaaa2 = dr_dx.numpy()
                ddd = np.sum((dic2[pre_name + '_dr_dx'] - dr_dx.numpy()) ** 2)
                print('ddd=%.6f' % ddd)

                dout_dr = paddle.grad(
                    outputs=[out.sum()],
                    inputs=[r],
                    create_graph=False,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
                    retain_graph=True)[0]
                aaaaaaaaaaa1 = dic2[pre_name + '_dout_dr']
                aaaaaaaaaaa2 = dout_dr.numpy()
                ddd = np.sum((dic2[pre_name + '_dout_dr'] - dout_dr.numpy()) ** 2)
                print('ddd=%.6f' % ddd)
        return out


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(axis=dim, keepdim=True) + eps).rsqrt()


class StyleGANv2ADA_MappingNetwork(nn.Layer):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
    ):
        super().__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim
        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)
        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', paddle.zeros([w_dim]))

    def forward(self, z, c, dic2, pre_name, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        if self.z_dim > 0:
            # 转换类型会导致与pytorch的梯度不同。
            # temp1 = paddle.cast(z, dtype='float32')
            x = normalize_2nd_moment(z)

            dx_dz1 = paddle.grad(
                outputs=[x.sum()],
                inputs=[z],
                create_graph=False,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
                retain_graph=True)[0]
            aaaaaaaaaaa1 = dic2[pre_name + '_dx_dz1']
            aaaaaaaaaaa2 = dx_dz1.numpy()
            ddd = np.sum((dic2[pre_name + '_dx_dz1'] - dx_dz1.numpy()) ** 2)
            print('ddd=%.6f' % ddd)
        if self.c_dim > 0:
            temp2 = paddle.cast(c, dtype='float32')
            y = normalize_2nd_moment(self.embed(temp2))
            x = paddle.concat([x, y], 1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x, dic2, pre_name + '_fc%d'%idx)
        dx_dz2 = paddle.grad(
            outputs=[x.sum()],
            inputs=[z],
            create_graph=False,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
            retain_graph=True)[0]
        aaaaaaaaaaa1 = dic2[pre_name + '_dx_dz2']
        aaaaaaaaaaa2 = dx_dz2.numpy()
        ddd = np.sum((dic2[pre_name + '_dx_dz2'] - dx_dz2.numpy()) ** 2)
        print('ddd=%.6f' % ddd)
        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            temp3 = x.detach().mean(axis=0)
            # temp3 = temp3.lerp(self.w_avg, self.w_avg_beta)
            temp3 = temp3 + self.w_avg_beta * (self.w_avg - temp3)
            # self.w_avg.copy_(temp3)
            self.w_avg = temp3

        # Broadcast.
        if self.num_ws is not None:
            x = x.unsqueeze(1).tile([1, self.num_ws, 1])
            # new_x = []
            # for j in range(self.num_ws):
            #     new_x.append(x.unsqueeze(1).clone())
            # new_x = paddle.concat(new_x, 1)
            # x = new_x

        # Apply truncation.
        if truncation_psi != 1:
            assert self.w_avg_beta is not None
            if self.num_ws is None or truncation_cutoff is None:
                # x = self.w_avg.lerp(x, truncation_psi)
                x = self.w_avg + truncation_psi * (x - self.w_avg)
            else:
                # x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
                x[:, :truncation_cutoff] = self.w_avg + truncation_psi * (x[:, :truncation_cutoff] - self.w_avg)
        return x

def modulated_conv2d(
    x,                          # Input tensor of shape [batch_size, in_channels, in_height, in_width].
    weight,                     # Weight tensor of shape [out_channels, in_channels, kernel_height, kernel_width].
    styles,                     # Modulation coefficients of shape [batch_size, in_channels].
    noise           = None,     # Optional noise tensor to add to the output activations.
    up              = 1,        # Integer upsampling factor.
    down            = 1,        # Integer downsampling factor.
    padding         = 0,        # Padding with respect to the upsampled image.
    resample_filter = None,     # Low-pass filter to apply when resampling activations. Must be prepared beforehand by calling upfirdn2d.setup_filter().
    demodulate      = True,     # Apply weight demodulation?
    flip_weight     = True,     # False = convolution, True = correlation (matches torch.nn.functional.conv2d).
    fused_modconv   = True,     # Perform modulation, convolution, and demodulation as a single fused operation?
):
    batch_size = x.shape[0]
    out_channels, in_channels, kh, kw = weight.shape
    # misc.assert_shape(weight, [out_channels, in_channels, kh, kw]) # [OIkk]
    # misc.assert_shape(x, [batch_size, in_channels, None, None]) # [NIHW]
    # misc.assert_shape(styles, [batch_size, in_channels]) # [NI]

    # Pre-normalize inputs to avoid FP16 overflow.
    if x.dtype == paddle.float16 and demodulate:
        d0, d1, d2, d3 = weight.shape
        weight_temp = weight.reshape((d0, d1, d2 * d3))
        weight_temp = paddle.norm(weight_temp, p=np.inf, axis=[1, 2], keepdim=True)
        weight_temp = weight_temp.reshape((d0, 1, 1, 1))
        weight = weight * (1 / np.sqrt(in_channels * kh * kw) / weight_temp) # max_Ikk
        styles_temp = paddle.norm(styles, p=np.inf, axis=1, keepdim=True)
        styles = styles / styles_temp # max_I
        print('aaaaaaaaaaaaaaaaaa')

    # Calculate per-sample weights and demodulation coefficients.
    w = None
    dcoefs = None
    if demodulate or fused_modconv:
        w = weight.unsqueeze(0)  # [NOIkk]
        w = w * styles.reshape((batch_size, 1, -1, 1, 1))  # [NOIkk]
    if demodulate:
        dcoefs = (w.square().sum(axis=[2, 3, 4]) + 1e-8).rsqrt()  # [NO]
    if demodulate and fused_modconv:
        w = w * dcoefs.reshape((batch_size, -1, 1, 1, 1))  # [NOIkk]

    # Execute by scaling the activations before and after the convolution.
    if not fused_modconv:
        assert styles.dtype == x.dtype
        # x = x * paddle.cast(styles, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
        x = x * styles.reshape((batch_size, -1, 1, 1))
        assert weight.dtype == x.dtype
        # x = conv2d_resample(x=x, w=paddle.cast(weight, dtype=x.dtype), filter=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        x = conv2d_resample(x=x, w=weight, filter=resample_filter, up=up, down=down, padding=padding, flip_weight=flip_weight)
        if demodulate and noise is not None:
            assert dcoefs.dtype == x.dtype
            assert noise.dtype == x.dtype
            # x = x * paddle.cast(dcoefs, dtype=x.dtype).reshape((batch_size, -1, 1, 1)) + paddle.cast(noise, dtype=x.dtype)
            x = x * paddle.cast(dcoefs, dtype=x.dtype).reshape((batch_size, -1, 1, 1)) + paddle.cast(noise, dtype=x.dtype)
        elif demodulate:
            assert dcoefs.dtype == x.dtype
            # x = x * paddle.cast(dcoefs, dtype=x.dtype).reshape((batch_size, -1, 1, 1))
            x = x * dcoefs.reshape((batch_size, -1, 1, 1))
        elif noise is not None:
            assert noise.dtype == x.dtype
            # x = x + paddle.cast(noise, dtype=x.dtype)
            x = x + noise
        return x

    # Execute as one fused op using grouped convolution.
    x = x.reshape((1, -1, *x.shape[2:]))
    w = w.reshape((-1, in_channels, kh, kw))
    assert w.dtype == x.dtype
    # x = conv2d_resample(x=x, w=paddle.cast(w, dtype=x.dtype), filter=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = conv2d_resample(x=x, w=w, filter=resample_filter, up=up, down=down, padding=padding, groups=batch_size, flip_weight=flip_weight)
    x = x.reshape((batch_size, -1, *x.shape[2:]))
    if noise is not None:
        x = x + noise
    return x


class Spade_Conv2dLayer(nn.Layer):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 kernel_size,  # Width and height of the convolution kernel.
                 bias=True,  # Apply additive bias before the activation function?
                 activation='relu',  # Activation function: 'relu', 'lrelu', etc.
                 up=1,  # Integer upsampling factor.
                 down=1,  # Integer downsampling factor.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
                 channels_last=False,  # Expect the input to have memory_format=channels_last?
                 trainable=True,  # Update the weights of this layer during training?
                 ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d_setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))

        def_gain = 1.0
        if activation in ['relu', 'lrelu', 'swish']:  # 除了这些激活函数的def_gain = np.sqrt(2)，其余激活函数的def_gain = 1.0
            def_gain = np.sqrt(2)
        self.act_gain = def_gain

        # 假设屎山的channels_last都是False
        assert channels_last == False
        # memory_format = torch.channels_last if channels_last else torch.contiguous_format
        # weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        self.bias = None
        if trainable:
            self.weight = self.create_parameter([out_channels, in_channels, kernel_size, kernel_size],
                                                default_initializer=paddle.nn.initializer.Normal())
            if bias is not None:
                if bias != False:
                    self.bias = self.create_parameter([out_channels, ],
                                                      default_initializer=paddle.nn.initializer.Constant(0.0))
        else:
            self.weight = self.create_parameter([out_channels, in_channels, kernel_size, kernel_size],
                                                default_initializer=paddle.nn.initializer.Normal())
            self.weight.stop_gradient = True
            if bias is not None:
                if bias != False:
                    self.bias = self.create_parameter([out_channels, ],
                                                      default_initializer=paddle.nn.initializer.Constant(0.0))
                    self.bias.stop_gradient = True

    def forward(self, x, gain=1, no_act=False):
        w = self.weight * self.weight_gain
        b = paddle.cast(self.bias, dtype=x.dtype) if self.bias is not None else None

        if not no_act:
            act_gain = self.act_gain * gain
            act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
            x = bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)

        flip_weight = (self.up == 1)  # slightly faster
        x = conv2d_resample(x=x, w=paddle.cast(w, dtype=x.dtype), filter=self.resample_filter, up=self.up, down=self.down,
                            padding=self.padding, flip_weight=flip_weight)

        return x

class Spade_Norm_Block(nn.Layer):
    def __init__(self,
        in_channels,
        norm_channels,
    ):
        super().__init__()
        self.conv_mlp = Spade_Conv2dLayer(in_channels, norm_channels, kernel_size=3, bias=False)
        self.conv_mlp_act = nn.ReLU()
        self.conv_gamma = Spade_Conv2dLayer(norm_channels, norm_channels, kernel_size=3, bias=False)
        self.conv_beta = Spade_Conv2dLayer(norm_channels, norm_channels, kernel_size=3, bias=False)

        self.param_free_norm = nn.InstanceNorm2D(norm_channels, weight_attr=False, bias_attr=False)

    def forward(self, x, denorm_feats):
        normalized = self.param_free_norm(x)
        actv = self.conv_mlp(denorm_feats, no_act=True)
        actv = self.conv_mlp_act(actv)
        gamma = self.conv_gamma(actv, no_act=True)
        beta = self.conv_beta(actv, no_act=True)

        out = normalized * (1+gamma) + beta
        return out


class Spade_ResBlock(nn.Layer):
    def __init__(self,
                 in_channels,  # Number of input channels.
                 out_channels,  # Number of output channels.
                 kernel_size=3,  # Width and height of the convolution kernel.
                 bias=True,  # Apply additive bias before the activation function?
                 activation='linear',  # Activation function: 'relu', 'lrelu', etc.
                 up=1,  # Integer upsampling factor.
                 down=1,  # Integer downsampling factor.
                 resample_filter=[1, 3, 3, 1],  # Low-pass filter to apply when resampling activations.
                 conv_clamp=None,  # Clamp the output to +-X, None = disable clamping.
                 channels_last=False,  # Expect the input to have memory_format=channels_last?
                 trainable=True,  # Update the weights of this layer during training?
                 resolution=128,
                 ):
        super().__init__()
        self.register_buffer('resample_filter', upfirdn2d_setup_filter(resample_filter))

        self.conv = Spade_Conv2dLayer(in_channels, in_channels, kernel_size=3, bias=False,
                                      resample_filter=resample_filter, conv_clamp=conv_clamp,
                                      channels_last=channels_last)
        self.conv0 = Spade_Conv2dLayer(in_channels, out_channels, kernel_size=3, bias=False,
                                       resample_filter=resample_filter, conv_clamp=conv_clamp,
                                       channels_last=channels_last)
        self.conv1 = Spade_Conv2dLayer(out_channels, out_channels, kernel_size=3, bias=False,
                                       resample_filter=resample_filter, conv_clamp=conv_clamp,
                                       channels_last=channels_last)
        self.skip = Spade_Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False,
                                      resample_filter=resample_filter, conv_clamp=conv_clamp,
                                      channels_last=channels_last)

        if resolution == 128:
            feat_channels = 128 * 2
        else:
            feat_channels = 64 * 2
        self.spade_skip = Spade_Norm_Block(feat_channels, in_channels)
        self.spade0 = Spade_Norm_Block(feat_channels, in_channels)
        self.spade1 = Spade_Norm_Block(feat_channels, out_channels)

    def forward(self, x, denorm_feat):
        x = self.conv(x, no_act=True)

        y = self.skip(self.spade_skip(x, denorm_feat), gain=np.sqrt(0.5))
        x = self.conv0(self.spade0(x, denorm_feat))
        x = self.conv1(self.spade1(x, denorm_feat), gain=np.sqrt(0.5))

        x = y + x
        return x

class SynthesisLayer(nn.Layer):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        w_dim,                          # Intermediate latent (W) dimensionality.
        resolution,                     # Resolution of this layer.
        kernel_size     = 3,            # Convolution kernel size.
        up              = 1,            # Integer upsampling factor.
        use_noise       = True,         # Enable noise input?
        activation      = 'lrelu',      # Activation function: 'relu', 'lrelu', etc.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        channels_last   = False,        # Use channels_last format for the weights?
    ):
        super().__init__()
        self.resolution = resolution
        self.up = up
        self.use_noise = use_noise
        self.use_noise = False
        self.activation = activation
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d_setup_filter(resample_filter))
        self.padding = kernel_size // 2

        def_gain = 1.0
        if activation in ['relu', 'lrelu', 'swish']:  # 除了这些激活函数的def_gain = np.sqrt(2)，其余激活函数的def_gain = 1.0
            def_gain = np.sqrt(2)
        self.act_gain = def_gain

        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        # 假设屎山的channels_last都是False
        assert channels_last == False
        # memory_format = torch.channels_last if channels_last else torch.contiguous_format
        # self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.weight = self.create_parameter([out_channels, in_channels, kernel_size, kernel_size],
                                            default_initializer=paddle.nn.initializer.Normal())

        if use_noise:
            self.register_buffer('noise_const', paddle.randn([resolution, resolution]))
            # self.noise_strength = torch.nn.Parameter(torch.zeros([]))
            # 噪声强度（振幅）
            self.noise_strength = self.create_parameter([1, ],
                                                        default_initializer=paddle.nn.initializer.Constant(0.0))
        # self.bias = torch.nn.Parameter(torch.zeros([out_channels]))
        self.bias = self.create_parameter([out_channels, ],
                                          default_initializer=paddle.nn.initializer.Constant(0.0))

    def forward(self, x, w, dic2, pre_name, noise_mode='random', fused_modconv=True, gain=1):
        assert noise_mode in ['random', 'const', 'none']
        in_resolution = self.resolution // self.up
        styles = self.affine(w)

        dstyles_dw = paddle.grad(
            outputs=[styles.sum()],
            inputs=[w],
            create_graph=False,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
            retain_graph=True)[0]
        aaaaaaaaaaa1 = dic2[pre_name + 'dstyles_dw']
        aaaaaaaaaaa2 = dstyles_dw.numpy()
        ddd = np.sum((aaaaaaaaaaa1 - aaaaaaaaaaa2) ** 2)
        print('ddd=%.6f' % ddd)

        noise = None
        if self.use_noise and noise_mode == 'random':
            noise = paddle.randn([x.shape[0], 1, self.resolution, self.resolution]) * self.noise_strength
        if self.use_noise and noise_mode == 'const':
            noise = self.noise_const * self.noise_strength

        flip_weight = (self.up == 1) # slightly faster
        img2 = modulated_conv2d(x=x, weight=self.weight, styles=styles, noise=noise, up=self.up,
            padding=self.padding, resample_filter=self.resample_filter, flip_weight=flip_weight, fused_modconv=fused_modconv)


        dimg2_dx = paddle.grad(
            outputs=[img2.sum()],
            inputs=[x],
            create_graph=False,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
            retain_graph=True)[0]
        aaaaaaaaaaa1 = dic2[pre_name + 'dimg2_dx']
        aaaaaaaaaaa2 = dimg2_dx.numpy()
        ddd = np.sum((aaaaaaaaaaa1 - aaaaaaaaaaa2) ** 2)
        print('ddd=%.6f' % ddd)

        dimg2_dw = paddle.grad(
            outputs=[img2.sum()],
            inputs=[w],
            create_graph=False,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
            retain_graph=True)[0]
        aaaaaaaaaaa1 = dic2[pre_name + 'dimg2_dw']
        aaaaaaaaaaa2 = dimg2_dw.numpy()
        ddd = np.sum((aaaaaaaaaaa1 - aaaaaaaaaaa2) ** 2)
        print('ddd=%.6f' % ddd)

        dimg2_dstyles = paddle.grad(
            outputs=[img2.sum()],
            inputs=[styles],
            create_graph=False,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
            retain_graph=True)[0]
        aaaaaaaaaaa3 = dic2[pre_name + 'dimg2_dstyles']
        aaaaaaaaaaa4 = dimg2_dstyles.numpy()
        ddd = np.sum((aaaaaaaaaaa3 - aaaaaaaaaaa4) ** 2)
        print('ddd=%.6f' % ddd)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        img3 = bias_act(img2, paddle.cast(self.bias, dtype=x.dtype), act=self.activation, gain=act_gain, clamp=act_clamp)


        dimg3_dimg2 = paddle.grad(
            outputs=[img3.sum()],
            inputs=[img2],
            create_graph=False,  # 最终loss里包含梯度，需要求梯度的梯度，所以肯定需要建立反向图。
            retain_graph=True)[0]
        aaaaaaaaaaa1 = dic2[pre_name + 'dimg3_dimg2']
        aaaaaaaaaaa2 = dimg3_dimg2.numpy()
        ddd = np.sum((aaaaaaaaaaa1 - aaaaaaaaaaa2) ** 2)
        print('ddd=%.6f' % ddd)

        return img3



class ToRGBLayer(nn.Layer):
    def __init__(self, in_channels, out_channels, w_dim, kernel_size=1, conv_clamp=None, channels_last=False):
        super().__init__()
        self.conv_clamp = conv_clamp
        self.affine = FullyConnectedLayer(w_dim, in_channels, bias_init=1)
        # 假设屎山的channels_last都是False
        assert channels_last == False
        # memory_format = torch.channels_last if channels_last else torch.contiguous_format
        # self.weight = torch.nn.Parameter(torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format))
        self.weight = self.create_parameter([out_channels, in_channels, kernel_size, kernel_size],
                                            default_initializer=paddle.nn.initializer.Normal())
        self.bias = self.create_parameter([out_channels, ],
                                          default_initializer=paddle.nn.initializer.Constant(0.0))
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))


    def forward(self, x, w, dic2, pre_name, fused_modconv=True):
        styles = self.affine(w) * self.weight_gain
        x = modulated_conv2d(x=x, weight=self.weight, styles=styles, demodulate=False, fused_modconv=fused_modconv)
        x = bias_act(x, paddle.cast(self.bias, dtype=x.dtype), clamp=self.conv_clamp)
        return x


'''
兼容原版仓库的
SynthesisBlockFull
SynthesisBlockV18
'''
class SynthesisBlock(nn.Layer):
    def __init__(self,
        in_channels,                        # Number of input channels, 0 = first block.
        out_channels,                       # Number of output channels.
        w_dim,                              # Intermediate latent (W) dimensionality.
        resolution,                         # Resolution of this block.
        img_channels,                       # Number of output color channels.
        is_last,                            # Is this the last block?
        architecture        = 'skip',       # Architecture: 'orig', 'skip', 'resnet'.
        resample_filter     = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp          = None,         # Clamp the output of convolution layers to +-X, None = disable clamping.
        use_fp16            = False,        # Use FP16 for this block?
        fp16_channels_last  = False,        # Use channels-last memory format with FP16?
        **layer_kwargs,                     # Arguments for SynthesisLayer.
    ):
        assert architecture in ['orig', 'skip', 'resnet']
        super().__init__()
        self.in_channels = in_channels
        self.w_dim = w_dim
        self.resolution = resolution
        self.img_channels = img_channels
        self.is_last = is_last
        self.architecture = architecture
        self.use_fp16 = use_fp16
        self.channels_last = (use_fp16 and fp16_channels_last)
        self.register_buffer('resample_filter', upfirdn2d_setup_filter(resample_filter))
        self.num_conv = 0
        self.num_torgb = 0

        if in_channels == 0:
            self.const = self.create_parameter([out_channels, resolution, resolution],
                                               default_initializer=paddle.nn.initializer.Normal())

        if in_channels != 0:
            self.conv0 = SynthesisLayer(in_channels, out_channels, w_dim=w_dim, resolution=resolution, up=2,
                resample_filter=resample_filter, conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
            self.num_conv += 1

        self.conv1 = SynthesisLayer(out_channels, out_channels, w_dim=w_dim, resolution=resolution,
            conv_clamp=conv_clamp, channels_last=self.channels_last, **layer_kwargs)
        self.num_conv += 1

        if is_last or architecture == 'skip':
            self.torgb = ToRGBLayer(out_channels, img_channels, w_dim=w_dim,
                conv_clamp=conv_clamp, channels_last=self.channels_last)
            self.num_torgb += 1

        if in_channels != 0 and architecture == 'resnet':
            self.skip = Conv2dLayer(in_channels, out_channels, kernel_size=1, bias=False, up=2,
                resample_filter=resample_filter, channels_last=self.channels_last)

    def forward(self, x, img, ws, ws_idx, dic2, pre_name, force_fp32=False, fused_modconv=None, **layer_kwargs):
    # def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
    #     w_iter = iter(ws.unbind(axis=1))
        dtype = paddle.float16 if self.use_fp16 and not force_fp32 else paddle.float32
        # 假设屎山的channels_last都是False
        assert self.channels_last == False
        # memory_format = torch.channels_last if self.channels_last and not force_fp32 else torch.contiguous_format
        if fused_modconv is None:
            fused_modconv = (not self.training) and (dtype == paddle.float32 or int(x.shape[0]) == 1)

        # Input.
        if self.in_channels == 0:
            x = paddle.cast(self.const, dtype=dtype)
            # x = x.unsqueeze(0).tile([ws.shape[0], 1, 1, 1])
            x = x.unsqueeze(0).tile([ws[0].shape[0], 1, 1, 1])
        else:
            x = paddle.cast(x, dtype=dtype)

        # Main layers.
        if self.in_channels == 0:
            # x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv1(x, ws[ws_idx], dic2, pre_name + '_conv1', fused_modconv=fused_modconv, **layer_kwargs); ws_idx+=1
        elif self.architecture == 'resnet':
            y = self.skip(x, gain=np.sqrt(0.5))
            # x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            # x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
            x = self.conv0(x, ws[ws_idx], dic2, pre_name + '_conv0', fused_modconv=fused_modconv, **layer_kwargs); ws_idx+=1
            x = self.conv1(x, ws[ws_idx], dic2, pre_name + '_conv1', fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs); ws_idx+=1
            x = y.add_(x)
        else:
            # x = self.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            # x = self.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
            x = self.conv0(x, ws[ws_idx], dic2, pre_name + '_conv0', fused_modconv=fused_modconv, **layer_kwargs); ws_idx+=1
            x = self.conv1(x, ws[ws_idx], dic2, pre_name + '_conv1', fused_modconv=fused_modconv, **layer_kwargs); ws_idx+=1

        # ToRGB.
        if img is not None:
            img = upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            # y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = self.torgb(x, ws[ws_idx], dic2, pre_name + '_torgb', fused_modconv=fused_modconv); ws_idx+=1
            y = paddle.cast(y, dtype=paddle.float32)
            img = img + y if img is not None else y

        assert x.dtype == dtype
        assert img is None or img.dtype == paddle.float32
        return x, img



class StyleGANv2ADA_SynthesisNetwork(nn.Layer):
    def __init__(self,
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output image resolution.
        img_channels,               # Number of color channels.
        channel_base    = 32768,    # Overall multiplier for the number of channels.
        channel_max     = 512,      # Maximum number of channels in any layer.
        num_fp16_res    = 0,        # Use FP16 for the N highest resolutions.
        **block_kwargs,             # Arguments for SynthesisBlock.
    ):
        assert img_resolution >= 4 and img_resolution & (img_resolution - 1) == 0  # 分辨率是2的n次方
        super().__init__()
        self.w_dim = w_dim
        self.img_resolution = img_resolution
        self.img_resolution_log2 = int(np.log2(img_resolution))
        self.img_channels = img_channels
        self.block_resolutions = [2 ** i for i in range(2, self.img_resolution_log2 + 1)]
        channels_dict = {res: min(channel_base // res, channel_max) for res in self.block_resolutions}
        fp16_resolution = max(2 ** (self.img_resolution_log2 + 1 - num_fp16_res), 8)

        self.num_ws = 0
        for res in self.block_resolutions:
            in_channels = channels_dict[res // 2] if res > 4 else 0
            out_channels = channels_dict[res]
            use_fp16 = (res >= fp16_resolution)
            use_fp16 = False
            is_last = (res == self.img_resolution)
            block = SynthesisBlock(in_channels, out_channels, w_dim=w_dim, resolution=res,
                img_channels=img_channels, is_last=is_last, use_fp16=use_fp16, **block_kwargs)
            self.num_ws += block.num_conv
            if is_last:
                self.num_ws += block.num_torgb
            setattr(self, f'b{res}', block)

    def forward(self, ws, dic2, pre_name, **block_kwargs):
        # block_ws = []
        # ws2 = paddle.cast(ws, dtype='float32')

        starts = []
        single_ws = []
        _dim = ws.shape[1]
        for i in range(_dim):
            single_ws.append(ws[:, i, :])
        w_idx = 0
        for res in self.block_resolutions:
            block = getattr(self, f'b{res}')
            # block_ws.append(ws.narrow(1, w_idx, block.num_conv + block.num_torgb))
            # block_ws.append(ws[:, w_idx:w_idx + block.num_conv + block.num_torgb, :])
            starts.append(w_idx)
            w_idx += block.num_conv

        x = img = None
        # for res, cur_ws in zip(self.block_resolutions, block_ws):
        #     block = getattr(self, f'b{res}')
        #     x, img = block(x, img, cur_ws, **block_kwargs)
        for res, start in zip(self.block_resolutions, starts):
            block = getattr(self, f'b{res}')
            x, img = block(x, img, single_ws, start, dic2, pre_name + '_b%d'%res, **block_kwargs)
        # return img, block_ws
        return img, single_ws




#----------------------------------------------------------------------------
# Coefficients of various wavelet decomposition low-pass filters.

wavelets = {
    'haar': [0.7071067811865476, 0.7071067811865476],
    'db1':  [0.7071067811865476, 0.7071067811865476],
    'db2':  [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'db3':  [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'db4':  [-0.010597401784997278, 0.032883011666982945, 0.030841381835986965, -0.18703481171888114, -0.02798376941698385, 0.6308807679295904, 0.7148465705525415, 0.23037781330885523],
    'db5':  [0.003335725285001549, -0.012580751999015526, -0.006241490213011705, 0.07757149384006515, -0.03224486958502952, -0.24229488706619015, 0.13842814590110342, 0.7243085284385744, 0.6038292697974729, 0.160102397974125],
    'db6':  [-0.00107730108499558, 0.004777257511010651, 0.0005538422009938016, -0.031582039318031156, 0.02752286553001629, 0.09750160558707936, -0.12976686756709563, -0.22626469396516913, 0.3152503517092432, 0.7511339080215775, 0.4946238903983854, 0.11154074335008017],
    'db7':  [0.0003537138000010399, -0.0018016407039998328, 0.00042957797300470274, 0.012550998556013784, -0.01657454163101562, -0.03802993693503463, 0.0806126091510659, 0.07130921926705004, -0.22403618499416572, -0.14390600392910627, 0.4697822874053586, 0.7291320908465551, 0.39653931948230575, 0.07785205408506236],
    'db8':  [-0.00011747678400228192, 0.0006754494059985568, -0.0003917403729959771, -0.00487035299301066, 0.008746094047015655, 0.013981027917015516, -0.04408825393106472, -0.01736930100202211, 0.128747426620186, 0.00047248457399797254, -0.2840155429624281, -0.015829105256023893, 0.5853546836548691, 0.6756307362980128, 0.3128715909144659, 0.05441584224308161],
    'sym2': [-0.12940952255092145, 0.22414386804185735, 0.836516303737469, 0.48296291314469025],
    'sym3': [0.035226291882100656, -0.08544127388224149, -0.13501102001039084, 0.4598775021193313, 0.8068915093133388, 0.3326705529509569],
    'sym4': [-0.07576571478927333, -0.02963552764599851, 0.49761866763201545, 0.8037387518059161, 0.29785779560527736, -0.09921954357684722, -0.012603967262037833, 0.0322231006040427],
    'sym5': [0.027333068345077982, 0.029519490925774643, -0.039134249302383094, 0.1993975339773936, 0.7234076904024206, 0.6339789634582119, 0.01660210576452232, -0.17532808990845047, -0.021101834024758855, 0.019538882735286728],
    'sym6': [0.015404109327027373, 0.0034907120842174702, -0.11799011114819057, -0.048311742585633, 0.4910559419267466, 0.787641141030194, 0.3379294217276218, -0.07263752278646252, -0.021060292512300564, 0.04472490177066578, 0.0017677118642428036, -0.007800708325034148],
    'sym7': [0.002681814568257878, -0.0010473848886829163, -0.01263630340325193, 0.03051551316596357, 0.0678926935013727, -0.049552834937127255, 0.017441255086855827, 0.5361019170917628, 0.767764317003164, 0.2886296317515146, -0.14004724044296152, -0.10780823770381774, 0.004010244871533663, 0.010268176708511255],
    'sym8': [-0.0033824159510061256, -0.0005421323317911481, 0.03169508781149298, 0.007607487324917605, -0.1432942383508097, -0.061273359067658524, 0.4813596512583722, 0.7771857517005235, 0.3644418948353314, -0.05194583810770904, -0.027219029917056003, 0.049137179673607506, 0.003808752013890615, -0.01495225833704823, -0.0003029205147213668, 0.0018899503327594609],
}

#----------------------------------------------------------------------------
# Helpers for constructing transformation matrices.


_constant_cache = dict()


def constant(value, shape=None, dtype=None):
    value = np.asarray(value)
    if shape is not None:
        shape = tuple(shape)
    if dtype is None:
        dtype = paddle.get_default_dtype()

    key = (value.shape, value.dtype, value.tobytes(), shape, dtype)
    tensor = _constant_cache.get(key, None)
    if tensor is None:
        # tensor = paddle.to_tensor(value.copy(), dtype=dtype)
        if isinstance(value, np.ndarray) and shape is None:
            tensor = paddle.to_tensor(value, dtype=dtype)
        else:
            tensor = paddle.ones(shape, dtype=dtype) * value
        _constant_cache[key] = tensor
    return tensor

def matrix(*rows):
    assert all(len(row) == len(rows[0]) for row in rows)
    elems = [x for row in rows for x in row]
    ref = [x for x in elems if isinstance(x, paddle.Tensor)]
    if len(ref) == 0:
        return constant(np.asarray(rows))
    elems = [x if isinstance(x, paddle.Tensor) else constant(x, shape=ref[0].shape) for x in elems]
    bbbbbbb = paddle.stack(elems, axis=-1)
    bbbbbbbb = bbbbbbb.reshape((tuple(ref[0].shape) + (len(rows), -1)))
    return bbbbbbbb

def translate2d(tx, ty, **kwargs):
    return matrix(
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1],
        **kwargs)

def translate3d(tx, ty, tz, **kwargs):
    return matrix(
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1],
        **kwargs)

def scale2d(sx, sy, **kwargs):
    return matrix(
        [sx, 0,  0],
        [0,  sy, 0],
        [0,  0,  1],
        **kwargs)

def scale3d(sx, sy, sz, **kwargs):
    return matrix(
        [sx, 0,  0,  0],
        [0,  sy, 0,  0],
        [0,  0,  sz, 0],
        [0,  0,  0,  1],
        **kwargs)

def rotate2d(theta, **kwargs):
    return matrix(
        [paddle.cos(theta), paddle.sin(-theta), 0],
        [paddle.sin(theta), paddle.cos(theta),  0],
        [0,                0,                 1],
        **kwargs)

def rotate3d(v, theta, **kwargs):
    if v.ndim == 1:
        vx = v[0]; vy = v[1]; vz = v[2]
    else:
        vx = v[..., 0]; vy = v[..., 1]; vz = v[..., 2]
    s = paddle.sin(theta); c = paddle.cos(theta); cc = 1 - c
    return matrix(
        [vx*vx*cc+c,    vx*vy*cc-vz*s, vx*vz*cc+vy*s, 0],
        [vy*vx*cc+vz*s, vy*vy*cc+c,    vy*vz*cc-vx*s, 0],
        [vz*vx*cc-vy*s, vz*vy*cc+vx*s, vz*vz*cc+c,    0],
        [0,             0,             0,             1],
        **kwargs)

def translate2d_inv(tx, ty, **kwargs):
    return translate2d(-tx, -ty, **kwargs)

def scale2d_inv(sx, sy, **kwargs):
    return scale2d(1 / sx, 1 / sy, **kwargs)

def rotate2d_inv(theta, **kwargs):
    return rotate2d(-theta, **kwargs)

#----------------------------------------------------------------------------
# Versatile image augmentation pipeline from the paper
# "Training Generative Adversarial Networks with Limited Data".
#
# All augmentations are disabled by default; individual augmentations can
# be enabled by setting their probability multipliers to 1.


class StyleGANv2ADA_AugmentPipe(nn.Layer):
    def __init__(self,
        xflip=0, rotate90=0, xint=0, xint_max=0.125,
        scale=0, rotate=0, aniso=0, xfrac=0, scale_std=0.2, rotate_max=1, aniso_std=0.2, xfrac_std=0.125,
        brightness=0, contrast=0, lumaflip=0, hue=0, saturation=0, brightness_std=0.2, contrast_std=0.5, hue_max=1, saturation_std=1,
        imgfilter=0, imgfilter_bands=[1,1,1,1], imgfilter_std=1,
        noise=0, cutout=0, noise_std=0.1, cutout_size=0.5,
    ):
        super().__init__()
        self.register_buffer('p', paddle.ones([1, ], dtype='float32'))       # Overall multiplier for augmentation probability.

        # Pixel blitting.
        self.xflip            = float(xflip)            # Probability multiplier for x-flip.
        self.rotate90         = float(rotate90)         # Probability multiplier for 90 degree rotations.
        self.xint             = float(xint)             # Probability multiplier for integer translation.
        self.xint_max         = float(xint_max)         # Range of integer translation, relative to image dimensions.

        # General geometric transformations.
        self.scale            = float(scale)            # Probability multiplier for isotropic scaling.
        self.rotate           = float(rotate)           # Probability multiplier for arbitrary rotation.
        self.aniso            = float(aniso)            # Probability multiplier for anisotropic scaling.
        self.xfrac            = float(xfrac)            # Probability multiplier for fractional translation.
        self.scale_std        = float(scale_std)        # Log2 standard deviation of isotropic scaling.
        self.rotate_max       = float(rotate_max)       # Range of arbitrary rotation, 1 = full circle.
        self.aniso_std        = float(aniso_std)        # Log2 standard deviation of anisotropic scaling.
        self.xfrac_std        = float(xfrac_std)        # Standard deviation of frational translation, relative to image dimensions.

        # Color transformations.
        self.brightness       = float(brightness)       # Probability multiplier for brightness.
        self.contrast         = float(contrast)         # Probability multiplier for contrast.
        self.lumaflip         = float(lumaflip)         # Probability multiplier for luma flip.
        self.hue              = float(hue)              # Probability multiplier for hue rotation.
        self.saturation       = float(saturation)       # Probability multiplier for saturation.
        self.brightness_std   = float(brightness_std)   # Standard deviation of brightness.
        self.contrast_std     = float(contrast_std)     # Log2 standard deviation of contrast.
        self.hue_max          = float(hue_max)          # Range of hue rotation, 1 = full circle.
        self.saturation_std   = float(saturation_std)   # Log2 standard deviation of saturation.

        # Image-space filtering.
        self.imgfilter        = float(imgfilter)        # Probability multiplier for image-space filtering.
        self.imgfilter_bands  = list(imgfilter_bands)   # Probability multipliers for individual frequency bands.
        self.imgfilter_std    = float(imgfilter_std)    # Log2 standard deviation of image-space filter amplification.

        # Image-space corruptions.
        self.noise            = float(noise)            # Probability multiplier for additive RGB noise.
        self.cutout           = float(cutout)           # Probability multiplier for cutout.
        self.noise_std        = float(noise_std)        # Standard deviation of additive RGB noise.
        self.cutout_size      = float(cutout_size)      # Size of the cutout rectangle, relative to image dimensions.

        # Setup orthogonal lowpass filter for geometric augmentations.
        self.register_buffer('Hz_geom', upfirdn2d_setup_filter(wavelets['sym6']))

        # Construct filter bank for image-space filtering.
        Hz_lo = np.asarray(wavelets['sym2'])            # H(z)
        Hz_hi = Hz_lo * ((-1) ** np.arange(Hz_lo.size)) # H(-z)
        Hz_lo2 = np.convolve(Hz_lo, Hz_lo[::-1]) / 2    # H(z) * H(z^-1) / 2
        Hz_hi2 = np.convolve(Hz_hi, Hz_hi[::-1]) / 2    # H(-z) * H(-z^-1) / 2
        Hz_fbank = np.eye(4, 1)                         # Bandpass(H(z), b_i)
        for i in range(1, Hz_fbank.shape[0]):
            Hz_fbank = np.dstack([Hz_fbank, np.zeros_like(Hz_fbank)]).reshape(Hz_fbank.shape[0], -1)[:, :-1]
            Hz_fbank = scipy.signal.convolve(Hz_fbank, [Hz_lo2])
            Hz_fbank[i, (Hz_fbank.shape[1] - Hz_hi2.size) // 2 : (Hz_fbank.shape[1] + Hz_hi2.size) // 2] += Hz_hi2
        self.register_buffer('Hz_fbank', paddle.to_tensor(Hz_fbank, dtype=paddle.float32))

    def forward(self, images, debug_percentile=None):
        assert images.ndim == 4
        batch_size, num_channels, height, width = images.shape
        if debug_percentile is not None:
            debug_percentile = paddle.to_tensor(debug_percentile, paddle=paddle.float32)

        # -------------------------------------
        # Select parameters for pixel blitting.
        # -------------------------------------

        # Initialize inverse homogeneous 2D transform: G_inv @ pixel_out ==> pixel_in
        I_3 = paddle.eye(3)
        G_inv = I_3

        # Apply x-flip with probability (xflip * strength).
        if self.xflip > 0:
            i = paddle.floor(paddle.rand([batch_size], dtype=paddle.float32) * 2)
            i = paddle.where(paddle.rand([batch_size], dtype=paddle.float32) < self.xflip * self.p, i, paddle.zeros_like(i))
            if debug_percentile is not None:
                i = paddle.full_like(i, paddle.floor(debug_percentile * 2))
            G_inv = G_inv @ scale2d_inv(1 - 2 * i, 1)

        # Apply 90 degree rotations with probability (rotate90 * strength).
        if self.rotate90 > 0:
            i = paddle.floor(paddle.rand([batch_size], dtype=paddle.float32) * 4)
            i = paddle.where(paddle.rand([batch_size], dtype=paddle.float32) < self.rotate90 * self.p, i, paddle.zeros_like(i))
            if debug_percentile is not None:
                i = paddle.full_like(i, paddle.floor(debug_percentile * 4))
            G_inv = G_inv @ rotate2d_inv(-np.pi / 2 * i)

        # Apply integer translation with probability (xint * strength).
        if self.xint > 0:
            t = (paddle.rand([batch_size, 2], dtype=paddle.float32) * 2 - 1) * self.xint_max
            t = paddle.where(paddle.rand([batch_size, 1], dtype=paddle.float32) < self.xint * self.p, t, paddle.zeros_like(t))
            if debug_percentile is not None:
                t = paddle.full_like(t, (debug_percentile * 2 - 1) * self.xint_max)
            G_inv = G_inv @ translate2d_inv(paddle.round(t[:,0] * width), paddle.round(t[:,1] * height))

        # --------------------------------------------------------
        # Select parameters for general geometric transformations.
        # --------------------------------------------------------

        # Apply isotropic scaling with probability (scale * strength).
        if self.scale > 0:
            zhishu = paddle.randn([batch_size], dtype=paddle.float32) * self.scale_std
            s = paddle.pow(paddle.zeros_like(zhishu, dtype=zhishu.dtype) + 2.0, zhishu)
            s = paddle.where(paddle.rand([batch_size], dtype=paddle.float32) < self.scale * self.p, s, paddle.ones_like(s))
            if debug_percentile is not None:
                zhishu = torch.erfinv(debug_percentile * 2 - 1) * self.scale_std
                temp = paddle.pow(paddle.zeros_like(zhishu, dtype=zhishu.dtype) + 2.0, zhishu)
                s = paddle.full_like(s, temp)
            G_inv = G_inv @ scale2d_inv(s, s)

        # Apply pre-rotation with probability p_rot.
        p_rot = (1 - self.rotate * self.p)
        p_rot = paddle.clip(p_rot, 0, 1)
        p_rot = 1 - paddle.sqrt(p_rot) # P(pre OR post) = p
        if self.rotate > 0:
            theta = (paddle.rand([batch_size], dtype=paddle.float32) * 2 - 1) * np.pi * self.rotate_max
            theta = paddle.where(paddle.rand([batch_size], dtype=paddle.float32) < p_rot, theta, paddle.zeros_like(theta))
            if debug_percentile is not None:
                theta = paddle.full_like(theta, (debug_percentile * 2 - 1) * np.pi * self.rotate_max)
            G_inv = G_inv @ rotate2d_inv(-theta) # Before anisotropic scaling.

        # Apply anisotropic scaling with probability (aniso * strength).
        if self.aniso > 0:
            zhishu = paddle.randn([batch_size], dtype=paddle.float32) * self.aniso_std
            s = paddle.pow(paddle.zeros_like(zhishu, dtype=zhishu.dtype) + 2.0, zhishu)
            s = paddle.where(paddle.rand([batch_size], dtype=paddle.float32) < self.aniso * self.p, s, paddle.ones_like(s))
            if debug_percentile is not None:
                s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.aniso_std))
            G_inv = G_inv @ scale2d_inv(s, 1 / s)

        # Apply post-rotation with probability p_rot.
        if self.rotate > 0:
            theta = (paddle.rand([batch_size], dtype=paddle.float32) * 2 - 1) * np.pi * self.rotate_max
            theta = paddle.where(paddle.rand([batch_size], dtype=paddle.float32) < p_rot, theta, paddle.zeros_like(theta))
            if debug_percentile is not None:
                theta = paddle.zeros_like(theta)
            G_inv = G_inv @ rotate2d_inv(-theta) # After anisotropic scaling.

        # Apply fractional translation with probability (xfrac * strength).
        if self.xfrac > 0:
            t = paddle.randn([batch_size, 2], dtype=paddle.float32) * self.xfrac_std
            t = paddle.where(paddle.rand([batch_size, 1], dtype=paddle.float32) < self.xfrac * self.p, t, paddle.zeros_like(t))
            if debug_percentile is not None:
                t = torch.full_like(t, torch.erfinv(debug_percentile * 2 - 1) * self.xfrac_std)
            G_inv = G_inv @ translate2d_inv(t[:,0] * width, t[:,1] * height)

        # ----------------------------------
        # Execute geometric transformations.
        # ----------------------------------

        # Execute if the transform is not identity.
        if G_inv is not I_3:

            # Calculate padding.
            cx = (width - 1) / 2
            cy = (height - 1) / 2
            cp = matrix([-cx, -cy, 1], [cx, -cy, 1], [cx, cy, 1], [-cx, cy, 1]) # [idx, xyz]
            cp = G_inv @ cp.t() # [batch, xyz, idx]
            Hz_pad = self.Hz_geom.shape[0] // 4
            # margin = cp[:, :2, :].permute(1, 0, 2).flatten(1) # [xy, batch * idx]
            margin = cp[:, :2, :]
            margin = paddle.transpose(margin, perm=[1, 0, 2])
            margin = paddle.flatten(margin, 1)
            margin = paddle.concat([-margin, margin])
            margin = margin.max(1) # [x0, y0, x1, y1]
            margin = margin + constant([Hz_pad * 2 - cx, Hz_pad * 2 - cy] * 2)
            margin = paddle.maximum(margin, constant([0, 0] * 2))
            margin = paddle.minimum(margin, constant([width-1, height-1] * 2))
            margin = margin.ceil()  # 向上取整
            margin = paddle.cast(margin, dtype=paddle.int32)
            mx0, my0, mx1, my1 = margin

            # Pad image and adjust origin.
            images = paddle.nn.functional.pad(images, pad=[mx0, mx1, my0, my1], mode='reflect')
            G_inv = translate2d((mx0 - mx1) / 2, (my0 - my1) / 2) @ G_inv

            # Upsample.
            images = upsample2d(x=images, f=self.Hz_geom, up=2)
            G_inv = scale2d(2, 2) @ G_inv @ scale2d_inv(2, 2)
            G_inv = translate2d(-0.5, -0.5) @ G_inv @ translate2d_inv(-0.5, -0.5)

            # Execute transformation.
            shape = [batch_size, num_channels, (height + Hz_pad * 2) * 2, (width + Hz_pad * 2) * 2]
            G_inv = scale2d(2 / images.shape[3], 2 / images.shape[2]) @ G_inv @ scale2d_inv(2 / shape[3], 2 / shape[2])

            # dic = {}
            # dic['G_inv'] = G_inv.numpy()
            # dic['images'] = images.numpy()
            # dic['shape'] = np.array(shape)
            # np.savez('affine_grid_data', **dic)

            grid = paddle.nn.functional.affine_grid(theta=G_inv[:, :2, :], out_shape=shape, align_corners=False)
            # grid_sample没有实现二阶梯度，需要用另外的等价实现。
            images = paddle.nn.functional.grid_sample(images, grid=grid, mode='bilinear', padding_mode='zeros', align_corners=False)

            # Downsample and crop.
            images = downsample2d(x=images, f=self.Hz_geom, down=2, padding=-Hz_pad*2, flip_filter=True)

        # --------------------------------------------
        # Select parameters for color transformations.
        # --------------------------------------------

        # Initialize homogeneous 3D transformation matrix: C @ color_in ==> color_out
        I_4 = paddle.eye(4)
        C = I_4

        # Apply brightness with probability (brightness * strength).
        if self.brightness > 0:
            b = paddle.randn([batch_size], dtype=paddle.float32) * self.brightness_std
            b = paddle.where(paddle.rand([batch_size], dtype=paddle.float32) < self.brightness * self.p, b, paddle.zeros_like(b))
            if debug_percentile is not None:
                b = paddle.full_like(b, paddle.erfinv(debug_percentile * 2 - 1) * self.brightness_std)
            C = translate3d(b, b, b) @ C

        # Apply contrast with probability (contrast * strength).
        if self.contrast > 0:
            zhishu = paddle.randn([batch_size], dtype=paddle.float32) * self.contrast_std
            c = paddle.pow(paddle.zeros_like(zhishu, dtype=zhishu.dtype) + 2.0, zhishu)
            c = paddle.where(paddle.rand([batch_size], dtype=paddle.float32) < self.contrast * self.p, c, paddle.ones_like(c))
            if debug_percentile is not None:
                c = torch.full_like(c, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.contrast_std))
            C = scale3d(c, c, c) @ C

        # Apply luma flip with probability (lumaflip * strength).
        v = constant(np.asarray([1, 1, 1, 0]) / np.sqrt(3)) # Luma axis.
        if self.lumaflip > 0:
            i = paddle.floor(paddle.rand([batch_size, 1, 1], dtype=paddle.float32) * 2)
            i = paddle.where(paddle.rand([batch_size, 1, 1], dtype=paddle.float32) < self.lumaflip * self.p, i, paddle.zeros_like(i))
            if debug_percentile is not None:
                i = paddle.full_like(i, paddle.floor(debug_percentile * 2))
            v2 = paddle.unsqueeze(v, 1)  # [n, 1]
            C = (I_4 - 2 * paddle.matmul(v2, v2.transpose((1, 0))) * i) @ C # Householder reflection.

        # Apply hue rotation with probability (hue * strength).
        if self.hue > 0 and num_channels > 1:
            theta = (paddle.rand([batch_size], dtype=paddle.float32) * 2 - 1) * np.pi * self.hue_max
            theta = paddle.where(paddle.rand([batch_size], dtype=paddle.float32) < self.hue * self.p, theta, paddle.zeros_like(theta))
            if debug_percentile is not None:
                theta = paddle.full_like(theta, (debug_percentile * 2 - 1) * np.pi * self.hue_max)
            C = rotate3d(v, theta) @ C # Rotate around v.

        # Apply saturation with probability (saturation * strength).
        if self.saturation > 0 and num_channels > 1:
            zhishu = paddle.randn([batch_size, 1, 1], dtype=paddle.float32) * self.saturation_std
            s = paddle.pow(paddle.zeros_like(zhishu, dtype=zhishu.dtype) + 2.0, zhishu)
            s = paddle.where(paddle.rand([batch_size, 1, 1], dtype=paddle.float32) < self.saturation * self.p, s, paddle.ones_like(s))
            if debug_percentile is not None:
                s = torch.full_like(s, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.saturation_std))
            v2 = paddle.unsqueeze(v, 1)  # [n, 1]
            C = (paddle.matmul(v2, v2.transpose((1, 0))) + (I_4 - paddle.matmul(v2, v2.transpose((1, 0)))) * s) @ C

        # ------------------------------
        # Execute color transformations.
        # ------------------------------

        # Execute if the transform is not identity.
        if C is not I_4:
            images = images.reshape([batch_size, num_channels, height * width])
            if num_channels == 3:
                images = C[:, :3, :3] @ images + C[:, :3, 3:]
            elif num_channels == 1:
                C = C[:, :3, :].mean(dim=1, keepdims=True)
                images = images * C[:, :, :3].sum(dim=2, keepdims=True) + C[:, :, 3:]
            else:
                raise ValueError('Image must be RGB (3 channels) or L (1 channel)')
            images = images.reshape([batch_size, num_channels, height, width])

        # ----------------------
        # Image-space filtering.
        # ----------------------

        if self.imgfilter > 0:
            num_bands = self.Hz_fbank.shape[0]
            assert len(self.imgfilter_bands) == num_bands
            expected_power = constant(np.array([10, 1, 1, 1]) / 13) # Expected power spectrum (1/f).

            # Apply amplification for each band with probability (imgfilter * strength * band_strength).
            g = paddle.ones([batch_size, num_bands], dtype=paddle.float32) # Global gain vector (identity).
            for i, band_strength in enumerate(self.imgfilter_bands):
                zhishu = paddle.randn([batch_size], dtype=paddle.float32) * self.imgfilter_std
                t_i = paddle.pow(paddle.zeros_like(zhishu, dtype=zhishu.dtype) + 2.0, zhishu)
                t_i = paddle.where(paddle.rand([batch_size], dtype=paddle.float32) < self.imgfilter * self.p * band_strength, t_i, paddle.ones_like(t_i))
                if debug_percentile is not None:
                    t_i = torch.full_like(t_i, torch.exp2(torch.erfinv(debug_percentile * 2 - 1) * self.imgfilter_std)) if band_strength > 0 else paddle.ones_like(t_i)
                t = paddle.ones([batch_size, num_bands], dtype=paddle.float32)          # Temporary gain vector.
                t[:, i] = t_i                                                           # Replace i'th element.
                t = t / (expected_power * t.square()).sum(dim=-1, keepdims=True).sqrt() # Normalize power.
                g = g * t                                                               # Accumulate into global gain.

            # Construct combined amplification filter.
            Hz_prime = g @ self.Hz_fbank                                    # [batch, tap]
            Hz_prime = Hz_prime.unsqueeze(1).repeat([1, num_channels, 1])   # [batch, channels, tap]
            Hz_prime = Hz_prime.reshape([batch_size * num_channels, 1, -1]) # [batch * channels, 1, tap]

            # Apply filter.
            p = self.Hz_fbank.shape[1] // 2
            images = images.reshape([1, batch_size * num_channels, height, width])
            images = paddle.nn.functional.pad(images, pad=[p, p, p, p], mode='reflect')
            images = conv2d_gradfix.conv2d(input=images, weight=Hz_prime.unsqueeze(2), groups=batch_size*num_channels)
            images = conv2d_gradfix.conv2d(input=images, weight=Hz_prime.unsqueeze(3), groups=batch_size*num_channels)
            images = images.reshape([batch_size, num_channels, height, width])

        # ------------------------
        # Image-space corruptions.
        # ------------------------

        # Apply additive RGB noise with probability (noise * strength).
        if self.noise > 0:
            sigma = paddle.randn([batch_size, 1, 1, 1], dtype=paddle.float32).abs() * self.noise_std
            sigma = paddle.where(paddle.rand([batch_size, 1, 1, 1], dtype=paddle.float32) < self.noise * self.p, sigma, paddle.zeros_like(sigma))
            if debug_percentile is not None:
                sigma = paddle.full_like(sigma, paddle.erfinv(debug_percentile) * self.noise_std)
            images = images + paddle.randn([batch_size, num_channels, height, width], dtype=paddle.float32) * sigma

        # Apply cutout with probability (cutout * strength).
        if self.cutout > 0:
            size = paddle.full([batch_size, 2, 1, 1, 1], self.cutout_size, dtype=paddle.float32)
            size = paddle.where(paddle.rand([batch_size, 1, 1, 1, 1], dtype=paddle.float32) < self.cutout * self.p, size, paddle.zeros_like(size))
            center = paddle.rand([batch_size, 2, 1, 1, 1], dtype=paddle.float32)
            if debug_percentile is not None:
                size = paddle.full_like(size, self.cutout_size)
                center = paddle.full_like(center, debug_percentile)
            coord_x = paddle.arange(width, device=device).reshape([1, 1, 1, -1])
            coord_y = paddle.arange(height, device=device).reshape([1, 1, -1, 1])
            mask_x = (((coord_x + 0.5) / width - center[:, 0]).abs() >= size[:, 0] / 2)
            mask_y = (((coord_y + 0.5) / height - center[:, 1]).abs() >= size[:, 1] / 2)
            mask = paddle.logical_or(mask_x, mask_y).to(paddle.float32)
            images = images * mask

        return images





