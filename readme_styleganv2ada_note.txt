算法笔记

1.StyleGANv2ADA_SynthesisNetwork
从分辨率4逐步生成最终分辨率的图片，每个分辨率的通道数是 min(channel_base // 分辨率, channel_max), 分辨率越大通道数越小。通道数不能超过设定的channel_max。
在最高的num_fp16_res个分辨率使用了fp16，减小显存压力。
每个分辨率使用一个SynthesisBlock。

2.SynthesisBlock
每个SynthesisBlock都会登记一个resample_filter变量（buffer，不可被优化器更新，类似BN层的均值、方差），
resample_filter变量是给upsample2d()即upfirdn2d()方法用的，resample_filter是1个形状为[4, 4]的张量，值为
[[1/64, 3/64, 3/64, 1/64],
 [3/64, 9/64, 9/64, 3/64],
 [3/64, 9/64, 9/64, 3/64],
 [1/64, 3/64, 3/64, 1/64]]
resample_filter即“重采样过滤器”

第0个SynthesisBlock（分辨率是4）会持有1个可训练参数const，形状是[out_channels, 4, 4]，stylegan所有的图片都是用这个const生成的（最初的来源）！！！
非第0个SynthesisBlock会持有1个SynthesisLayer，名为conv0；
所有的SynthesisBlock会持有1个SynthesisLayer，名为conv1；
所有的SynthesisBlock会持有1个ToRGBLayer，名为torgb；

前向传播：
def forward(self, x, img, ws, force_fp32=False, fused_modconv=None, **layer_kwargs):
第0个SynthesisBlock（分辨率是4）的x = img = None
所以，它的x是const重复{批大小}次得到的。
        if self.in_channels == 0:
            x = self.const.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
然后x和对应的ws[:, i, :]经过conv0(第0个block没有，不需要，conv0会让x的分辨率*2)、conv1、torgb，1个SynthesisLayer用掉1个ws[:, i, :]，
上一层的torgb的和下一层的conv0共用同1个ws[:, i, :]（torgb()调用之后下标i不+1）

另外：
        # ToRGB.
        if img is not None:
            img = upsample2d(img, self.resample_filter)
        if self.is_last or self.architecture == 'skip':
            y = self.torgb(x, next(w_iter), fused_modconv=fused_modconv)
            y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
            img = img.add_(y) if img is not None else y
即下一层的输入x其实是上一层conv1的输出（不是上一层torgb的输出）；最终生成的图片img其实是每一层（每一个SynthesisBlock）的torgb的输出y的累加的结果，
y的通道数都是3，RGB嘛！
累加之前，先对前n层的累加结果img进行上采样upsample2d()，传入的额外参数是self.resample_filter

upsample2d()让img的分辨率*2，里面调用了upfirdn2d()，
具体做法是对img间隔地填充0，然后用self.resample_filter卷积它
upfirdn2d内部,resample_filter先变成
f = f * (gain ** (f.ndim / 2))
即f = f * (4 ** (2 / 2))    f = f * 4
[[1/16, 3/16, 3/16, 1/16],
 [3/16, 9/16, 9/16, 3/16],
 [3/16, 9/16, 9/16, 3/16],
 [1/16, 3/16, 3/16, 1/16]]
然后f形状扩展成[1, 1, 4, 4]，再重复img_channel=3次，变成[3, 1, 4, 4]
用f分组卷积x.shape=[N, 3, 11, 11]得到x.shape=[N, 3, 8, 8]从而完成上采样。

试想一下，最初x的形状是[N, 3, 4, 4]，pad之后变成[N, 3, 8, 8]，再pad之后变成[N, 3, 11, 11]
[N, 3, 8, 8]时值为
[[x00, 0, x01, 0, x02, 0, x03, 0],
 [  0, 0,   0, 0,   0, 0,   0, 0],
 [x10, 0, x11, 0, x12, 0, x13, 0],
 [  0, 0,   0, 0,   0, 0,   0, 0],
 [x20, 0, x21, 0, x22, 0, x23, 0],
 [  0, 0,   0, 0,   0, 0,   0, 0],
 [x30, 0, x31, 0, x32, 0, x33, 0],
 [  0, 0,   0, 0,   0, 0,   0, 0]]
卷积后得到
y00 = (x00 * 1 + x01 * 3 + x10 * 3 + x11 * 9)/16
即pad之后的x的4x4格子组合成新图片的1个像素，格子中间的元素权重大，边缘的元素权重小，这就是为什么resample_filter叫做“重采样过滤器”，所以它是固定的，不应该被优化器更新；
即upfirdn2d()不是简单地填充0的上采样，填充0之后，用“重采样过滤器”进行4x4且stride==1的卷积，4x4方格中间的元素权重大，边缘的元素权重小，组成最后的上采样的图片。

最终生成的img是-1到1之间的值，交给判别器去判断。真实图片也会通过
phase_real_img = phase_real_img.to(device).to(torch.float32) / 127.5 - 1
转成-1到1之间的值，交给判别器去判断。总之，判别器接收的输入为-1到1之间的值。

假图片不止只由const和网络权重生成，还依赖潜在因子ws（StyleGANv2ADA_MappingNetwork的输出）一起生成。
StyleGANv2ADA_MappingNetwork里的ws会经过以下代码：
        if self.num_ws is not None:
            x = x.unsqueeze(1).repeat([1, self.num_ws, 1])
变成1个三维张量[N, num_ws, 512]，即每一个SynthesisLayer和ToRGBLayer用掉的ws[:, i, :]在数值上是一样的。ws即潜在因子，
在每一种分辨率上使用数值上相等的潜在因子生成假图片。


3.SynthesisLayer 详解
前向传播：
def forward(self, x, w, noise_mode='random', fused_modconv=True, gain=1):
    styles = self.affine(w)
潜在因子w经过1个全连接层变成风格向量styles

之后进入modulated_conv2d()
未完待续...


4.StyleGANv2ADA_MappingNetwork
StyleGANv2ADA_MappingNetwork的输入是随机噪声z，形状是[N, z_dim]
    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        if self.z_dim > 0:
            x = normalize_2nd_moment(z.to(torch.float32))
        if self.c_dim > 0:
            y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
            x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            assert self.w_avg_beta is not None
            if self.num_ws is None or truncation_cutoff is None:
                x = self.w_avg.lerp(x, truncation_psi)
            else:
                x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
首先，z先通过normalize_2nd_moment()进行归一化，得到x，之后，经过num_layers个全连接层，激活函数默认是leaky_relu()；
之后，如果是训练状态，以EMA的方式更新self.w_avg，self.w_avg是潜在因子的平均值！！！，形状是[w_dim, ]，
self.w_avg_beta默认是0.995， 即更新时旧的self.w_avg占的比重是0.995，x.detach().mean(dim=0)占的比重是0.005。
最后，潜在因子重复num_ws次。

truncation即截断，这段代码一般预测时用。预测时，如果truncation_cutoff是None，
    x = self.w_avg.lerp(x, truncation_psi)
即self.w_avg和生成的潜在因子x插值组成最后的潜在因子，x占的比重是truncation_psi；
如果truncation_cutoff不是None，
    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
即只有前truncation_cutoff个潜在因子插值。


在训练时的run_G()方法中，如果
            if self.style_mixing_prob > 0:
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                # ws[:, cutoff:] = self.mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
                temp = self.mapping(torch.randn_like(z), c, skip_w_avg_update=True)[:, cutoff:]
                temp2 = ws[:, :cutoff]
                ws = torch.cat([temp2, temp], 1)
会用另外的噪声torch.randn_like(z)生成另外的潜在因子temp，ws的后cutoff个潜在因子被替换成了temp，以此训练模型的style_mixing能力。
怎么理解呢？即生成假图片时，低分辨率用的是z生成的潜在因子，高分辨率用的是torch.randn_like(z)生成的潜在因子，由此，生成的假图片
有了z的动作姿态，却有了torch.randn_like(z)提供的皮肤。

python tools/demo.py style_mixing -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c styleganv2ada_512_afhqcat.pth --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu

在这条命令中，col_styles 0,1,2,3,4,5,6表示的是row_seeds生成的潜在因子的0,1,2,3,4,5,6个被替换成了col_seeds的0,1,2,3,4,5,6个，
即col_seeds提供了动作姿态，row_seeds提供了皮肤。
