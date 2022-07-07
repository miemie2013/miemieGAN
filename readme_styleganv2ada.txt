# 如果命令不能成功执行，说明咩酱实现中，，，
nvidia-smi


pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


梯度对齐：
1.(原版仓库也要设置)设置 StyleGANv2ADAModel 的
    self.style_mixing_prob = -1.0
    self.align_grad = True
解除上面语句的注释即可。
以及，对下面所有的以if self.align_grad:开头的代码块解除注释
if self.align_grad:
    xxx

计算loss_Gpl那里，
pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
改为
pl_noise = torch.ones_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])

run_D()方法中，注释掉：
            img = self.augment_pipe(img)
解除注释：
            # debug_percentile = 0.7
            # img = self.augment_pipe(img, debug_percentile)


2.trainer.py下面代码解除注释
        # 对齐梯度用
        # if self.rank == 0:
        #     if (self.iter + 1) == 20:
        #         self.save_ckpt(ckpt_name="%d" % (self.epoch + 1))

3.(原版仓库也要设置)设置 SynthesisLayer 的
    self.use_noise = False
4.(原版仓库也要设置)设置 StyleGANv2ADA_SynthesisNetwork 的
    use_fp16 = False
5.(原版仓库也要设置)设置 StyleGANv2ADA_Discriminator 的
    use_fp16 = False

6. styleganv2ada_method_base.py 优化器要换成SGD：
                optimizer = torch.optim.SGD(
                    itertools.chain(self.model.synthesis.parameters(), self.model.mapping.parameters()), lr=0.001, momentum=0.9
                )
                ...
                optimizer = torch.optim.SGD(
                    self.model.discriminator.parameters(), lr=0.002, momentum=0.9
                )
因为Adam更新参数有一定随机性，同样的情况下，跑2次结果不同！！！（但是SGD也有轻微的不同，影响不大。）


第三方实现stylegan2-ada时，不要忘记创建G和D的实例时，都需要设置其的requires_grad_(False)
    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
    D = dnnlib.util.construct_class_by_name(**D_kwargs, **common_kwargs).train().requires_grad_(False).to(device) # subclass of torch.nn.Module
因为第0步训练Gmain阶段时，D的权重应该不允许得到梯度。
而且，除了augment_pipe，其它4个 G.mapping、G.synthesis、D、G_ema 都是DDP模型。



python tools/convert_weights.py -f exps/styleganv2ada/styleganv2ada_32_custom.py -c_G G_00.pth -c_Gema G_ema_00.pth -c_D D_00.pth -oc styleganv2ada_32_00.pth


python tools/convert_weights.py -f exps/styleganv2ada/styleganv2ada_32_custom.py -c_G G_19.pth -c_Gema G_ema_19.pth -c_D D_19.pth -oc styleganv2ada_32_19.pth


CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/styleganv2ada/styleganv2ada_32_custom.py -d 1 -b 8 -eb 1 -c styleganv2ada_32_00.pth


CUDA_VISIBLE_DEVICES=0,1
python tools/train.py -f exps/styleganv2ada/styleganv2ada_32_custom.py -d 2 -b 8 -eb 2 -c styleganv2ada_32_00.pth


CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/styleganv2ada/styleganv2ada_32_custom.py --dist-url tcp://192.168.0.104:12319 --num_machines 2 --machine_rank 1 -b 8 -eb 2 -c styleganv2ada_32_00.pth



python diff_weights.py --cp1 styleganv2ada_32_19.pth --cp2 StyleGANv2ADA_outputs/styleganv2ada_32_custom/1.pth --d_value 0.0005



----------------------- 进阶：单卡和多卡进行对齐（用 单卡批大小8 实现 双卡总批大小8 的效果） -----------------------
把 DiscriminatorEpilogue 类的__init__()方法的
        # mbstd_num_channels = 0
解除注释，即不使用 MinibatchStdLayer ，因为 MinibatchStdLayer 会有类似BN层的不同图片的数据交流，
使用的话要实现类似同步BN的操作，会加大对齐难度，所以不使用这个层。

除了上述“梯度对齐”的修改外，还需要修改：
1.设置 StyleGANv2ADAModel 的
对下面所有的以if self.align_2gpu_1gpu:开头的代码块解除注释
if self.align_2gpu_1gpu:
    xxx

注释掉 accumulate_gradients() 里的：
gen_img, gen_ws = self.run_G(gen_z[:batch_size], gen_c_, sync=sync)

2.(原版仓库也要设置)把 DiscriminatorEpilogue 类的__init__()方法的
        # mbstd_num_channels = 0
解除注释

3.把 StyleGANv2ADA_MappingNetwork 类的forward()方法的
            self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))
注释掉。把
            # bz = x.shape[0]
            # gpu0_w_avg = x.detach()[:bz//2].mean(dim=0).lerp(self.w_avg, self.w_avg_beta)
            # gpu1_w_avg = x.detach()[bz//2:].mean(dim=0).lerp(self.w_avg, self.w_avg_beta)
            # self.w_avg.copy_(gpu0_w_avg)
解除注释。即： 单机多卡训练时，w_avg的更新情况：强制使用 0号gpu 更新后的w_avg 作为整体更新后的w_avg


输入以下命令验证是否已经对齐：
python tools/convert_weights.py -f exps/styleganv2ada/styleganv2ada_32_custom.py -c_G G_00.pth -c_Gema G_ema_00.pth -c_D D_00.pth -oc styleganv2ada_32_00.pth

python tools/convert_weights.py -f exps/styleganv2ada/styleganv2ada_32_custom.py -c_G G_19.pth -c_Gema G_ema_19.pth -c_D D_19.pth -oc styleganv2ada_32_19.pth

CUDA_VISIBLE_DEVICES=0
python tools/train.py -f exps/styleganv2ada/styleganv2ada_32_custom.py -d 1 -b 8 -eb 1 -c styleganv2ada_32_00.pth

python diff_weights.py --cp1 styleganv2ada_32_19.pth --cp2 StyleGANv2ADA_outputs/styleganv2ada_32_custom/1.pth --d_value 0.0005




----------------------- 进阶：验证每张卡上的训练数据是否不重复 -----------------------
1.exps/styleganv2ada/styleganv2ada_256_custom.py
        # 有前16张照片，用来验证每张卡上的训练数据是否不重复
        # self.dataroot = '../data/data110820/faces2'
解除注释。
        self.print_interval = 10
改成1
        self.save_step_interval = 1000
改成2

2.trainer.py下面代码解除注释
            # if self.rank == 0:
            #     logger.info(raw_idx)
            # else:
            #     print(raw_idx)
肉眼观察raw_idx即可(0~15的值)。

单机1卡
python tools/train.py -f exps/styleganv2ada/styleganv2ada_256_custom.py -d 1 -b 4 -eb 1 -c styleganv2ada_512_afhqcat.pth

python tools/train.py -f exps/styleganv2ada/styleganv2ada_256_custom.py -d 1 -b 4 -eb 1 -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/0_2.pth --resume


单机2卡
CUDA_VISIBLE_DEVICES=0,1
python tools/train.py -f exps/styleganv2ada/styleganv2ada_256_custom.py -d 2 -b 4 -eb 2 -c styleganv2ada_512_afhqcat.pth

CUDA_VISIBLE_DEVICES=0,1
python tools/train.py -f exps/styleganv2ada/styleganv2ada_256_custom.py -d 2 -b 4 -eb 2 -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/0_2.pth --resume




----------------------- 转换权重 -----------------------
python tools/convert_weights.py -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c_Gema G_ema_afhqcat.pth -c_G G_afhqcat.pth -c_D D_afhqcat.pth -oc styleganv2ada_512_afhqcat.pth


python tools/convert_weights.py -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c_Gema G_ema_metfaces.pth -c_G G_metfaces.pth -c_D D_metfaces.pth -oc styleganv2ada_1024_metfaces.pth


python tools/convert_weights.py -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c_Gema G_ema_ffhq.pth -c_G G_ffhq.pth -c_D D_ffhq.pth -oc styleganv2ada_1024_ffhq.pth




----------------------- 预测 -----------------------

python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_128_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_128_custom/48.pth --seeds 85,100,75,458,1500 --save_result --device gpu


python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --seeds 85,100,75,458,1500 --save_result --device gpu


python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --seeds 85,100,75,458,1500 --noise_mode random --trunc 0.3 --save_result --device gpu

(trunc==1.0时，StyleGANv2ADA_MappingNetwork的w_avg不参与图像生成；
trunc==0.9时，StyleGANv2ADA_MappingNetwork的w_avg参与图像生成，噪声生成的ws占的比重是0.9，w_avg占的比重是0.1；
trunc==0.0时，StyleGANv2ADA_MappingNetwork的w_avg参与图像生成，噪声生成的ws占的比重是0.0，w_avg占的比重是1.0，即噪声生成的ws不参与图像生成，只用统计的w_avg（ws的平均值，即w_avg妈妈！！！）来生成图像；
其实就是python tools/demo.py A2B那样的插值生成渐变图像，只不过其中1个ws是w_avg
)
(这句代码召唤w_avg妈妈！！！你知道w_avg妈妈有多好看吗！！！她是整个噪声空间经过MappingNetwork之后得到的ws的平均值，倾国倾城都不能形容她的美貌！)
python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --seeds 85,100,75,458,1500 --noise_mode const --trunc 0.0 --save_result --device gpu

w_avg妈妈！！！

python tools/demo.py image -f exps/styleganv3/styleganv3_s_256_custom.py -c StyleGANv3_outputs/styleganv3_s_256_custom/77.pth --seeds 85,100,75,458,1500 --noise_mode const --trunc 0.0 --save_result --device gpu

python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_256_uppercloth.py -c StyleGANv2ADA_outputs/styleganv2ada_256_uppercloth/123.pth --seeds 85,100,75,458,1500 --noise_mode const --trunc 0.0 --save_result --device gpu


python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --seeds 0_1500 --save_result --device gpu

(styleganv3)
python tools/demo.py image -f exps/styleganv3/styleganv3_s_256_custom.py -c StyleGANv3_outputs/styleganv3_s_256_custom/77.pth --seeds 0_1500 --save_result --device gpu

(styleganv2ada, uppercloth)
python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_256_uppercloth.py -c StyleGANv2ADA_outputs/styleganv2ada_256_uppercloth/123.pth --seeds 0_1500 --save_result --device gpu


(afhq)
python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c styleganv2ada_512_afhqcat.pth --seeds 85,100,75,458,1500 --save_result --device gpu

python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c styleganv2ada_512_afhqcat.pth --seeds 0_1500 --save_result --device gpu

w_avg妈妈：
python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c styleganv2ada_512_afhqcat.pth --seeds 85,100,75,458,1500 --noise_mode const --trunc 0.0 --save_result --device gpu



(afhqv2)
python tools/demo.py image -f exps/styleganv3/styleganv3_r_512_afhqv2.py -c stylegan3_r_afhqv2_512.pth --seeds 85,100,75,458,1500 --save_result --device gpu



(metfaces)
python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c styleganv2ada_1024_metfaces.pth --seeds 85,100,75,458,1500 --save_result --device gpu


python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c styleganv2ada_1024_ffhq.pth --seeds 85,100,75,458,1500 --save_result --device gpu


w_avg妈妈：
python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c styleganv2ada_1024_ffhq.pth --seeds 85,100,75,458,1500 --noise_mode const --trunc 0.0 --save_result --device gpu


----------------------- 渐变，从随机种子A的图像渐变成随机种子B的图像 -----------------------
--frames 表示渐变的帧数。

(动漫头像数据集)
python tools/demo.py A2B -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --seeds 85,100 --frames 120 --video_fps 30 --save_result --device gpu

python tools/demo.py A2B -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --seeds 85,100,75,458,1500 --frames 120 --video_fps 30 --save_result --device gpu

(--A2B_mixing_seed 85 表示这个随机种子生成的ws在每一帧和每一帧的ws进行style_mixing，
 --col_styles 0,1,2,3,4,5,6表示A2B_mixing_seed生成的ws提供了动作姿态，每一帧的ws提供了皮肤；
 --col_styles 7,8,9,10,11,12,13表示A2B_mixing_seed生成的ws提供了皮肤，每一帧的ws提供了动作姿态；

 --A2B_mixing_seed w_avg 表示直接用w_avg妈妈进行style_mixing。
)
python tools/demo.py A2B -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --seeds 85,100,75,458,1500 --A2B_mixing_seed 458 --col_styles 0,1,2,3,4,5,6 --frames 120 --video_fps 30 --save_result --device gpu

python tools/demo.py A2B -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --seeds 85,100,75,458,1500 --A2B_mixing_seed 458 --col_styles 7,8,9,10,11,12,13 --frames 120 --video_fps 30 --save_result --device gpu

python tools/demo.py A2B -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --seeds 85,100,75,458,1500 --A2B_mixing_seed 458 --col_styles 2,3,4,5,6,7,8,9,10,11,12,13 --frames 120 --video_fps 30 --save_result --device gpu

(w_avg妈妈提供动作姿态！！！神仙颜值，什么发色瞳色都能驾驭！！！)
python tools/demo.py A2B -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --seeds 85,100,75,458,1500 --A2B_mixing_seed w_avg --col_styles 0,1,2,3,4,5,6 --frames 120 --video_fps 30 --save_result --device gpu

(w_avg妈妈提供皮肤！！！)
python tools/demo.py A2B -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --seeds 85,100,75,458,1500 --A2B_mixing_seed w_avg --col_styles 7,8,9,10,11,12,13 --frames 120 --video_fps 30 --save_result --device gpu

(更低的分辨率使用了w_avg妈妈的潜在因子，人物更像w_avg妈妈！！！)
python tools/demo.py A2B -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --seeds 85,100,75,458,1500 --A2B_mixing_seed w_avg --col_styles 2,3,4,5,6,7,8,9,10,11,12,13 --frames 120 --video_fps 30 --save_result --device gpu



python tools/demo.py A2B -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --seeds 0_20 --frames 120 --video_fps 30 --save_result --device gpu


(styleganv3)
python tools/demo.py A2B -f exps/styleganv3/styleganv3_s_256_custom.py -c StyleGANv3_outputs/styleganv3_s_256_custom/77.pth --seeds 85,100,75,458,1500 --frames 120 --video_fps 30 --save_result --device gpu

python tools/demo.py A2B -f exps/styleganv3/styleganv3_s_256_custom.py -c StyleGANv3_outputs/styleganv3_s_256_custom/77.pth --seeds 85,100,75,458,1500 --A2B_mixing_seed 458 --col_styles 0,1,2,3,4,5,6 --frames 120 --video_fps 30 --save_result --device gpu

python tools/demo.py A2B -f exps/styleganv3/styleganv3_s_256_custom.py -c StyleGANv3_outputs/styleganv3_s_256_custom/77.pth --seeds 85,100,75,458,1500 --A2B_mixing_seed 458 --col_styles 7,8,9,10,11,12,13,14,15 --frames 120 --video_fps 30 --save_result --device gpu

python tools/demo.py A2B -f exps/styleganv3/styleganv3_s_256_custom.py -c StyleGANv3_outputs/styleganv3_s_256_custom/77.pth --seeds 85,100,75,458,1500 --A2B_mixing_seed 458 --col_styles 2,3,4,5,6,7,8,9,10,11,12,13,14,15 --frames 120 --video_fps 30 --save_result --device gpu

(w_avg妈妈提供动作姿态！！！神仙颜值，什么发色瞳色都能驾驭！！！)
python tools/demo.py A2B -f exps/styleganv3/styleganv3_s_256_custom.py -c StyleGANv3_outputs/styleganv3_s_256_custom/77.pth --seeds 85,100,75,458,1500 --A2B_mixing_seed w_avg --col_styles 0,1,2,3,4,5,6 --frames 120 --video_fps 30 --save_result --device gpu

(w_avg妈妈提供皮肤！！！)
python tools/demo.py A2B -f exps/styleganv3/styleganv3_s_256_custom.py -c StyleGANv3_outputs/styleganv3_s_256_custom/77.pth --seeds 85,100,75,458,1500 --A2B_mixing_seed w_avg --col_styles 7,8,9,10,11,12,13,14,15 --frames 120 --video_fps 30 --save_result --device gpu

(更低的分辨率使用了w_avg妈妈的潜在因子，人物更像w_avg妈妈！！！)
python tools/demo.py A2B -f exps/styleganv3/styleganv3_s_256_custom.py -c StyleGANv3_outputs/styleganv3_s_256_custom/77.pth --seeds 85,100,75,458,1500 --A2B_mixing_seed w_avg --col_styles 2,3,4,5,6,7,8,9,10,11,12,13,14,15 --frames 120 --video_fps 30 --save_result --device gpu




(afhq，你会发现stylegan2ada特有的“屏幕粘毛”视觉观感)
python tools/demo.py A2B -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c styleganv2ada_512_afhqcat.pth --seeds 85,100,75,458,1500 --frames 120 --video_fps 30 --save_result --device gpu

python tools/demo.py A2B -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c styleganv2ada_512_afhqcat.pth --seeds 458,1500 --frames 120 --video_fps 30 --save_result --device gpu

(afhqv2，你会发现stylegan3解决了“屏幕粘毛”的问题)
python tools/demo.py A2B -f exps/styleganv3/styleganv3_r_512_afhqv2.py -c stylegan3_r_afhqv2_512.pth --seeds 85,100,75,458,1500 --frames 120 --video_fps 30 --save_result --device gpu


(metfaces)
python tools/demo.py A2B -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c styleganv2ada_1024_metfaces.pth --seeds 85,100,75,458,1500 --frames 120 --video_fps 30 --save_result --device gpu

python tools/demo.py A2B -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c styleganv2ada_1024_metfaces.pth --seeds 0_76 --frames 120 --video_fps 30 --save_result --device gpu



----------------------- style_mixing -----------------------
col_styles 0,1,2,3,4,5,6表示的是row_seeds生成的潜在因子的0,1,2,3,4,5,6个被替换成了col_seeds的0,1,2,3,4,5,6个，
即col_seeds提供了动作姿态，row_seeds提供了皮肤。具体解析请看readme_styleganv2ada_note.txt


python tools/demo.py style_mixing -f exps/styleganv2ada/styleganv2ada_128_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_128_custom/48.pth --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu


python tools/demo.py style_mixing -f exps/styleganv2ada/styleganv2ada_512_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_512_custom/67.pth --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu


python tools/demo.py style_mixing -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu

python tools/demo.py style_mixing -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 7,8,9,10,11,12,13 --save_result --device gpu


(styleganv3)
python tools/demo.py style_mixing -f exps/styleganv3/styleganv3_s_256_custom.py -c StyleGANv3_outputs/styleganv3_s_256_custom/77.pth --seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu


(afhq)
python tools/demo.py style_mixing -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c styleganv2ada_512_afhqcat.pth --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu


(afhqv2，如果显存不足)
python tools/demo.py style_mixing -f exps/styleganv3/styleganv3_r_512_afhqv2.py -c stylegan3_r_afhqv2_512.pth --row_seeds 85 --col_seeds 55 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu


(metfaces)
python tools/demo.py style_mixing -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c styleganv2ada_1024_metfaces.pth --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu


python tools/demo.py style_mixing -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c styleganv2ada_1024_ffhq.pth --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu


----------------------- 获取真实图片的潜在因子ws（真实图片投影到潜在空间） -----------------------
基于优化的方法。从w_avg妈妈开始，渐变成--target_fname指定的图片。

python tools/demo.py projector -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --outdir out --target_fname ./target_imgs/xy.jpg --save_result --device gpu


python tools/demo.py projector -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --outdir out --target_fname ./target_imgs/hqsw.jpg --save_result --device gpu


python tools/demo.py projector -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --outdir out --target_fname ./target_imgs/000138-01.jpg --save_result --device gpu


python tools/demo.py projector -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c styleganv2ada_512_afhqcat.pth --outdir out --target_fname ./target_imgs/flickr_cat_000008.jpg --save_result --device gpu


python tools/demo.py projector -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c styleganv2ada_1024_ffhq.pth --outdir out --target_fname ./target_imgs/xyjy.jpg --save_result --device gpu


(styleganv3)
python tools/demo.py projector -f exps/styleganv3/styleganv3_s_256_custom.py -c StyleGANv3_outputs/styleganv3_s_256_custom/77.pth --outdir out --target_fname ./target_imgs/xy.jpg --save_result --device gpu

python tools/demo.py projector -f exps/styleganv3/styleganv3_s_256_custom.py -c StyleGANv3_outputs/styleganv3_s_256_custom/77.pth --outdir out --target_fname ./target_imgs/hqsw.jpg --save_result --device gpu

python tools/demo.py projector -f exps/styleganv3/styleganv3_s_256_custom.py -c StyleGANv3_outputs/styleganv3_s_256_custom/77.pth --outdir out --target_fname ./target_imgs/000138-01.jpg --save_result --device gpu



----------------------- 训练 -----------------------
后台启动：
nohup xxx     > stylegan2ada.log 2>&1 &


nohup python tools/train.py -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -d 1 -b 8 -eb 1     > stylegan2ada.log 2>&1 &



----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
后台启动：
nohup xxx     > stylegan2ada.log 2>&1 &


迁移学习动漫头像数据集：
CUDA_VISIBLE_DEVICES=0
nohup python tools/train.py -f exps/styleganv2ada/styleganv2ada_256_custom.py -d 1 -b 6 -eb 1 -c styleganv2ada_512_afhqcat.pth     > stylegan2ada.log 2>&1 &


CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/styleganv2ada/styleganv2ada_256_custom.py -d 2 -b 8 -eb 2 -c styleganv2ada_512_afhqcat.pth     > stylegan2ada.log 2>&1 &


迁移学习上衣256数据集：
CUDA_VISIBLE_DEVICES=0
nohup python tools/train.py -f exps/styleganv2ada/styleganv2ada_256_uppercloth.py -d 1 -b 6 -eb 1 -c styleganv2ada_512_afhqcat.pth     > stylegan2ada.log 2>&1 &


CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/styleganv2ada/styleganv2ada_256_uppercloth.py -d 2 -b 8 -eb 2 -c styleganv2ada_512_afhqcat.pth     > stylegan2ada.log 2>&1 &




----------------------- Linux常用命令 -----------------------
查看日志
cat stylegan2ada.log

查看日志(最后20行)
tail -n 20 stylegan2ada.log


看显存占用、GPU利用率
watch -n 0.1 nvidia-smi


虚拟环境相关：
conda create -n pasta python=3.9

conda activate pasta

export CUDA_VISIBLE_DEVICES=0

export CUDA_VISIBLE_DEVICES=1

export CUDA_VISIBLE_DEVICES=4




----------------------- 恢复训练（加上参数--resume） -----------------------
后台启动：
nohup xxx     > stylegan2ada.log 2>&1 &


迁移学习动漫头像数据集：
CUDA_VISIBLE_DEVICES=0
nohup python tools/train.py -f exps/styleganv2ada/styleganv2ada_256_custom.py -d 1 -b 6 -eb 1 -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/0_1000.pth --resume     > stylegan2ada.log 2>&1 &


CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/styleganv2ada/styleganv2ada_256_custom.py -d 2 -b 8 -eb 2 -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/0_1000.pth --resume     > stylegan2ada.log 2>&1 &


迁移学习上衣256数据集：
CUDA_VISIBLE_DEVICES=0
nohup python tools/train.py -f exps/styleganv2ada/styleganv2ada_256_uppercloth.py -d 1 -b 6 -eb 1 -c StyleGANv2ADA_outputs/styleganv2ada_256_uppercloth/0_1000.pth --resume     > stylegan2ada.log 2>&1 &


CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/styleganv2ada/styleganv2ada_256_uppercloth.py -d 2 -b 8 -eb 2 -c StyleGANv2ADA_outputs/styleganv2ada_256_uppercloth/0_1000.pth --resume     > stylegan2ada.log 2>&1 &




----------------------- 计算指标 -----------------------
转换inceptionv3的权重：
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt
python tools/inception_convert_weights.py

(afhq)
python tools/calc_metrics.py -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c styleganv2ada_512_afhqcat.pth -b 2 -n 50000 --inceptionv3_path inception-2015-12-05.pth --device gpu


(动漫头像数据集)
python tools/calc_metrics.py -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth -b 2 -n 50000 --inceptionv3_path inception-2015-12-05.pth --device gpu


1机1卡，总批大小6，23.pth，
2022-04-29 18:33:53.665 | INFO     | __main__:calc_stylegan2ada_metric:230 - total time: 1839.711170s
2022-04-29 18:33:53.665 | INFO     | __main__:calc_stylegan2ada_metric:231 - Speed: 0.036794s per image,  27.2 FPS.
2022-04-29 18:33:57.580 | INFO     | __main__:calc_stylegan2ada_metric:239 - FID: 14.723749

1机2卡，总批大小8，22.pth，
2022-05-07 10:17:41.038 | INFO     | __main__:calc_stylegan2ada_metric:230 - total time: 1744.382321s
2022-05-07 10:17:41.039 | INFO     | __main__:calc_stylegan2ada_metric:231 - Speed: 0.034888s per image,  28.7 FPS.
2022-05-07 10:17:45.055 | INFO     | __main__:calc_stylegan2ada_metric:239 - FID: 14.092427

1机2卡，总批大小8，32.pth，
2022-05-07 11:08:10.423 | INFO     | __main__:calc_stylegan2ada_metric:230 - total time: 1748.136298s
2022-05-07 11:08:10.426 | INFO     | __main__:calc_stylegan2ada_metric:231 - Speed: 0.034963s per image,  28.6 FPS.
2022-05-07 11:08:14.463 | INFO     | __main__:calc_stylegan2ada_metric:239 - FID: 12.598085

1机2卡，总批大小8，65.pth，
2022-05-09 10:03:12.317 | INFO     | __main__:calc_stylegan2ada_metric:230 - total time: 1754.911271s
2022-05-09 10:03:12.317 | INFO     | __main__:calc_stylegan2ada_metric:231 - Speed: 0.035098s per image,  28.5 FPS.
2022-05-09 10:03:16.335 | INFO     | __main__:calc_stylegan2ada_metric:239 - FID: 8.209035


(styleganv3，1机2卡，总批大小8，40.pth)
python tools/calc_metrics.py -f exps/styleganv3/styleganv3_s_256_custom.py -c StyleGANv3_outputs/styleganv3_s_256_custom/77.pth -b 2 -n 50000 --inceptionv3_path inception-2015-12-05.pth --device gpu

2022-05-23 15:47:35.642 | INFO     | __main__:calc_stylegan2ada_metric:230 - total time: 4849.241369s
2022-05-23 15:47:35.643 | INFO     | __main__:calc_stylegan2ada_metric:231 - Speed: 0.096985s per image,  10.3 FPS.
2022-05-23 15:47:39.536 | INFO     | __main__:calc_stylegan2ada_metric:239 - FID: 10.792626

1机2卡，总批大小8，77.pth，
2022-05-27 13:59:12.171 | INFO     | __main__:calc_stylegan2ada_metric:230 - total time: 5360.057720s
2022-05-27 13:59:12.172 | INFO     | __main__:calc_stylegan2ada_metric:231 - Speed: 0.107201s per image,  9.3 FPS.
2022-05-27 13:59:16.578 | INFO     | __main__:calc_stylegan2ada_metric:239 - FID: 8.085018



----------------------- 导出为ncnn -----------------------
python tools/demo.py ncnn -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c styleganv2ada_512_afhqcat.pth --ncnn_output_path styleganv2ada_512_afhqcat --seeds 0_1500

python tools/demo.py ncnn -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/65.pth --ncnn_output_path styleganv2ada_256_custom_epoch_65 --seeds 0_1500

python tools/demo.py ncnn -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c styleganv2ada_1024_metfaces.pth --ncnn_output_path styleganv2ada_1024_metfaces --seeds 0_1500

python tools/demo.py ncnn -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c styleganv2ada_1024_ffhq.pth --ncnn_output_path styleganv2ada_1024_ffhq --seeds 0_1500




cd build/examples


./stylegan 0 512 16 1.0 seeds/458.bin styleganv2ada_512_afhqcat_mapping.param styleganv2ada_512_afhqcat_mapping.bin styleganv2ada_512_afhqcat_synthesis.param styleganv2ada_512_afhqcat_synthesis.bin

./stylegan 1 512 16 1.0 seeds/458.bin seeds/293.bin styleganv2ada_512_afhqcat_mapping.param styleganv2ada_512_afhqcat_mapping.bin styleganv2ada_512_afhqcat_synthesis.param styleganv2ada_512_afhqcat_synthesis.bin 0 1 2 3 4 5 6

./stylegan 2 512 16 1.0 seeds/458.bin seeds/293.bin styleganv2ada_512_afhqcat_mapping.param styleganv2ada_512_afhqcat_mapping.bin styleganv2ada_512_afhqcat_synthesis.param styleganv2ada_512_afhqcat_synthesis.bin 120 30



./stylegan 0 512 14 1.0 seeds/85.bin styleganv2ada_256_custom_epoch_65_mapping.param styleganv2ada_256_custom_epoch_65_mapping.bin styleganv2ada_256_custom_epoch_65_synthesis.param styleganv2ada_256_custom_epoch_65_synthesis.bin

./stylegan 1 512 14 1.0 seeds/85.bin seeds/100.bin styleganv2ada_256_custom_epoch_65_mapping.param styleganv2ada_256_custom_epoch_65_mapping.bin styleganv2ada_256_custom_epoch_65_synthesis.param styleganv2ada_256_custom_epoch_65_synthesis.bin 0 1 2 3 4 5 6

./stylegan 2 512 14 1.0 seeds/85.bin seeds/100.bin styleganv2ada_256_custom_epoch_65_mapping.param styleganv2ada_256_custom_epoch_65_mapping.bin styleganv2ada_256_custom_epoch_65_synthesis.param styleganv2ada_256_custom_epoch_65_synthesis.bin 120 30



./stylegan 0 512 18 1.0 seeds/458.bin styleganv2ada_1024_metfaces_mapping.param styleganv2ada_1024_metfaces_mapping.bin styleganv2ada_1024_metfaces_synthesis.param styleganv2ada_1024_metfaces_synthesis.bin

./stylegan 1 512 18 1.0 seeds/458.bin seeds/293.bin styleganv2ada_1024_metfaces_mapping.param styleganv2ada_1024_metfaces_mapping.bin styleganv2ada_1024_metfaces_synthesis.param styleganv2ada_1024_metfaces_synthesis.bin 0 1 2 3 4 5 6



./stylegan 0 512 18 1.0 seeds/458.bin styleganv2ada_1024_ffhq_mapping.param styleganv2ada_1024_ffhq_mapping.bin styleganv2ada_1024_ffhq_synthesis.param styleganv2ada_1024_ffhq_synthesis.bin

./stylegan 1 512 18 1.0 seeds/458.bin seeds/293.bin styleganv2ada_1024_ffhq_mapping.param styleganv2ada_1024_ffhq_mapping.bin styleganv2ada_1024_ffhq_synthesis.param styleganv2ada_1024_ffhq_synthesis.bin 0 1 2 3 4 5 6




----------------------- 导出为TensorRT -----------------------





