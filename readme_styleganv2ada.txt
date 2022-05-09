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
python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c StyleGANv2ADA_outputs/styleganv2ada_512_afhqcat/1.pth --seeds 85,100,75,458,1500 --save_result --device gpu


python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_128_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_128_custom/48.pth --seeds 85,100,75,458,1500 --save_result --device gpu


python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/23.pth --seeds 85,100,75,458,1500 --save_result --device gpu


(afhq)
python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c styleganv2ada_512_afhqcat.pth --seeds 85,100,75,458,1500 --save_result --device gpu


(metfaces)
python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c styleganv2ada_1024_metfaces.pth --seeds 85,100,75,458,1500 --save_result --device gpu


python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c styleganv2ada_1024_ffhq.pth --seeds 85,100,75,458,1500 --save_result --device gpu



----------------------- style_mixing -----------------------
python tools/demo.py style_mixing -f exps/styleganv2ada/styleganv2ada_128_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_128_custom/48.pth --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu


python tools/demo.py style_mixing -f exps/styleganv2ada/styleganv2ada_512_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_512_custom/67.pth --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu


python tools/demo.py style_mixing -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/2.pth --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu

python tools/demo.py style_mixing -f exps/styleganv2ada/styleganv2ada_256_custom.py -c StyleGANv2ADA_outputs/styleganv2ada_256_custom/23.pth --row_seeds 85,100 --col_seeds 55,821 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu


(afhq)
python tools/demo.py style_mixing -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c styleganv2ada_512_afhqcat.pth --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu


(metfaces)
python tools/demo.py style_mixing -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c styleganv2ada_1024_metfaces.pth --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu


python tools/demo.py style_mixing -f exps/styleganv2ada/styleganv2ada_1024_metfaces.py -c styleganv2ada_1024_ffhq.pth --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu




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


----------------------- 导出为ONNX -----------------------




----------------------- 导出为TensorRT -----------------------





