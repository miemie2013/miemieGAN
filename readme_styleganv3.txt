# 如果命令不能成功执行，说明咩酱实现中，，，
nvidia-smi


pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


梯度对齐：
1.(原版仓库也要设置)设置 StyleGANv3Model 的
    self.augment_pipe = None
    self.style_mixing_prob = -1.0
    self.align_grad = True
解除上面语句的注释即可。
以及，对下面所有的以if self.align_grad:开头的代码块解除注释
if self.align_grad:
    xxx

2.计算loss_Gpl那里，
pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
改为
pl_noise = torch.ones_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
3.trainer.py下面代码解除注释
        # 对齐梯度用
        # if (self.iter + 1) == 20:
        #     self.save_ckpt(ckpt_name="%d" % (self.epoch + 1))
需要注释掉下面的代码：
            if resume:
                # 需要修改配置
                model.ada_kimg = 100        # Make ADA react faster at the beginning.
                model.ema_rampup = None     # Disable EMA rampup.
                model.blur_init_sigma = 0   # Disable blur rampup.
因为stylegan3-m2仓库是随机初始化的权重，没有resume权重。


4.不需要设置 StyleGANv3_SynthesisNetwork 的 （因为显存不够）
    use_fp16 = False
5.不需要设置 StyleGANv3_Discriminator 的 （因为显存不够）
    use_fp16 = False

6. styleganv3_method_base.py 优化器要换成SGD：
                optimizer = torch.optim.SGD(
                    itertools.chain(self.model.synthesis.parameters(), self.model.mapping.parameters()), lr=0.00001, momentum=0.9
                )
                ...
                optimizer = torch.optim.SGD(
                    self.model.discriminator.parameters(), lr=0.00002, momentum=0.9
                )
因为Adam更新参数有一定随机性，同样的情况下，跑2次结果不同！！！（但是SGD也有轻微的不同，影响不大。）


python tools/convert_weights.py -f exps/styleganv3/styleganv3_r_32_custom.py -c_G G_00.pth -c_Gema G_ema_00.pth -c_D D_00.pth -oc stylegan3_r_32_00.pth


python tools/convert_weights.py -f exps/styleganv3/styleganv3_r_32_custom.py -c_G G_19.pth -c_Gema G_ema_19.pth -c_D D_19.pth -oc stylegan3_r_32_19.pth


python tools/train.py -f exps/styleganv3/styleganv3_r_32_custom.py -d 1 -b 2 -eb 1 -c stylegan3_r_32_00.pth


python diff_weights.py --cp1 stylegan3_r_32_19.pth --cp2 StyleGANv3_outputs/styleganv3_r_32_custom/1.pth --d_value 0.5


python diff_weights.py --cp1 stylegan3_r_32_19.pth --cp2 stylegan3_r_32_00.pth --d_value 0.5





原仓库中，docs/configs.md中
--gamma=16.4  表示  self.model_cfg['r1_gamma']=16.4
--mirror=1  表示  self.dataset_train_cfg['xflip']=True
--kimg如果不指定，默认是25000

作者说，r1_gamma对数据集分辨率比较敏感，
建议试着用推荐的r1_gamma、2倍的r1_gamma、4倍的r1_gamma、0.5倍的r1_gamma、0.25倍的r1_gamma训练，选择产生最低FID的即可。

作者说，迁移学习推荐--kimg=5000


----------------------- 转换权重 -----------------------
python tools/convert_weights.py -f exps/styleganv3/styleganv3_r_512_afhqv2.py -c_G G_afhqv2_r_512.pth -c_Gema G_ema_afhqv2_r_512.pth -c_D D_afhqv2_r_512.pth -oc stylegan3_r_afhqv2_512.pth

python tools/convert_weights.py -f exps/styleganv3/styleganv3_t_1024_metfaces.py -c_G G_metfaces_t_1024.pth -c_Gema G_ema_metfaces_t_1024.pth -c_D D_metfaces_t_1024.pth -oc stylegan3_t_metfaces_1024.pth

python tools/convert_weights.py -f exps/styleganv3/styleganv3_r_1024_ffhq.py -c_G G_ffhq_r_1024.pth -c_Gema G_ema_ffhq_r_1024.pth -c_D D_ffhq_r_1024.pth -oc stylegan3_r_ffhq_1024.pth

python tools/convert_weights.py -f exps/styleganv3/styleganv3_t_512_afhqv2.py -c_G G_afhqv2_t_512.pth -c_Gema G_ema_afhqv2_t_512.pth -c_D D_afhqv2_t_512.pth -oc stylegan3_t_afhqv2_512.pth

python tools/convert_weights.py -f exps/styleganv3/styleganv3_t_1024_ffhq.py -c_G G_ffhq_t_1024.pth -c_Gema G_ema_ffhq_t_1024.pth -c_D D_ffhq_t_1024.pth -oc stylegan3_t_ffhq_1024.pth

python tools/convert_weights.py -f exps/styleganv3/styleganv3_t_1024_metfaces.py -c_G G_metfaces_t_1024.pth -c_Gema G_ema_metfaces_t_1024.pth -c_D D_metfaces_t_1024.pth -oc stylegan3_t_metfaces_1024.pth



----------------------- 预测 -----------------------
python tools/demo.py image -f exps/styleganv3/styleganv3_r_512_afhqv2.py -c stylegan3_r_afhqv2_512.pth --seeds 85,100,75,458,1500 --save_result --device gpu

python tools/demo.py image -f exps/styleganv3/styleganv3_t_1024_metfaces.py -c stylegan3_t_metfaces_1024.pth --seeds 85,100,75,458,1500 --save_result --device gpu

python tools/demo.py image -f exps/styleganv3/styleganv3_r_1024_ffhq.py -c stylegan3_r_ffhq_1024.pth --seeds 85,100,75,458,1500 --save_result --device gpu




----------------------- style_mixing -----------------------
python tools/demo.py style_mixing -f exps/styleganv3/styleganv3_r_512_afhqv2.py -c stylegan3_r_afhqv2_512.pth --row_seeds 85,100,75,458,1500 --col_seeds 55,821,1789,293 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu

(如果显存不足)
python tools/demo.py style_mixing -f exps/styleganv3/styleganv3_r_512_afhqv2.py -c stylegan3_r_afhqv2_512.pth --row_seeds 85 --col_seeds 55 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu


python tools/demo.py style_mixing -f exps/styleganv3/styleganv3_t_1024_metfaces.py -c stylegan3_t_metfaces_1024.pth --row_seeds 85 --col_seeds 55 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu


python tools/demo.py style_mixing -f exps/styleganv3/styleganv3_r_1024_ffhq.py -c stylegan3_r_ffhq_1024.pth --row_seeds 85 --col_seeds 55 --col_styles 0,1,2,3,4,5,6 --save_result --device gpu





----------------------- 评估 -----------------------



----------------------- 训练 -----------------------
后台启动：
nohup xxx     > stylegan3.log 2>&1 &


python tools/train.py -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -d 1 -b 8 -eb 1


(从头训练动漫头像数据集，styleganv3_s是自己新增的配置，减小了channel_base、channel_max，因为stylegan3太吃显存了！)
CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/styleganv3/styleganv3_s_256_custom.py -d 2 -b 8 -eb 2     > stylegan3.log 2>&1 &

CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/styleganv3/styleganv3_s_256_custom.py -d 2 -b 8 -eb 2 -c StyleGANv3_outputs/styleganv3_s_256_custom/0_1000.pth --resume     > stylegan3.log 2>&1 &


tail -n 20 stylegan3.log


----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
后台启动：
nohup xxx     > stylegan3.log 2>&1 &



python tools/train.py -f exps/styleganv3/styleganv3_r_32_custom.py -d 1 -b 2 -eb 1 -c stylegan3_r_afhqv2_512.pth


（花卉数据集，1机2卡训练时，总批大小只能开到2，每张卡占9G左右显存）
python tools/train.py -f exps/styleganv3/styleganv3_r_128_custom.py -d 1 -b 1 -eb 1 -c stylegan3_r_afhqv2_512.pth

CUDA_VISIBLE_DEVICES=0,1
nohup python tools/train.py -f exps/styleganv3/styleganv3_r_128_custom.py -d 2 -b 2 -eb 2 -c stylegan3_r_afhqv2_512.pth     > stylegan3.log 2>&1 &


tail -n 20 stylegan3.log


python tools/train.py -f exps/styleganv3/styleganv3_r_256_custom.py -d 1 -b 2 -eb 1 -c stylegan3_r_afhqv2_512.pth



python tools/train.py -f exps/styleganv3/styleganv3_t_256_custom.py -d 1 -b 2 -eb 1 -c stylegan3_t_metfaces_1024.pth




nohup python tools/train.py -f exps/styleganv2ada/styleganv2ada_512_custom.py -d 1 -b 4 -eb 1 -c styleganv2ada_512_afhqcat.pth > stylegan2ada.log 2>&1 &



看显存占用、GPU利用率
watch -n 0.1 nvidia-smi


conda create -n pasta python=3.9

conda activate pasta

export CUDA_VISIBLE_DEVICES=0

export CUDA_VISIBLE_DEVICES=1

export CUDA_VISIBLE_DEVICES=4




----------------------- 恢复训练（加上参数--resume） -----------------------
python tools/train.py -f exps/styleganv2ada/styleganv2ada_512_custom.py -d 1 -b 4 -eb 1 -c StyleGANv2ADA_outputs/styleganv2ada_512_custom/7.pth --resume


nohup python tools/train.py -f exps/styleganv2ada/styleganv2ada_512_custom.py -d 1 -b 4 -eb 1 -c StyleGANv2ADA_outputs/styleganv2ada_512_custom/7.pth --resume > stylegan2ada.log 2>&1 &


nohup python tools/train.py -f exps/styleganv2ada/styleganv2ada_128_custom.py -d 1 -b 14 -eb 1 -c StyleGANv2ADA_outputs/styleganv2ada_128_custom/7.pth --resume > stylegan2ada_128.log 2>&1 &




----------------------- 导出为ONNX -----------------------




----------------------- 导出为TensorRT -----------------------





