# 如果命令不能成功执行，说明咩酱实现中，，，
nvidia-smi


pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


梯度对齐：
1.(原版仓库也要设置)设置 StyleGANv2ADAModel 的
    self.augment_pipe = None
    self.style_mixing_prob = -1.0
2.设置学习率与原版仓库相等
3.(原版仓库也要设置)设置 SynthesisLayer 的
    self.use_noise = False
4.(原版仓库也要设置)设置 StyleGANv2ADA_SynthesisNetwork 的
    use_fp16 = False
5.(原版仓库也要设置)设置 StyleGANv2ADA_Discriminator 的
    use_fp16 = False
6.计算loss_Gpl那里，
pl_noise = torch.randn_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])
改为
pl_noise = torch.ones_like(gen_img) / np.sqrt(gen_img.shape[2] * gen_img.shape[3])

7.如果显存不足，借用一下11G的卡

8.原版仓库先设置不让优化器更新参数，即注释掉phase.opt.step()，先对齐前20个step的输出；
输出完全对齐后phase.opt.step()解除注释，再继续对齐。




----------------------- 转换权重 -----------------------
python tools/convert_weights.py -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c_G G_afhqcat.pth -c_Gema G_ema_afhqcat.pth -c_D D_afhqcat.pth -oc styleganv2ada_512_afhqcat.pth



----------------------- 预测 -----------------------
python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c styleganv2ada_512_afhqcat.pth --seeds 85,100,75,458,1500 --save_result --device gpu





----------------------- 评估 -----------------------



----------------------- 训练 -----------------------
python tools/train.py -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -d 1 -b 8 -eb 1



----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
python tools/train.py -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -d 1 -b 2 -eb 1 -c styleganv2ada_512_afhqcat.pth

python tools/train.py -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -d 1 -b 1 -eb 1 -c styleganv2ada_512_afhqcat.pth



----------------------- 恢复训练（加上参数--resume） -----------------------
python tools/train.py -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -d 1 -b 2 -eb 1 -c StyleGANv2ADA_outputs/styleganv2ada_512_afhqcat/7.pth --resume



----------------------- 导出为ONNX -----------------------




----------------------- 导出为TensorRT -----------------------





