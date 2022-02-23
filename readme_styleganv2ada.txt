# 如果命令不能成功执行，说明咩酱实现中，，，
nvidia-smi


pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple


----------------------- 转换权重 -----------------------
python tools/convert_weights.py -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c_G G_afhqcat.pth -c_Gema G_ema_afhqcat.pth -c_D D_afhqcat.pth -oc styleganv2ada_512_afhqcat.pth



----------------------- 预测 -----------------------
python tools/demo.py image -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -c styleganv2ada_512_afhqcat.pth --seeds 85,100,75,458,1500 --save_result --device gpu





----------------------- 评估 -----------------------



----------------------- 训练 -----------------------
python tools/train.py -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -d 1 -b 8 -eb 1



----------------------- 迁移学习，带上-c（--ckpt）参数读取预训练模型。 -----------------------
python tools/train.py -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -d 1 -b 2 -eb 1 -c styleganv2ada_512_afhqcat.pth



----------------------- 恢复训练（加上参数--resume） -----------------------
python tools/train.py -f exps/styleganv2ada/styleganv2ada_512_afhqcat.py -d 1 -b 8 -eb 1 -c PPYOLO_outputs/ppyolo_r50vd_2x/13.pth --resume



----------------------- 导出为ONNX -----------------------




----------------------- 导出为TensorRT -----------------------





