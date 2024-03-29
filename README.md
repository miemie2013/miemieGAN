简体中文 | [English](README_en.md)

# miemiegan

## 概述
miemieGAN是[咩酱](https://github.com/miemie2013)基于[YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)进行二次开发的个人检测库（使用的深度学习框架为pytorch）， 支持单机单卡、单机多卡、多机多卡训练模式（多卡训练模式建议使用Linux系统），支持Windows、Linux系统，以咩酱的名字命名。miemieGAN是一个不需要安装的检测库用户可以直接更改其代码改变执行逻辑，所见即所得！所以往miemieGAN里加入新的算法是一件很容易的事情。得益于YOLOX的优秀架构，miemieGAN里的算法训练速度都非常快，数据读取不再是训练速度的瓶颈！目前miemieGAN支持StyleGAN2ADA、StyleGAN3等算法，预计未来会加入更多算法，所以请大家点个star吧！


## 安装依赖

```
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install torch==1.10.1+cu102 torchvision==0.11.2+cu102 torchaudio==0.10.1 -f https://download.pytorch.org/whl/torch_stable.html
```
torch版本建议1.10.1+cu102或者更高；torchvision版本建议0.11.2+cu102或者更高。

## 支持的算法

- [StyleGAN2ADA](docs/README_StyleGAN.md)
- [StyleGAN3](docs/README_StyleGAN.md)

## Updates!!
* 【2022/07/12】 StyleGAN3算法支持导出到NCNN！也支持生成视频！详情请参考[README_StyleGAN](docs/README_StyleGAN.md) 文档的“如何导出ncnn?”小节。
* 【2022/07/08】 StyleGAN2ADA算法支持导出到NCNN！详情请参考[README_StyleGAN](docs/README_StyleGAN.md) 文档的“如何导出ncnn?”小节。技术细节请参考本人的知乎文章[全网第一个！stylegan2-ada和stylegan3的pytorch实现二合一！支持导出ncnn！](https://zhuanlan.zhihu.com/p/539140181)

## 传送门

算法1群：645796480（人已满） 

算法2群：894642886 

粉丝群：704991252

关于仓库的疑问尽量在Issues上提，避免重复解答。

B站不定时女装: [_糖蜜](https://space.bilibili.com/646843384)

知乎不定时谢邀、写文章: [咩咩2013](https://www.zhihu.com/people/mie-mie-2013)

西瓜视频: [咩咩2013](https://www.ixigua.com/home/2088721227199148/?list_entrance=search)

微信：wer186259

本人微信公众号：miemie_2013

技术博客：https://blog.csdn.net/qq_27311165

AIStudio主页：[asasasaaawws](https://aistudio.baidu.com/aistudio/personalcenter/thirdview/165135)

欢迎在GitHub或上面的平台关注我（求粉）~


## 打赏

如果你觉得这个仓库对你很有帮助，可以给我打钱↓

![Example 0](weixin/sk.png)

咩酱爱你哟！


## 引用

```
```
