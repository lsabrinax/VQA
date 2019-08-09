# VQA
介绍当前解决VQA问题的基本思路以及在基本网络结构上的各种改进模型
## 基本网络结构
![基本网络结构]()
- [图像特征提取](#图像特征提取)
- [问题特征提取](#问提特征提取)
- [特征融合](#特征融合)
- [答案输出](#答案输出)
网络结构主要包含这四个部分，后续的文章也主要从这四个不同的方面对网络模型做出改进
## 主流网络模型
### 图像特征提取
- 整图特征
- 基于Edge Boxes
	- [Where To Look: Focus Regions for Visual Question Answering](#where to look)
- 基于 Uniform Grid
- 基于 Region Proposal
	- [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering](http://openaccess.thecvf.com/content_cvpr_2018/papers/Anderson_Bottom-Up_and_Top-Down_CVPR_2018_paper.pdf)(CVPR2018)
### 问题特征提取
- LSTM
- GRU
- skip-thought vectors
### 特征融合
- 简单机制
- 双线性池化(bilinear pooling)
	- MCB:[Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding](https://arxiv.org/pdf/1606.01847)
		- ![MCB]()
		- 双线性池化(向量外积)在提取多模态的相关特征上有更强的表达能力,但是会带来参数爆炸的问题
		- 提出MCB模块，用count sketches映射函数将向量外积转化为向量的卷积,降低参数维度,并用快速傅里叶变换优化计算速度
	- MLB:[HADAMARD PRODUCT FOR LOW-RANK BILINEAR POOLING](https://arxiv.org/pdf/1610.04325)(ICLR2017)
		- ![MLB]()
	- [MUTAN: Multimodal Tucker Fusion for Visual Question Answering](http://openaccess.thecvf.com/content_ICCV_2017/papers/Ben-younes_MUTAN_Multimodal_Tucker_ICCV_2017_paper.pdf)(ICCV2017)
		- ![MUTAN]()
		- 对MCB和MLB提出了改进,用tucker分解的方法将一个高维度的三维张量分解为三个二维矩阵和一个三维的核张量，进一步降低了参数量并提高了提取图像和文本特征相关性的能力
		- 对核张量添加了秩的限制，进一度降低数据维度，增加了模型的可解释性
- attention机制
	- <span id="where to look">[Where To Look: Focus Regions for Visual Question Answering](http://openaccess.thecvf.com/content_cvpr_2016/papers/Shih_Where_to_Look_CVPR_2016_paper.pdf)(CVPR2016)</span>
		- ![Where To Look]()
		- 利用图像特征和问题特征的点积产生attention map,对图像和文本特征做加权平均
	- [Hierarchical Question-Image Co-Attention for Visual Question Answering](http://papers.nips.cc/paper/6202-hierarchical-question-image-co-attention-for-visual-question-answering.pdf)(NIPS2016)
	- [Deep Modular Co-Attention Networks for Visual Question Answering](http://openaccess.thecvf.com/content_CVPR_2019/papers/Yu_Deep_Modular_Co-Attention_Networks_for_Visual_Question_Answering_CVPR_2019_paper.pdf)(CVPR2019)
	- 
### 答案输出
- 分类模型
- 生成模型
### 其他
- 问题分解
	- [Neural Module Networks](http://openaccess.thecvf.com/content_cvpr_2016/papers/Andreas_Neural_Module_Networks_CVPR_2016_paper.pdf)(CVPR2016)
	- [Learning to Reason: End-to-End Module Networks for Visual Question Answering](http://openaccess.thecvf.com/content_ICCV_2017/papers/Hu_Learning_to_Reason_ICCV_2017_paper.pdf)(ICCV2017)
