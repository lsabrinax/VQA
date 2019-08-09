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
- 基于 Uniform Grid
- 基于 Region Proposal
### 问题特征提取
- LSTM
- GRU
- skip-thought vectors
### 特征融合
- 简单机制
- 双线性池化(bilinear pooling)
	- Multimodal Compact Bilinear Pooling for Visual Question Answering and Visual Grounding
- attention机制
### 答案输出
- 分类模型
- 生成模型
### 其他
- 问题分解
