---
title: Paper Reading
top: false
cover: false
toc: true
mathjax: true
date: 2022-01-05 18:01:04
password:
summary:
description:
categories:
- Papers
tags:
---

# Differentiable Graph Module(DGM)for Graph Convolutional Networks_PAMI-2022



> 这篇文章基于每一层的输出特征来学习 graph，并在训练过程中优化网络参数。     所提出的架构包括 Differentiable Graph Module(DGM) 和 Diffusion Module 两个模块。

> latent graph: 
>
> In many problems, the data can be assumed to have some underlying graph structure, however, the graph itself might not be explicitly given, a setting we refer to as latent graph.

## Differentiable Graph Module

<font color=green>**DGM --> 构建表示输入空间的 weighted graph。**</font>即：

> 输入：feature matrix X $\in$ $\mathcal{R}^{N \times d}$ 或 initial graph $\mathcal{G_0}$
>
> 输出：graph $\mathcal{G}$

> DGM 由两部分组成：
>
> ① 将输入特征转换为辅助特征 auxiliary features；
>
> ② 用辅助特征构造 graph。

> ① 将输入特征转换为辅助特征：
>
> 输入特征 $X \in \mathcal{R}^{N \times d}$
>
> 辅助特征 $\hat{X}=\mathcal{f}_\theta(x) \in \mathcal{R}^{N \times \hat{d}}$ 
>
> $$ \hat{X}=\mathcal{f}_\theta(x) \left\{
>
> \begin{array}{rcl}
>
> 对 \mathcal{G_0} 进行 edge-/graph-convolution 得到     &        & {if \mathcal{G_0}已知}\\
>
> \mathcal{f_{\theta}} 独立应用于每个节点特征，按行作用于矩阵 X     &        &{otherwise}
>
> \end{array} \right. $$















## Diffusion Module

















# Dynamic Graph Convolutional Networks_PR-2020

dynamic graph --> 指每个 graph 的顶点 / 边随时间的变化而变化。

面对很多不同的分类任务，首先需要对结构数据(structured data) 进行处理，通常的处理方式是 --> 将这些结构数据建模为 graphs。

而针对那些顶点 / 边随时间的变化而变化的 dynamic graph 来说，目标则是 --> 利用现有的神经网络将这些数据集建模为随时间变化而变化的图结构(graph structures) 。--> 由于使用现有的架构不能解决目标问题，所以作者提出两种方法来实现这个目标，即：结合 Long Short-Term Memory networks 和 Graph Convolutional Networks 来学习长短期依赖关系和图结构。













































