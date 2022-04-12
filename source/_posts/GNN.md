---
title: GNN
top: false
cover: false
toc: true
mathjax: false
date: 2022-04-12 09:00:41
password:
summary:
description:
categories:
- GNN
tags:
- GNN
---

# 从 CNN 到 GNN

CNN --> 用来图区欧氏空间数据的特征，它针对规则的 2D 栅格结构的 image（传统数据），其像素点的排列顺序有明显的上下左右的位置关系。

但在现实生活中，很多 graph（图数据）是从非欧氏空间中生成的，graph 不再是规则的栅格结构。

GCN --> 针对不规则的 graph（图数据），节点之间无空间上的位置关系。

## graph 的定义

在数学中，图是由顶点（Vertex）和连接顶点的边（Edge）构成的。顶点表示研究的对象，边表示两个对象之间的特定关系。

图表示顶点和边的集合，记为 *G = (V, E)*，其中，V是顶点集合，E 是边集合。        设图 G 的顶点数为 N，边数为 M。                                                                                           一条连接顶点 *v<sub>i</sub>* , *v<sub>j</sub>* $ \in $ *V* 的边记为（*v<sub>i</sub>* , *v<sub>j</sub>* ）或者 *e<sub>i j</sub>* 。

<font color =  green>**邻居和度**</font>                                                                                                                                 如果存在一条边连接顶点 *v<sub>i</sub>* 和 *v<sub>j</sub>* ，则称 *v<sub>j</sub>* 是 *v<sub>i</sub>* 的邻居，反之亦然。                                                                              *v<sub>i</sub>* 的所有邻居为集合 *N*(*v<sub>i</sub>*)，即:  *N*(*v<sub>i</sub>*) = {*v<sub>j</sub>* | ∃ *e<sub>i j</sub>* \in E or *e<sub>j i</sub>* \in E}.                                                                                                                 以 *v<sub>i</sub>* 为端点的边的数目称为 *v<sub>i</sub>* 的度（Degree），记为 deg(*v<sub>i</sub>* ),  deg(*v<sub>i</sub>* ) = | *N*(*v<sub>i</sub>*) |.

<font color =  green>**邻接矩阵&关联矩阵&度矩阵**</font>

图 *G = (V, E)*， <font color =  green>**邻接矩阵**</font> *A* 描述图中顶点之间的关系，*A ∈ R<sup>N×N</sup>* ,其定义为：

  $\in$

$S_i$


$\in$





# GNN

## GCN

GCN：将卷积运算从传统 image 推广到 graph 图数据。核心思想：学习一个函数映射 $ f(.) $，通过映射图中的节点 $v_{i}$ 可聚合自身特征 $x_{i}$ 与它的邻居特征 $x_{j} (j \in N(v_{i}))$ 来生成节点 $v_{i}$ 的新表示。

GCN 的方法分为两类：

1）Spectral-based GCN --> 从图信号处理角度引入滤波器来定义卷积。

2）Spatial-based GCN --> 从邻域聚合特征信息。

### Spectral-based Graph Convolutional Networks



### Spatial-based Graph Convolutional Networks































