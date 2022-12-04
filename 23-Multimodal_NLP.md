
# 23. Multimodal NLP

## 23.1 Overview

#### Competition

**[2017 搜狐图文匹配算法大赛](https://biendata.com/competition/luckydata/)**

> 训练集：10万条新闻文本+10万张新闻配图，任务：给定新的新闻文本和新的配图集合，为每一篇新闻文本找到匹配度最高的10张图片并排序 

- [SOHU图文匹配竞赛-方案分享](https://blog.csdn.net/wzmsltw/article/details/73330439)

#### Article

- [这可能是「多模态机器学习」最通俗易懂的介绍 - 2018](https://zhuanlan.zhihu.com/p/53511144)

- [【赛尔笔记】 多模态情感分析简述 - 2020](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247492349&idx=6&sn=40bec66b380eb190d1c3e61918701f8b)

- 【Great】[通用的图像-文本语言表征学习：多模态预训练模型 UNITER - 2020](https://mp.weixin.qq.com/s/GxQ27vY5naaAXtp_ZTV0ZA)

- [如何让BERT拥有视觉感知能力？两种方式将视频信息注入BERT - 2020](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247485842&idx=1&sn=cc24542d51e17533781d5c60bd693cec)

    Video + BERT

- [当NLPer爱上CV：后BERT时代生存指南之VL-BERT篇 - 2020](https://mp.weixin.qq.com/s/s5YIG6rBEy6fZkFLh-CzoA)

    Image + BERT


## 23.2 Feature

类别特征（离散） VS 数值特征（连续）

NLP特征 VS CV特征

Deep VS Wide


#### Paper 

- [An Embedding Learning Framework for Numerical Features in CTR Prediction - Huawei2021](https://arxiv.org/abs/2012.08986)

    **Code**: https://github.com/mindspore-ai/models/tree/master/research/recommend/autodis

        某种实现：
        - AutoDisModule: 
          - ResBlock: = Dropout(LeakyReLU(Linear(64, 64)(x) + alpha * x))
          - tau = Parameter(torch.ones([8])), shape=(8)
          - Dropout(Leaky_ReLU(Linear(3, 64)(x))) -(N, 64)-> K*ResBlock(.) -(N, 64)-> Leaky_ReLU(Linear(64, 8)(.)) -(N, 8)-> Softmax(.*tau) -(N, 8)-> output
        - AutoDisEncoder:
          - for one feature:
            - x ==> [sqrt(x), x, x^2], shape=(N, 3)
            - ME(Meta Embedding): Embedding(8, 128)
            - AutoDisModule(.) * ME -(N, 128)-> embedding 
          - for all K features:
            - list(embedding) -K个(N, 128)-> 
              - 输出embedding: torch.cat(., 0) -(K*N, 128)-> Tensor.view(-1, K, 128) -(N, K, 128)-> 
              - 后接FC: torch.cat(., 1) -(N, K*128)-> Linear(K*128, output_size) -(N, output_size)-> output

    **Article**: [KDD2021｜AutoDis: 连续型特征embedding新方法！](https://zhuanlan.zhihu.com/p/436036309)

        AutoDis：对连续数值型特征进行embedding。优点是高模型容量、端到端训练、连续特征embedding具有唯一表示。

        预备知识：
        
        类别特征由于枚举值有限，一般直接采用查表(look-up)的方式来进行Embedding，e=[E1*x1, E2*x2, ..., En*xn], Ei是二维矩阵，xi是一维onehot向量
        
        然而数值型特征有无限的取值，无法直接查表，目前方法主要有：

        - No Embedding: 直接把原始数值输入DNN，比如在DLRM架构中，使用多层感知机结构对所有数值型特征进行建模, e=[DNN([x1, x2, ..., xn])], xi是一个数值

        - Field Embedding: 学界常用，为每个连续特征域单独学习一个Embedding(ei)，然后用原始数值与之相乘, e=[x1*e1, x2*e2, ..., xn*en], xi是一个数值，ei是一个一维向量
        
        - Discretization：工业界常用，将连续数值进行离散化分桶，然后像类别特征那样直接表查(look-up)，e=[E1*d1(x1), E2*d2(x2), ..., En*dn(xn)], Ei是二维矩阵，xi是数值，di(xi)是离散化后的一维onehot向量

        第1种方法，Capacity of Representations太低，第2种方法Capacity也有限，且不同特征领域有linear scaling relationship，第3种方法会存在SBD(Similar value But Dis-similar embedding)和DBS(Dis-similar value But Same embedding)，以及TPP(Two Phase Problem)无法端到端优化。

        AutoDis介绍：

        大致流程依次如下：ej=f(dj(xj), MEj)
        
        - Meta-Embeddings：为每一个领域(field j)的连续型特征定义一组(Hj个)Meta-Embeddings，即二维矩阵MEj(shape is Hj*d)，每个Meta-Embedding相当于1个bucket
        
        - Automatic Discretization：自动（可学习）对每个领域的特征值离散化，并将其分配到不同的Meta-Embeddings桶中（每个值可能分到一个或多个桶中），即dj(.)
          - xj^hat=Softmax(Wj*(LReLU(wj*xj)+alpha*hj)：xj-->Layer(wj)-->Leaky_ReLU-->Layer(Wj,alpha)-->Softmax-->xj^hat
          - xj^hat=dj(xj)=[xj^hat^1,xj^hat^2,...,xj^hat^h,...,xj^hat^Hj]，xj^hat^h是xj与第h个bucket的Correlation，表示xj离散化到第h个bucket的概率
          - 相比工业界那种hard discretization，AutoDis中是soft discretization，能改善SBD和DBS问题，也支持end-to-end
         
        - Aggregation Function：将多个桶的Embedding聚合，得到最后连续特征值的Embedding，即聚合函数f(.)，使用Weighted-Average Aggregation：ej=sum_{h=1}^Hj(xj^hat^h*MEj^h)

        注意：论文中在Training中提到，需要在数据预处理环节nomarlize numerical feature到[0, 1]范围：xj<--(xj-xj_min)/(xj_max-xj_min)。TODO：在别的任务中也需要正则化吗？？？

        Embedding Analysis中，Macro-Analysis是通过t-SNE来可视化各种value对应的embedding，Micro-Analysis是观察各种value在meta-embedding上的分布(Softmax Score)
        

    **Article**: [KDD2021|华为联合上交提出CTR预估数值特征embedding学习框架AutoDis](https://blog.csdn.net/hestendelin/article/details/122314822)


## 23.3 Image Captioning

#### Paper

- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention - UToronto2016](https://arxiv.org/abs/1502.03044)

    **Code**: <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/eager/python/examples/generative_examples/image_captioning_with_attention.ipynb> (Tensorflow)


#### Practice

- <https://github.com/KoalaTree/models/tree/master/im2txt>

    Show and Tell: A Neural Image Caption Generator