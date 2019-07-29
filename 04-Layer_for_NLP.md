

# 3. Layer for NLP

## 3.1 Overview

#### 汇总对比

| Unit | Function |
| :-: | :-: |
| LSTM/GRU | 解决依赖问题，适合第一层建模，缺点是慢 |
| CNN | 配合池化抓取关键词特征 |
| Capsules | 可以替代CNN，效果一般优于CNN |
| MaxPooling | 只要关键特征，其他全部过滤 |
| Attention | 突出关键特征，但难以过滤掉所有非重点特征，类似于池化机制 |

由此，可得出以下若干小结论：

- 对于短文本，LSTM/GRU+Capsules是一个不错的模型

- 对于长文本，只使用CNN几乎没法用，最好在前面加一层LSTM/GRU，以及Attention机制

以上参考文章 [如何到top5%？NLP文本分类和情感分析竞赛总结](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247486159&idx=1&sn=522345e275df807942c7b56b0054fec9)


## 3.2 Activation

#### Paper

- [Comparing Deep Learning Activation Functions Across NLP tasks - 2019](https://arxiv.org/abs/1901.02671)

    **Github**: <https://github.com/UKPLab/emnlp2018-activation-functions>

    **Article**: [21种NLP任务激活函数大比拼：你一定猜不到谁赢了](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650756158&idx=2&sn=90cb49c49be078e7406539eb93561c9e)


#### Article

- [聊一聊深度学习的activation function - 2017](https://zhuanlan.zhihu.com/p/25110450)

- [从ReLU到Sinc，26种神经网络激活函数可视化 - 2017](http://www.dataguru.cn/article-12255-1.html)

- [ReLu(Rectified Linear Units)激活函数 - 2015](https://www.cnblogs.com/neopenx/p/4453161.html)

- [Softmax](http://neuralnetworksanddeeplearning.com/chap3.html#softmax)


## 3.2 CNN


#### Tool

- [3D Visualization of a CNN](http://scs.ryerson.ca/~aharley/vis/conv/)


#### Article

- [YJango的卷积神经网络——介绍 - 2019](https://zhuanlan.zhihu.com/p/27642620)

- [Convolutional Neural Networks backpropagation: from intuition to derivation - 2016](https://grzegorzgwardys.wordpress.com/2016/04/22/8/)

- [机器学习原来这么有趣！第三章:图像识别【鸟or飞机】？深度学习与卷积神经网络 - 2017](https://zhuanlan.zhihu.com/p/24524583)

- [CNN详解——反向传播过程](https://blog.csdn.net/HappyRocking/article/details/80512587)


## 3.3 RNN


## 3.4 LSTM/GRU

### 3.4.1 LSTM

#### Paper

[Long short-term memory](https://www.bioinf.jku.at/publications/older/2604.pdf)

#### Article

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

- [Deep Learning for NLP Best Practices blog](http://ruder.io/deep-learning-nlp-best-practices/)

#### Practice

- [如何准备用于LSTM模型的数据并进行序列预测？（附代码）](https://www.jiqizhixin.com/articles/2018-12-18-16)


### 3.4.2 AWD-LSTM

AWD-LSTM 对 LSTM 模型进行了改进，包括在隐藏层间加入 dropout ，加入词嵌入 dropout ，权重绑定等。建议**使用 AWD-LSTM 来替代 LSTM 以获得更好的模型效果**。

#### Paper

[Regularizing and Optimizing LSTM Language Models - 2017](https://arxiv.org/abs/1708.02182)

#### Code

- [Github by Salesforce](https://github.com/salesforce/awd-lstm-lm)
  
- [Github by fastAI](https://github.com/fastai/fastai/blob/master/fastai/text/models/awd_lstm.py)


### 3.4.3 Adversarial LSTM

#### Paper

[Adversarial Training Methods for Semi-Supervised Text Classification - 2016](https://arxiv.org/abs/1605.07725)

核心思想：通过对 word Embedding 上添加噪音生成对抗样本，将对抗样本以和原始样本 同样的形式喂给模型，得到一个 Adversarial Loss，通过和原始样本的 loss 相加得到新的损失，通过优化该新 的损失来训练模型，作者认为这种方法能对 word embedding 加上正则化，避免过拟合。

#### Practice

- [文本分类实战（七）—— Adversarial LSTM 模型](https://www.cnblogs.com/jiangxinyang/p/10208363.html)


### 3.4.4 GRU


## 3.5 Other RNN

- QRNN: [Quasi-Recurrent Neural Networks - Salesforce2016](https://arxiv.org/abs/1611.01576)

- SRU: [Simple Recurrent Units for Highly Parallelizable Recurrence - ASAPP2018](https://arxiv.org/abs/1709.02755)

    SRU单元在本质上与QRNN单元很像。从网络构建上看，SRU单元有点像QRNN单元中的一个特例，但是又比QRNN单元多了一个直连的设计。

- IndRNN: [Independently Recurrent Neural Network (IndRNN): Building A Longer and Deeper RNN - Australia2018](https://arxiv.org/abs/1803.04831)

    将IndRNN单元配合ReLu等非饱和激活函数一起使用，会使模型表现出更好的鲁棒性。

- JANET: [The unreasonable effectiveness of the forget gate - Cambridge2018](https://arxiv.org/abs/1804.04849)
