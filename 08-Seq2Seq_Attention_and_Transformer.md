

# 8. Seq2Seq & Attention & Transformer

## 8.1 Overview

Attention, Transformer, Pointer Network


## 8.2 Seq2Seq & Encoder2Decoder


#### Practice

- [Play couplet with seq2seq model. 用深度学习对对联](https://github.com/wb14123/seq2seq-couplet) (Tensorflow)


## 8.3 Attention

### 8.3.1 Overview

Attention的打分机制是关键，表示Encoder的状态a和Decoder的状态s之间的匹配程度，有多种，包括：**加性模型，点积模型，缩放点积模型，双线性模型**，如下图所示：

![](https://raw.githubusercontent.com/liuyaox/ImageHosting/master/for_markdown/attetion_scoring.jpg)

**YAO**: 

有时候在一些模型结构中可简化，比如没有s只有a，把a输入至一个小神经网络里得到a的权重，表示关于a的Attention，随后便可与a加权求和，结果可继续输入后续结构中。

Attention是一种理念和思想，核心要点在于：**通过小神经网络计算关于输入a的权重，即Attention**，从而后续结构在使用a时能够加权使用，有所侧重。

依据这一思想，Attention-based模型可以有很多种，可以很简单。

#### Paper

- [An Attentive Survey of Attention Models - LinkedIn2019](https://arxiv.org/abs/1904.02874)


#### Article

- 【Great】[Attention注意力机制超全综述 - 2019](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247486941&idx=2&sn=53d0a7b224cc8717047fb6eba6e1c368)

    *YAO**: 6种打分机制, Attention发展历程(Attention in Seq2Seq, SoftAttention, HardAttention, GlobalAttention, LocalAttention, Attention in Transformer), 实例分析, 机制实现分析

- [从各种注意力机制窥探深度学习在NLP中的神威](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247485751&idx=1&sn=4a76c7864f09b13764b0e9a6108a5a56)

- [目前主流的attention方法都有哪些？](https://www.zhihu.com/question/68482809)

- [遍地开花的 Attention ，你真的懂吗？ - 2019](https://mp.weixin.qq.com/s?__biz=MzIzOTU0NTQ0MA==&mid=2247491048&idx=1&sn=ceb1cd0fecad478a252b7681ed3231d4)


### 8.3.2 Attention

#### Paper

[Neural Machine Translation by Jointly Learning to Align and Translate - Germany2014](https://arxiv.org/abs/1409.0473v2)

**YAO**: Attention打分机制使用的是**加性模型**

#### Code

- <https://github.com/tensorflow/nmt> (Tensorflow)

- <https://github.com/brightmart/text_classification> (Tensorflow)

- <https://github.com/Choco31415/Attention_Network_With_Keras> (Keras)

    **YAO**: **加性模型**，与吴恩达课程练习5-3-1里的Attention实现方式差不多

- [基于Keras的attention实战](https://blog.csdn.net/jinyuan7708/article/details/81909549)

    **YAO**: 大道至简，2种简单的另类Attention。
    
    **YAO**: 好像说是：输入(或经简单处理如LSTM处理后)为inputs，inputs输入全连接层(小神经网络)，结果就是Attention，随后与inputs Merge在一起(Merge方式有很多)，再进行后续操作。这就是Attention-Based模型了！？！？


#### Article

- [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

- [Attention and Memory in Deep Learning and NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)

- [斯坦福 CS224n 课程对 Attention 机制的介绍 from 1:00:55](https://www.youtube.com/watch?v=XXtpJxZBa2c)


### 8.3.3 Hierarchical Attention Network (HAN)

#### Paper

HAN: [Hierarchical Attention Networks for Document Classification - CMU2016](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

Structure: Word Encoder(BiGRU) -> Word Attention -> Sentence Encoder(BiGRU) -> Sentence Attention -> Softmax

共有Word和Sentence这2种level的Encoder + Attention

Encoder: To get rich representation of word/sentence

Attention: To get important word/sentence among words/sentences

![hierarchical_attention_network_structure](./image/hierarchical_attention_network01.png)

巧妙之处：受Attention启发，这种结构不仅可以获得每个Sentence中哪些words较为重要，而且可获得每个document(很多sentence)中哪些sentences较为重要！It enables the model to capture important information in different levels.

#### Code

- <https://github.com/richliao/textClassifier> (Keras)

- <https://github.com/brightmart/text_classification> (Tensorflow)

- <https://github.com/indiejoseph/doc-han-att> (Tensorflow)


### 8.3.4 BahdanauAttention & LuongAttention

#### Paper

- LuongAttention: [Effective Approaches to Attention-based Neural Machine Translation - Stanford2015](https://arxiv.org/abs/1508.04025)

    **YAO**: 貌似使用得较多，TF里有现成的API. Attention打分机制提到了4种：**双线性模型general, 点积模型dot，最初的加性模型concat**以及location-based Attention时**只使用Target Hidden State**(而没有Source Hidden State)，待继续……

- BahdanauAttention: [Neural Machine Translation by Jointly Learning to Align and Translate - Germany2016](https://arxiv.org/abs/1409.0473)

    与LuongAttention长得略微有点不同，但是功能一样。

- Normed BahdanauAttention: [Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks - OpenAI2016](https://arxiv.org/abs/1602.07868)

    在BahdanauAttention类中有一个权重归一化的版本（normed_BahdanauAttention），它可以加快随机梯度下降的收敛速度。在使用时，将初始化函数中的参数normalize设为True即可。


### 8.3.5 单调 & 混合 Attention

#### Paper

[Online and Linear-Time Attention by Enforcing Monotonic Alignments - 2017](https://arxiv.org/abs/1704.00784)

    单调注意力机制(Monotonic Attention)，是在原有注意力机制上添加了一个单调约束。该单调约束的内容为：已经被关注过的输入序列，其前面的序列中不再被关注。


[Attention-Based Models for Speech Recognition - Poland2015](https://arxiv.org/abs/1506.07503)

    混合注意力机制很强大，比一般的注意力专注的地方更多，信息更丰富。因为混合注意力中含有位置信息，所以它可以在输入序列中选择下一个编码的位置。这样的机制更适用于输出序列大于输入序列的Seq2Seq任务，例如语音合成任务。


## 8.4 Transformer - TOTODO

### 8.4.1 Transformer

#### Paper

[Attention Is All You Need - Google2017](https://arxiv.org/abs/1706.03762)

#### Code

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) (PyTorch)

- <https://github.com/jadore801120/attention-is-all-you-need-pytorch> (PyTorch)

- <https://github.com/foamliu/Self-Attention-Keras> (Keras)

#### Library

- <https://github.com/CyberZHG/keras-transformer> (Keras)

- <https://github.com/CyberZHG/keras-self-attention> (Keras)

#### Article

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

- [Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
  
- [Google 发布的 attention 机制介绍官方视频](https://www.youtube.com/watch?v=rBCqOTEfxvg)

- [BERT大火却不懂Transformer？读这一篇就够了](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651666707&idx=1&sn=2e9149ccdba746eaec687038ce560349)

#### Pratice

- [放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)
  
- [使用一个简单的 Transformer 模型进行序列标注](https://medium.com/@kolloldas/building-the-mighty-transformer-for-sequence-tagging-in-pytorch-part-i-a1815655cd8)

- [一个写对联的 Transformer 序列到序列模型](https://github.com/andy-yangz/couplets_seq2seq_transformer) (Tensorflow)

    解读：[为了写春联，我用Transformer训练了一个“对穿肠”](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651667456&idx=1&sn=b2ffe9990f8bf8a242e52face2044b65)


### 8.4.2 Transformer-XL

#### Paper

[Transformer-XL: Attentive Language Models Beyond a Fixed Length Context - Google2019](https://arxiv.org/abs/1901.02860)

该模型对 Transformer 进行了改进，但这一改进没有被 BERT 采用

#### Code

- <https://github.com/CyberZHG/keras-transformer-xl> (Keras)

- <https://github.com/kimiyoung/transformer-xl/tree/master/pytorch> (PyTorch)

- <https://github.com/kimiyoung/transformer-xl/tree/master/tf> (Tensorflow)

#### Article

- [Transformer-XL — CombiningTransformers and RNNs Into a State-of-the-art Language Model](https://www.lyrn.ai/2019/01/16/transformer-xl-sota-language-model)
