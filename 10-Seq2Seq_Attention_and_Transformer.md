

# 10. Seq2Seq & Attention & Transformer

## 10.1 Overview

Attention, Transformer, Pointer Network


## 10.2 Seq2Seq & Encoder2Decoder


#### Practice

- [Play couplet with seq2seq model. 用深度学习对对联](https://github.com/wb14123/seq2seq-couplet) (Tensorflow)


## 10.3 Attention

### 10.3.1 Attention

#### Paper

[Neural Machine Translation by Jointly Learning to Align and Translate-2014](https://arxiv.org/abs/1409.0473v2)

#### Code

- <https://github.com/brightmart/text_classification> (Tensorflow)

#### Article

- [Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention)](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

- [Attention and Memoryin Deep Learning and NLP](http://www.wildml.com/2016/01/attention-and-memory-in-deep-learning-and-nlp/)

- [斯坦福 CS224n 课程对 attention 机制的介绍 from 1:00:55](https://www.youtube.com/watch?v=XXtpJxZBa2c)

- [从各种注意力机制窥探深度学习在NLP中的神威](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247485751&idx=1&sn=4a76c7864f09b13764b0e9a6108a5a56&chksm=eb501da4dc2794b2a7fa2b052c43e917e8e65467605b20b0cc4a885c60fcf526c4c88635fca1&mpshare=1&scene=1&srcid=01202KvMl6kcXRI6OaMFaLfg#rd)


### 10.3.2 Hierarchical Attention Network

#### Paper

[Hierarchical Attention Networks for Document Classification](https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

Structure: Word Encoder(BiGRU) -> Word Attention -> Sentence Encoder(BiGRU) -> Sentence Attention -> Softmax

共有Word和Sentence这2种level的Encoder + Attention

Encoder: To get rich representation of word/sentence

Attention: To get important word/sentence among words/sentences

![hierarchical_attention_network_structure](./image/hierarchical_attention_network01.png)

巧妙之处：受Attention启发，这种结构不仅可以获得每个Sentence中哪些words较为重要，而且可获得每个document(很多sentence)中哪些sentences较为重要！It enables the model to capture important information in different levels.

#### Code

- <https://github.com/brightmart/text_classification> (Tensorflow)


## 10.4 Transformer - TOTODO

### 10.4.1 Transformer

#### Paper

[Attention Is All You Need - Google2017](https://arxiv.org/abs/1706.03762)

#### Code

- <https://github.com/brightmart/text_classification> (Tensorflow)

- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) (PyTorch)

#### Article

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)

- [Transformer: A Novel Neural Network Architecture for Language Understanding](https://ai.googleblog.com/2017/08/transformer-novel-neural-network.html)
  
- [Google 发布的 attention 机制介绍官方视频](https://www.youtube.com/watch?v=rBCqOTEfxvg)

- [BERT大火却不懂Transformer？读这一篇就够了](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651666707&idx=1&sn=2e9149ccdba746eaec687038ce560349&chksm=bd4c1e808a3b97968a15cb3d21032b5394461a1be275476e4fd26563aa28d99be0b798ccee17&token=1915554219&lang=zh_CN&scene=21#wechat_redirect)

#### Pratice

- [放弃幻想，全面拥抱Transformer：自然语言处理三大特征抽取器（CNN/RNN/TF）比较](https://zhuanlan.zhihu.com/p/54743941)
  
- [使用一个简单的 Transformer 模型进行序列标注](https://medium.com/@kolloldas/building-the-mighty-transformer-for-sequence-tagging-in-pytorch-part-i-a1815655cd8)

- [一个写对联的 Transformer 序列到序列模型](https://github.com/andy-yangz/couplets_seq2seq_transformer) (Tensorflow)

    解读：[为了写春联，我用Transformer训练了一个“对穿肠”](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651667456&idx=1&sn=b2ffe9990f8bf8a242e52face2044b65&chksm=bd4c1b938a3b9285a94c75b1867febba6411f01c4b5954af3e8848d36ecbf84ebfef068c825f&mpshare=1&scene=1&srcid=#rd)


### 10.4.2 Transformer-XL

#### Paper

[Transformer-XL: Attentive Language Models Beyond a Fixed Length Context - Google2019](https://arxiv.org/abs/1901.02860)

该模型对 Transformer 进行了改进，但这一改进没有被 BERT 采用

#### Code

- <https://github.com/kimiyoung/transformer-xl/tree/master/pytorch> (PyTorch)

- <https://github.com/kimiyoung/transformer-xl/tree/master/tf> (Tensorflow)

#### Article

- [Transformer-XL — CombiningTransformers and RNNs Into a State-of-the-art Language Model](https://www.lyrn.ai/2019/01/16/transformer-xl-sota-language-model)


## 10.6 Summary
