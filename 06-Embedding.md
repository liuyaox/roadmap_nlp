# 6. Embedding

## 6.1 Overview

**Keywords**: Word2Vec  Wiki2Vec  GloVe  Ngram2Vec  Para2Vec  Doc2Vec StarSpace

#### Paper

- [On the Dimensionality of Word Embedding - 2018](https://arxiv.org/abs/1812.04224)

    针对Word2Vec(Skip-Gram), GloVe, LSA，找到训练集词向量的最佳维度embed_dim

    **Code**: <https://github.com/ziyin-dl/word-embedding-dimensionality-selection>

#### Article

- [Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html)
  
  本文主要整理深度学习中一些前沿的Contextual Pretrained Model，以及这些开源模型使用场景、开源代码和一些公开数据集。

- [Embedding从入门到专家必读的十篇论文](https://zhuanlan.zhihu.com/p/58805184)
  
  包括Word2Vec基础、Word2Vec衍生及应用、Graph Embedding 共3部分。

- [万物皆Embedding，从经典的word2vec到深度学习基本操作item2vec - 王喆](https://zhuanlan.zhihu.com/p/53194407)
  
  Embedding是DL的基础，介绍了Word2Vec, Item2Vec

- [如何理解在各类NLP或CTR预估模型中，将embedding求平均这一做法的物理意义？](https://www.zhihu.com/question/332347498)

- [神奇的Embedding - 2019](https://zhuanlan.zhihu.com/p/53058456)


#### Practice

- [How deep learning can represent War and Peace as a vector](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526)

    **Chinese**：[神经网络词嵌入：如何将《战争与和平》表示成一个向量？](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247485826&idx=2&sn=8b946e7401f239c819623b6447af8bbc)

- <https://github.com/liuhuanyong/Word2Vector>

    Self complemented word embedding methods using CBOW，skip-Gram，word2doc matrix , word2word matrix.


## 6.2 Word2Vec

#### Paper

- Word2Vec - [Efficient Estimation of Word Representations in Vector Space - Google2013](https://arxiv.org/abs/1301.3781)

- Hierarchical Softmax & Negative Sampling & Subsampling of Frequent Words & Phrase Learning - [Distributed Representations of Words and Phrases and their Compositionality - Google2013](https://arxiv.org/abs/1310.4546)

- Negative Sampling & Machine Translation - [On Using Very Large Target Vocabulary for Neural Machine Translation-2015](https://arxiv.org/abs/1412.2007)

Yao: These are not easy to understand, and you'd better learn them by reading some other articles, such as the ones in 'Articles' below.

#### Code

- <https://github.com/tmikolov/word2vec/blob/master/word2vec.c> (C, original)

- <https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c> (C, with detailed comments)

- <https://github.com/jdeng/word2vec> (C++)

- <http://deeplearning4j.org/word2vec> (Java)

- <https://github.com/danielfrg/word2vec> (Python)

- <https://github.com/bamtercelboo/pytorch_word2vec> (PyTorch)

- <https://github.com/lonePatient/chinese-word2vec-pytorch> (PyTorch)

- <https://github.com/bojone/tf_word2vec/blob/master/word2vec_keras.py> (Keras)

  **Article**: [Keras版的Word2Vec - 2017](https://kexue.fm/archives/4515)

#### Library

- Gensim: <https://radimrehurek.com/gensim/models/word2vec.html>

#### Article

- [Word2Vec Tutorial - The Skip-Gram Model - 2016](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

- [Word2Vec Tutorial Part 2 - Negative Sampling - 2017](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)

    Negative Sampling & Subsampling of Frequent Words & Phrase Learning

- 【Great】[The Illustrated Word2vec - 2019](http://jalammar.github.io/illustrated-word2vec/)

    **Chinese**: [图解Word2vec，读这一篇就够了](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651669277&idx=2&sn=bc8f0590f9e340c1f1359982726c5a30)

    **YAO**: 

    假设文本是[x1,x2,x3,x4,x5,x6]

    三层结构：输入层-->隐层-->输出层，输入层和输出层维度都是|V|，输入层至隐层的权重矩阵W1，隐层至输出层的权重矩阵W2，前者即为待计算的Embedding

    CBOW: 两边预测中间，对context的处理是经W1处理后直接相加求平均得到一个向量，再经W2处理，输入Softmax处理后输出

    Skip-Gram: 中间预测两边，比如<x2, x1>,<x2, x3>都是训练样本，对word直接经W1处理后就是一个向量，随后同CBOW

    SGNS: Skip-Gram + Negative Sampling，上文训练样本变成<x2, x1, 1>和<x2, x3, 1>，label=1表示是相邻，label=0表示不是相邻。目前问题是只是正例，没有负例，于是需要负采样。即x2不变，从Vocabulary中随机选择若干单词如x7,x8与x2组成负例<x2, x7, 0>和<x2, x8, 0>。以上，**把Softmax问题转化为Sigmoid问题**，训练速度大大提高！这里有2个超参数，窗口大小和负样本数量。

    - 窗口大小：较小时，表示2个word更可以互换，较大时，表示更相关，Gensim默认是5

    - 负样本数量：论文认为5-20个比较理想，Gensim默认是5

    Train: 随机化Embedding和Context两个矩阵，shape都是<vocab_size, embed_dim>，前者用于x2找到对应的vector1，后者用于x1,x3,x7,x8这些找到对应的vector2，分别计算vector1与各个vector2的Dot-Product，以表示x2与xi的相似程度，再用sigmoid转化为概率，与真实Label计算Loss，随后训练更新Embedding和Context。训练结束，Embedding即为所需的Word Embedding

    Hierarchy Softmax: 根据词频构建的一个Huffman二叉树，每个非叶子节点是个二分类器，每个叶子节点对应V中的一个单词，词频高的单词所在叶子节点离root较近。Softmax的求和会被一系列的二分类器替代，最终输出是相关路径上所有二分类器输出的乘积，在梯度更新时，只需更新相关路径上的二分类器的参数即可。现在很少采取了，主要使用NS

- [word2vec原理推导与代码分析](http://www.hankcs.com/nlp/word2vec.html)

- [word2vec中的数学原理详解](https://www.cnblogs.com/peghoty/p/3857839.html)

#### Practice - TODO

- [使用维基从头训练词嵌入](https://github.com/HoratioJSY/cn-words)

- [中英文维基百科语料上的Word2Vec实验](http://www.52nlp.cn/%E4%B8%AD%E8%8B%B1%E6%96%87%E7%BB%B4%E5%9F%BA%E7%99%BE%E7%A7%91%E8%AF%AD%E6%96%99%E4%B8%8A%E7%9A%84word2vec%E5%AE%9E%E9%AA%8C)

- <https://github.com/AimeeLee77/wiki_zh_word2vec>

    利用Python构建Wiki中文语料词向量模型试验

- [Word2vec Tutorial](https://rare-technologies.com/word2vec-tutorial/)

- 参考<https://github.com/liuyaox/coding_awesome/blob/master/Gensim/gensim_demo.py>中gensim的使用

#### Further - TODO

- [Applying Word2Vec to Recommenders and Advertising - 2018](http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/)

- [Using Word2Vec for Music Recommendations - 2017](https://towardsdatascience.com/using-word2vec-for-music-recommendations-bb9649ac2484)

- [WSABIE: Scaling Up To Large Vocabulary Image Annotation - Google2017](http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf)
  
  **Key Points**: 指定有限个label，给object打标！

  **Article**: [WSABIE 算法解释](https://heleifz.github.io/14696374110477.html)

  **Key Points**: 对待打标签的object(的向量)进行线性变换，同时借鉴Word2Vec的思想，把有限个label转换成Embedding，从而把object与label映射到同一向量空间，向量内积即可度量相似性。


## 6.3 GloVe

GloVe: [Global Vectors for Word Representation - Stanford2014](https://nlp.stanford.edu/projects/glove/)

#### Code

- <https://github.com/stanfordnlp/GloVe>

#### Article

- [理解GloVe模型（Global vectors for word representation）](https://blog.csdn.net/coderTC/article/details/73864097)
  
  **Yao**：本质上是想让词向量(经一定的函数处理后)具有共现矩阵的性质，方法上使用了较多脑洞和技巧，有种强行凑出来的感觉，而非严谨的数学推导，不过对于想做研究的人来说，可以是一种有启发的借鉴，借鉴的不是想法，而是做法！

- [理解GloVe模型（+总结）](https://blog.csdn.net/u014665013/article/details/79642083)

- [为什么很多NLP的工作在使用预训练词向量时选择GloVe而不是Word2Vec或其他](https://www.zhihu.com/question/339184168)

#### Practice

- To Use Pretrained Word2Vec/GloVe Word Embeddings

- To Create Embedding Layer for Neural Network


## 6.4 Character Embedding

#### Paper

- CWE - [Joint learning of character and word embeddings - THU2015](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/ijcai2015_character.pdf)

    **Chinese**：[Character and Word Embedding读书报告](https://zkq1314.github.io/2018/07/14/Character-and-Word-Embedding%E8%AF%BB%E4%B9%A6%E6%8A%A5%E5%91%8A/)

- [Joint Embeddings of Chinese Words, Characters, and Fine-grained Subcharacter Components - HKUST2017](https://www.aclweb.org/anthology/D17-1027)

#### Article

- [Character Level NLP](https://www.lighttag.io/blog/character-level-NLP/)

    **Chinese**：[字符级NLP优劣分析：在某些场景中比词向量更好用](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650759154&idx=4&sn=5b823a28c7755427fd0e7e9a1b95dd9f)


## 6.5 Ngram2Vec

Maybe it's useful for attr-attrval matching!?!

Phrase2Vec???

[Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics - RUC2017](http://www.aclweb.org/anthology/D17-1023)

#### Code

- <https://github.com/zhezhaoa/ngram2vec/>


## 6.6 Sentence2Vec

#### Paper

- [Skip-Thought Vectors - Toronto2015](https://arxiv.org/abs/1506.06726)

    无监督模型

    **Article**: [My thoughts on Skip-Thoughts](https://medium.com/@sanyamagarwal/my-thoughts-on-skip-thoughts-a3e773605efa)

- [Universal Sentence Encoder - 2018](https://arxiv.org/abs/1803.11175)

  **Code**: <https://tfhub.dev/google/universal-sentence-encoder/2>

- [Supervised Learning of Universal Sentence Representations from Natural Language Inference Data - 2018](https://arxiv.org/abs/1705.02364)

  **Code**: <https://github.com/facebookresearch/InferSent>

- [Shortcut-Stacked Sentence Encoders for Multi-Domain Inference - 2017](https://arxiv.org/abs/1708.02312)

  **Code**: <https://github.com/easonnie/ResEncoder>


#### Practice

- [Unsupervised Sentence Representation with Deep Learning](https://blog.myyellowroad.com/unsupervised-sentence-representation-with-deep-learning-104b90079a93)

    介绍对比了3种表征句子的无监督深度学习方法：自编码器，语言模型和Skip-Thought向量模型，并与基线模型Average Word2Vec进行对比。

    **Chinese**: [简述表征句子的3种无监督深度学习方法 - 2018](http://www.sohu.com/a/229225932_164987)


## 6.7 Doc2Vec & Paragraph2Vec

- Doc2Vec: [Distributed Representations of Sentences and Documents - Google2014](https://arxiv.org/abs/1405.4053)

    和 Word2Vec 一样，该模型也存在两种方法：Distributed Memory(DM) 和 Distributed Bag of Words(DBOW)。DM 试图在给定上下文和段落向量的情况下预测单词的概率。在一个句子或者文档的训练过程中，段落 ID 保持不变，共享着同一个段落向量。DBOW 则在仅给定段落向量的情况下预测段落中一组随机单词的概率。 

#### Code

- Doc2Vec: <https://github.com/klb3713/sentence2vec>


#### Library

- Doc2Vec: gensim - <https://radimrehurek.com/gensim/models/doc2vec.html>


## 6.8 StarSpace

[StarSpace: Embed All The Things!](https://arxiv.org/abs/1709.03856)

损失函数：相似Entity之间相似度较高

#### code

- <https://github.com/facebookresearch/StarSpace>

#### Article

- [StarSpace（embed all the things嵌入表示）编译和测试](https://blog.csdn.net/sparkexpert/article/details/78957607)


## 6.9 Item2Vec - TOTODO

[Item2Vec: Neural Item Embedding for Collaborative Filtering - Microsoft2016](https://arxiv.org/abs/1603.04259)

**Keywords**: Collaborative Filtering; Item Similarity; Recommender System; Neural Network Embedding

#### Article

- [从用户行为去理解内容-item2vec及其应用](https://cloud.tencent.com/developer/article/1039868)
  
  讲述了item2vec或其理念在分类、推荐召回和语义召回上的应用，以及直接作为深度模型的输入特征。




## 6.11 Others

### 6.11.1 Wiki2Vec

**Code**: <https://github.com/idio/wiki2vec>


### 6.11.2 Tweet2Vec

一些社交文本中的语言结构跟书面语大不相同，作者别出心裁的特意做了一个基于字符组合的模型，其可以基于整个微博环境下复杂、非正常语言的字符串中学习到一种向量化的表达方式。

[Tweet2Vec: Character-Based Distributed Representations for Social Media - CMU2016](https://arxiv.org/abs/1605.03481)

**Code**: <https://github.com/bdhingra/tweet2vec>


### 6.11.3 Illustration-2vec

**Code**: <https://github.com/rezoo/illustration2vec>


### 6.11.4 cw2Vec

基于笔画的中文词向量算法

[cw2vec: Learning Chinese Word Embeddings with Stroke n-gram Information - Alibaba2018](https://raw.githubusercontent.com/ShelsonCao/cw2vec/master/cw2vec.pdf)

**Code**: <https://github.com/bamtercelboo/cw2vec> (C++)

**Article**: [蚂蚁金服公开最新基于笔画的中文词向量算法](https://www.sohu.com/a/217456047_99940985)


### 6.11.5 Lda2Vec

[Mixing Dirichlet Topic Models and Word Embeddings to Make lda2vec - 2016](https://arxiv.org/abs/1605.02019)

**Code**: <https://github.com/cemoody/lda2vec>

**Article**: <http://www.slideshare.net/ChristopherMoody3/word2vec-lda-and-introducing-a-new-hybrid-algorithm-lda2vec-57135994>


### 6.11.6 TopicVec

[Generative Topic Embedding: a Continuous Representation of Documents - Singapore2016](https://arxiv.org/abs/1606.02979)

**Code**: <https://github.com/askerlee/topicvec>


### 6.11.7 Entity2Vec

[Fast and space-efficient entity linking in queries - Yahoo201](https://www.dc.fi.udc.es/~roi/publications/wsdm2015.pdf)

**Code**: <https://github.com/ot/entity2vec>


### 6.11.8 Str2Vec

**code**: <https://github.com/pengli09/str2vec>


### 6.11.9 Author2Vec

[Author2Vec: Learning Author Representations by Combining Content and Link Information - Microsoft2016](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/06/jawahar16_www-2.pdf)


### 6.11.10 Playlist2Vec

**Code**: <https://github.com/mattdennewitz/playlist-to-vec>


### 6.11.11 Sense2Vec

[sense2vec - A Fast and Accurate Method for Word Sense Disambiguation In Neural Word Embeddings - 2015](https://arxiv.org/abs/1511.06388)


### 6.11.12 Medical2Vec

[Multi-layer Representation Learning for Medical Concepts - Georgia2016](https://arxiv.org/abs/1602.05568)

**Code**: <https://github.com/mp2893/med2vec>

**Code**: <https://github.com/ai-ku/wvec>


### 6.11.13 Game2Vec

**Code**: <https://github.com/warchildmd/game2vec>


### 6.11.14 Paper2Vec

[Paper2vec: Citation-Context Based Document Distributed Representation for Scholar Recommendation - SYSU2017](https://arxiv.org/abs/1703.06587)


## 6.12 Embeddings Dimensionality Reduction

[Simple & Effective Dimensionality Reduction for Word Embeddings - Microsoft2017](https://arxiv.org/abs/1708.03629)

#### Code

- tools_nlp项目: <https://github.com/liuyaox/tools_nlp/blob/master/Preprocessing/embedding_reduction.py>

    Reference: <https://github.com/vyraun/Half-Size>