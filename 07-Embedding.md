
# 7. Embedding

## 7.1 Overview

**Keywords**: Word2Vec  Wiki2Vec  GloVe  Ngram2Vec  Para2Vec  Doc2Vec StarSpace

#### Article

- [Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html)
  
  本文主要整理深度学习中一些前沿的Contextual Pretrained Model，以及这些开源模型使用场景、开源代码和一些公开数据集。

- [Embedding从入门到专家必读的十篇论文](https://zhuanlan.zhihu.com/p/58805184)
  
  包括Word2Vec基础、Word2Vec衍生及应用、Graph Embedding 共3部分。

- [万物皆Embedding，从经典的word2vec到深度学习基本操作item2vec - 王喆](https://zhuanlan.zhihu.com/p/53194407)
  
  Embedding是DL的基础，介绍了Word2Vec, Item2Vec

- [从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史 - 张俊林](https://zhuanlan.zhihu.com/p/49271699)

#### Practice

- [How deep learning can represent War and Peace as a vector](https://towardsdatascience.com/neural-network-embeddings-explained-4d028e6f0526)

    中文解读：[神经网络词嵌入：如何将《战争与和平》表示成一个向量？](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247485826&idx=2&sn=8b946e7401f239c819623b6447af8bbc)

- <https://github.com/liuhuanyong/Word2Vector>

    Self complemented word embedding methods using CBOW，skip-Gram，word2doc matrix , word2word matrix.


## 7.2 Word2Vec

#### Paper - OK

- Word2Vec - [Efficient Estimation of Word Representations in Vector Space - Google2013](https://arxiv.org/abs/1301.3781)

- Hierarchical Softmax & Negative Sampling & Subsampling of Frequent Words & Phrase Learning - [Distributed Representations of Words and Phrases and their Compositionality - Google2013](https://arxiv.org/abs/1310.4546)

- Negative Sampling & Machine Translation - [On Using Very Large Target Vocabulary for Neural Machine Translation-2015](https://arxiv.org/abs/1412.2007)

Yao: These are not easy to understand, and you'd better learn them by reading some other articles, such as the ones in 'Articles' below.

#### Code

- <https://github.com/tmikolov/word2vec/blob/master/word2vec.c> (C, original)

- <https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c> (C, with detailed comments)

- <https://github.com/danielfrg/word2vec> (Python)

#### Article - OK

- Skip-Gram: [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)

- Negative Sampling & Subsampling of Frequent Words & Phrase Learning: [Word2Vec Tutorial Part 2 - Negative Sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)

- [The Illustrated Word2vec](http://jalammar.github.io/illustrated-word2vec/)

#### Library

- Gensim: <https://radimrehurek.com/gensim/models/word2vec.html>

    ```from gensim.models import Word2Vec```

#### Practice - TODO

- [使用维基从头训练词嵌入](https://github.com/HoratioJSY/cn-words)

- How to Create a Simple Word2Vec Network and What Are the Input and Output?

- How to Use Word2Vec Tools to Train Your Own Embeddings? Such as gensim.models.word2vec and others. Both English and Chinese.

可以参考 <https://github.com/liuyaox/coding_awesome/blob/master/Gensim/gensim_demo.py> 中gensim的使用

For English

Wikipeida --> Word Embeddings; Wiki2Vec

For Chinese

People's Daily  --> Word Embeddings

缺点：

一个单词对应一个固定的向量，无法很好处理多义词。

#### Further - TODO

- [Interesting Job: Applying Word2Vec to Recommenders and Advertising](http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/)

- [WSABIE: Scaling Up To Large Vocabulary Image Annotation - Google2017](http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf)
  
  **要点**: 指定有限个label，给object打标！

  **Article**: [WSABIE 算法解释](https://heleifz.github.io/14696374110477.html)

  **要点**: 对待打标签的object(的向量)进行线性变换，同时借鉴Word2Vec的思想，把有限个label转换成Embedding，从而把object与label映射到同一向量空间，向量内积即可度量相似性。


## 7.3 GloVe

#### Paper

GloVe: [Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)

#### Code

- <https://github.com/stanfordnlp/GloVe>

#### Article

- [理解GloVe模型（Global vectors for word representation）](https://blog.csdn.net/coderTC/article/details/73864097)
  
  Yao：本质上是想让词向量(经一定的函数处理后)具有共现矩阵的性质，方法上使用了较多脑洞和技巧，有种强行凑出来的感觉，而非严谨的数学推导，不过对于想做研究的人来说，可以是一种有启发的借鉴，借鉴的不是想法，而是做法！

#### Practice

- How to Use Pretrained Word2Vec/GloVe Word Embeddings?

- To Do Simple Job

- To Create Embedding Layer for Neural Network


## 7.4 Character Embedding

#### Paper

- CWE - [Joint learning of character and word embeddings - THU2015](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/ijcai2015_character.pdf)

    中文解读：[Character and Word Embedding读书报告](https://zkq1314.github.io/2018/07/14/Character-and-Word-Embedding%E8%AF%BB%E4%B9%A6%E6%8A%A5%E5%91%8A/)

- [Joint Embeddings of Chinese Words, Characters, and Fine-grained Subcharacter Components - HKUST2017](https://www.aclweb.org/anthology/D17-1027)

#### Article

- [Character Level NLP](https://www.lighttag.io/blog/character-level-NLP/)

    中文解读：[字符级NLP优劣分析：在某些场景中比词向量更好用](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650759154&idx=4&sn=5b823a28c7755427fd0e7e9a1b95dd9f)



## 7.4 Ngram2Vec

Maybe it's useful for attr-attrval matching!?!

Phrase2Vec???

#### Paper

[Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics - RUC2017](http://www.aclweb.org/anthology/D17-1023)

#### Code

- <https://github.com/zhezhaoa/ngram2vec/>


## 7.5 Sentence2Vec

#### Paper

- Universal Sentence Encoder-2018
  
  **Paper**: <https://arxiv.org/abs/1803.11175>

  **Code**: <https://tfhub.dev/google/universal-sentence-encoder/2>

- Supervised Learning of Universal Sentence Representations from Natural Language Inference Data-2018
  
  **Paper**: <https://arxiv.org/abs/1705.02364>

  **Code**: <https://github.com/facebookresearch/InferSent>

- Shortcut-Stacked Sentence Encoders for Multi-Domain Inference-2017
  
  **Paper**: <https://arxiv.org/abs/1708.02312>

  **Code**: <https://github.com/easonnie/ResEncoder>


## 7.6 Doc2Vec & Paragraph2Vec

#### Paper

[Distributed Representations of Sentences and Documents - Google2014](https://arxiv.org/abs/1405.4053)

#### Library

- gensim: <https://radimrehurek.com/gensim/models/doc2vec.html>


## 7.6 StarSpace

#### Paper

[StarSpace: Embed All The Things!](https://arxiv.org/abs/1709.03856)

损失函数：相似Entity之间相似度较高

#### code

- <https://github.com/facebookresearch/StarSpace>

#### Article

- [StarSpace（embed all the things嵌入表示）编译和测试](https://blog.csdn.net/sparkexpert/article/details/78957607)


## 7.7 Item2Vec - TOTODO

#### Paper

[Item2Vec: Neural Item Embedding for Collaborative Filtering - Microsoft2016](https://arxiv.org/abs/1603.04259)

**Keywords**: Collaborative Filtering; Item Similarity; Recommender System; Neural Network Embedding

**要点**: 

#### Article

- [从用户行为去理解内容-item2vec及其应用](https://cloud.tencent.com/developer/article/1039868)
  
  讲述了item2vec或其理念在分类、推荐召回和语义召回上的应用，以及直接作为深度模型的输入特征。


## 7.8 Graph Embedding

### 7.8.1 Overview

#### Article

- [深度学习中不得不学的 Graph Embedding 方法](https://zhuanlan.zhihu.com/p/64200072)


### 7.8.2 Node2Vec - TOTODO

#### Paper

[node2vec: Scalable Feature Learning for Networks - Stanford2016](https://arxiv.org/abs/1607.00653)

node2vec is an algorithmic framework for learning continuous feature representations for nodes in networks. In node2vec, we learn a mapping of nodes to a low-dimensional space of features that maximizes the likelihood of preserving network neighborhoods of nodes.

node2vec主要用于处理网络结构中的多分类和链路预测任务，具体来说是对网络中的节点和边的特征向量表示方法。简单点来说就是将原有社交网络中的图结构，表达成特征向量矩阵，每一个node(人、物或内容等)表示成一个特征向量，用向量与向量之间的矩阵运算来得到相互的关系。

#### Code

- <http://snap.stanford.edu/node2vec/>

- <https://github.com/aditya-grover/node2vec>

#### Article

- [关于Node2vec算法中Graph Embedding同质性和结构性的进一步探讨](https://zhuanlan.zhihu.com/p/64756917)

- [node2vec: Embeddings for Graph Data](https://towardsdatascience.com/node2vec-embeddings-for-graph-data-32a866340fef)


## 7.9 Other Embeddings

Wiki2Vec: <https://github.com/idio/wiki2vec>


## 7.10 Embeddings Dimensionality Reduction

#### Paper

[Simple & Effective Dimensionality Reduction for Word Embeddings - Microsoft2017](https://arxiv.org/abs/1708.03629)

#### Code

- tools_nlp项目: <https://github.com/liuyaox/tools_nlp/blob/master/Preprocessing/embedding_reduction.py>

    Reference: <https://github.com/vyraun/Half-Size>