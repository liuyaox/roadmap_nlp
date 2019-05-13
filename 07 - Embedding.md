
## 7. Embedding

### 7.1 Overview

**Keywords**: Word2Vec  Wiki2Vec  GloVe  Ngram2Vec  Para2Vec  Doc2Vec StarSpace

### 7.2 Word2Vec

#### Paper - OK

Word2Vec - Efficient Estimation of Word Representations in Vector Space-2013(<https://arxiv.org/abs/1301.3781>)

Hierarchical Softmax & Negative Sampling & Subsampling of Frequent Words & Phrase Learning - Distributed Representations of Words and Phrases and their Compositionality-2013(<https://arxiv.org/abs/1310.4546>)

#### Comment

Yao: These are not easy to understand, and you'd better learn them by reading some other articles, such as the ones in 'Articles' below.

#### Source

**Website**: 

**Github**: <https://github.com/danielfrg/word2vec>

#### Article - OK

Skip-Gram: <http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/>

Negative Sampling & Subsampling of Frequent Words & Phrase Learning: <http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/>

#### Tool/Library

Gensim: <https://radimrehurek.com/gensim/models/word2vec.html>

#### Practice - TODO

a. How to Create a Simple Word2Vec Network and What Are the Input and Output?

b. How to Use Word2Vec Tools to Train Your Own Embeddings? Such as gensim.models.word2vec and others. Both English and Chinese.

For English

Wikipeida --> Word Embeddings; Wiki2Vec

For Chinese

People's Daily  --> Word Embeddings

缺点：

一个单词对应一个固定的向量，无法很好处理多义词。

#### Further - TODO

Interesting Job: Applying Word2Vec to Recommenders and Advertising?(<http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/>)

### 7.3 GloVe

#### Paper

GloVe: Global Vectors for Word Representation (<https://nlp.stanford.edu/projects/glove/>)

#### Source

**Github**: <https://github.com/stanfordnlp/GloVe>

#### Article

a. (<https://blog.csdn.net/coderTC/article/details/73864097>)

Yao：本质上是想让词向量(经一定的函数处理后)具有共现矩阵的性质，方法上使用了较多脑洞和技巧，有种强行凑出来的感觉，而非严谨的数学推导，不过对于想做研究的人来说，可以是一种有启发的借鉴，借鉴的不是想法，而是做法！

#### Practice

How to Use Pretrained Word2Vec/GloVe Word Embeddings?

To Do Simple Jobs

To Create Embedding Layer for Neural Network

### 7.4 EMLO - TOTODO

#### Paper

Deep contextualized word representations - AllenAI2018 (<https://arxiv.org/abs/1802.05365>)

#### Article

b. ELMO模型（Deep contextualized word representation）(<https://www.cnblogs.com/jiangxinyang/p/10060887.html>)

#### Practice

b. 文本分类实战（九）—— ELMO 预训练模型 (<https://www.cnblogs.com/jiangxinyang/p/10235054.html>)

### 7.5 BERT - TOTODO

#### Paper

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding - Google2018 (<https://arxiv.org/abs/1810.04805>)

#### Source

**Github**: <https://github.com/google-research/bert> (Tensorflow)

#### Practice

a. <https://github.com/brightmart/text_classification/tree/master/a00_Bert> (Tensorflow)

b. 文本分类实战（十）—— BERT 预训练模型 (<https://www.cnblogs.com/jiangxinyang/p/10241243.html>)

GPT

### 7.6 Ngram2Vec
Maybe it's useful for attr-attrval matching!?!

Phrase2Vec???

#### Paper

Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics (<http://www.aclweb.org/anthology/D17-1023>)

#### Source

**Github**: <https://github.com/zhezhaoa/ngram2vec/>

### 7.7 Sentence2Vec

- Universal Sentence Encoder-2018 

**Paper**: <https://arxiv.org/abs/1803.11175>

**Source**: <https://tfhub.dev/google/universal-sentence-encoder/2>

- Supervised Learning of Universal Sentence Representations from Natural Language Inference Data-2018

**Paper**: <https://arxiv.org/abs/1705.02364>

**Source**: <https://github.com/facebookresearch/InferSent>

- Shortcut-Stacked Sentence Encoders for Multi-Domain Inference-2017

**Paper**: <https://arxiv.org/abs/1708.02312>

**Source**: <https://github.com/easonnie/ResEncoder>

### 7.8 Doc2Vec & Paragraph2Vec


### 7.9 StarSpace

#### Paper

StarSpace: Embed All The Things! (<https://arxiv.org/abs/1709.03856>)

损失函数：相似Entity之间相似度较高

#### Source

**Github**: <https://github.com/facebookresearch/StarSpace>


### 7.10 Node2Vec

#### Paper

node2vec: Scalable Feature Learning for Networks - Stanford2016 (<https://arxiv.org/abs/1607.00653>)

node2vec is an algorithmic framework for learning continuous feature representations for nodes in networks. In node2vec, we learn a mapping of nodes to a low-dimensional space of features that maximizes the likelihood of preserving network neighborhoods of nodes.

node2vec主要用于处理网络结构中的多分类和链路预测任务，具体来说是对网络中的节点和边的特征向量表示方法。简单点来说就是将原有社交网络中的图结构，表达成特征向量矩阵，每一个node(人、物或内容等)表示成一个特征向量，用向量与向量之间的矩阵运算来得到相互的关系。

#### Source

**Website**: <http://snap.stanford.edu/node2vec/>

**Github**: <https://github.com/aditya-grover/node2vec>


### 7.11 Others

Wiki2Vec: <https://github.com/idio/wiki2vec>


### 7.11 Summary