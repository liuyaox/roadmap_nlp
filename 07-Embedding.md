
# 7. Embedding

## 7.1 Overview

**Keywords**: Word2Vec  Wiki2Vec  GloVe  Ngram2Vec  Para2Vec  Doc2Vec StarSpace

#### Article

- [Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html)
  
  本文主要整理深度学习中一些前沿的Contextual Pretrained Model，以及这些开源模型使用场景、开源代码和一些公开数据集。

- [Embedding从入门到专家必读的十篇论文](https://zhuanlan.zhihu.com/p/58805184)
  
  包括Word2Vec基础、Word2Vec衍生及应用、Graph Embedding 共3部分。

- [万物皆Embedding，从经典的word2vec到深度学习基本操作item2vec](https://zhuanlan.zhihu.com/p/53194407)
  
  Embedding是DL的基础，介绍了Word2Vec, Item2Vec

- [Language Models and Contextualised Word Embeddings](http://www.davidsbatista.net/blog/2018/12/06/Word_Embeddings/)
  
  对 ELMo, BERT 及其他模型进行了一个简单的综述


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

Gensim: <https://radimrehurek.com/gensim/models/word2vec.html>

#### Practice - TODO

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

How to Use Pretrained Word2Vec/GloVe Word Embeddings?

To Do Simple Jobs

To Create Embedding Layer for Neural Network

## 7.4 EMLO - TOTODO

EMLO 是第一个使用预训练模型进行词嵌入的方法，将句子输入ELMO，可以得到句子中每个词的向量表示。

#### Paper

[Deep contextualized word representations - AllenAI2018](https://arxiv.org/abs/1802.05365)

#### Article

- [ELMO模型(Deep contextualized word representation)](https://www.cnblogs.com/jiangxinyang/p/10060887.html)

- [对 ELMo 的视频介绍](https://vimeo.com/277672840)

#### Practice

- [文本分类实战（九）—— ELMO 预训练模型](https://www.cnblogs.com/jiangxinyang/p/10235054.html)

## 7.5 BERT - TOTODO

#### Paper

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding - Google2019](https://arxiv.org/abs/1810.04805)

#### Code

- <https://github.com/google-research/bert> (Tensorflow)

#### Article

- 编码器: [Dissecting BERT Part 1: The Encoder](https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3)

- 网络结构: [Understanding BERT Part 2: BERT Specifics](https://medium.com/dissecting-bert/dissecting-bert-part2-335ff2ed9c73)

- 解码器: [Dissecting BERT Appendix: The Decoder](https://medium.com/dissecting-bert/dissecting-bert-appendix-the-decoder-3b86f66b0e5f)

- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/)

- [彻底搞懂BERT](https://www.cnblogs.com/rucwxb/p/10277217.html)

#### Practice

- [Pretrained PyTorch models for BERT, OpenAI GPT & GPT-2, Google/CMU Transformer-XL](https://github.com/huggingface/pytorch-pretrained-bert) (PyTorch)

- Good!!! [Implemention some Baseline Model upon Bert for Text Classification](https://github.com/songyingxin/bert-textclassification) (PyTorch)

- <https://github.com/brightmart/text_classification/tree/master/a00_Bert> (Tensorflow)

- [文本分类实战（十）—— BERT 预训练模型](https://www.cnblogs.com/jiangxinyang/p/10241243.html)

#### Further

- [哈工大讯飞联合实验室发布基于全词覆盖的中文BERT预训练模型](https://github.com/ymcui/Chinese-BERT-wwm)
  
  **Paper**: [Pre-Training with Whole Word Masking for Chinese BERT - HIT2019](https://arxiv.org/abs/1906.08101)

- [站在BERT肩膀上的NLP新秀们（PART I）](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247489437&idx=4&sn=d1d7ca7e3b4b0a1710252e8d52affe4d&chksm=ebb42f49dcc3a65ffbf86a6016db944a04911a17bd22979cfe17f2da52c2aa0d68833bf5eda8&mpshare=1&scene=1&srcid=&key=67a728f6339a0a1e70d20293822ea56da128bf66230228b7602acf0e31e1e1d7235379d978254cf7172f1639e89d6c2cd984a2205d92963f471a6c99238e89225aa2efc53febb55162ee78dd20023973&ascene=1&uin=MjcwMjE1Nzk1&devicetype=Windows+7&version=62060833&lang=en&pass_ticket=UYja1qL4FjsIjJwYJqT7vvLoCJho0%2Bf7%2FxcTCgiuCuAokcVpCfGb7MuLVdYj2QHK)
  
  给 BERT 模型增加外部知识信息，使其能更好地感知真实世界，主要讲了 ERNIE from Baidu 和 ERNIE from THU

- [站在BERT肩膀上的NLP新秀们（PART II）](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650409996&idx=1&sn=ddf837339e50001be4514fee743bfe9d&chksm=becd8a5689ba03405b3e11c882e376effc407b1ecde745a1df0295008329b5d83c4a6ab5fc1c&scene=21#wechat_redirect)
  
  主要讲了 XLMs from Facebook, LASER from Facebook, MASS from Microsoft 和 UNILM from Microsoft

- [站在BERT肩膀上的NLP新秀们（PART III）](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650410110&idx=1&sn=310f675cf0cc1e2a1f4cc7b919743bc4&chksm=becd8a2489ba033262daff98227a9887f5a80fba45590b75c9328567258f9420f98423af6e19&mpshare=1&scene=1&srcid=&key=67a728f6339a0a1ede767062b26eef6e4f4dc12c1a81bf2f24d459516816478c0c05b3e3a6625e2e074c356aecc75e63253a21de6491e01d23be22709b9e32605da5ae8c545820a5d0c02a0428e7ed3a&ascene=1&uin=MjcwMjE1Nzk1&devicetype=Windows+7&version=62060833&lang=en&pass_ticket=SWBCv%2Bah0eIEISXOXsPuddmJM8%2Bvbzjxrwkg2kH2Il116bWpQYmtXQht1D9khSa%2B)
  
  主要看看预训练模型中的增强训练（多任务学习/数据增强）以及BERT多模态应用： MT-DNN from Microsoft, MT-DNN-2 from Microsoft, GPT-2 from OpenAI 和 VideoBERT from Google

## 7.6 GPT

#### Paper

- GPT1: [Improving Language Understanding by Generative Pre-Training - OpenAI2018](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

- GPT2: [Language Models are Unsupervised Multitask Learners - OpenAI2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

#### Code

- GPT1: <https://github.com/openai/finetune-transformer-lm> (Tensorflow)

- GPT2: <https://github.com/openai/gpt-2> (Tensorflow)

#### Article

- GPT1: [Improving Language Understanding with Unsupervised Learning](https://openai.com/blog/language-unsupervised/)

- GPT2: [Better Language Models and Their Implications](https://www.openai.com/blog/better-language-models/)

## 7.6 Ngram2Vec

Maybe it's useful for attr-attrval matching!?!

Phrase2Vec???

#### Paper

[Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics](http://www.aclweb.org/anthology/D17-1023)

#### Code

- <https://github.com/zhezhaoa/ngram2vec/>

## 7.7 Sentence2Vec

- Universal Sentence Encoder-2018
  
  **Paper**: <https://arxiv.org/abs/1803.11175>

  **Code**: <https://tfhub.dev/google/universal-sentence-encoder/2>

- Supervised Learning of Universal Sentence Representations from Natural Language Inference Data-2018
  
  **Paper**: <https://arxiv.org/abs/1705.02364>

  **Code**: <https://github.com/facebookresearch/InferSent>

- Shortcut-Stacked Sentence Encoders for Multi-Domain Inference-2017
  
  **Paper**: <https://arxiv.org/abs/1708.02312>

  **Code**: <https://github.com/easonnie/ResEncoder>

## 7.8 Doc2Vec & Paragraph2Vec


## 7.9 StarSpace

#### Paper

[StarSpace: Embed All The Things!](https://arxiv.org/abs/1709.03856)

损失函数：相似Entity之间相似度较高

#### code

- <https://github.com/facebookresearch/StarSpace>


## 7.10 Node2Vec - TOTODO

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

## 7.11 Item2Vec - TOTODO

#### Paper

[Item2Vec: Neural Item Embedding for Collaborative Filtering - Microsoft2016](https://arxiv.org/abs/1603.04259)

**Keywords**: Collaborative Filtering; Item Similarity; Recommender System; Neural Network Embedding

**要点**: 

#### Article

- [从用户行为去理解内容-item2vec及其应用](https://cloud.tencent.com/developer/article/1039868)
  
  讲述了item2vec或其理念在分类、推荐召回和语义召回上的应用，以及直接作为深度模型的输入特征。


## 7.12 Graph Embedding

#### Article

- [深度学习中不得不学的 Graph Embedding 方法](https://zhuanlan.zhihu.com/p/64200072)



## 7.13 Others

Wiki2Vec: <https://github.com/idio/wiki2vec>


## 7.14 Summary
