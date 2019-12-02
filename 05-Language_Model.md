# 5. Language Model

## 5.1 Overview

**Keywords**: BOW  NNLM  (Tools: Onehot TFIDF Ngram)

Statistical Language Models

#### Code

- <https://github.com/codekansas/keras-language-modeling> (Keras)

    Some language modeling tools for Keras


## 5.2 NGram Model


#### Article

- [自然语言处理中的N-Gram模型详解](https://blog.csdn.net/baimafujinji/article/details/51281816)


## 5.3 Bag-of-Word (BOW)

BOW, BOW + NGram, BOW + TFIDF, BOW + NGram + TFIDF, 其实还可以再推进一步：BOW + NGram + TFIDF + SVD


## 5.4 NNLM

Purpose: To predict N-th word using previous N-1 words

2003年提出神经网络语言模型NNLM，模型结构很简单：

**Input(1~N-1 words, onehot)-->Embedding(Lookup Table)-->Concate-->Dense(tanh)-->Softmax(N-th word)**

Word2Vec与NNLM基本类似，结构非常相似，Word2Vec中的CBOW和Skip-gram结构更简化一些，删除了隐层Dense(tanh)，

因为是语言模型，而语言模型本质上就是看到上文预测下文，Word Embedding只是它的副产品，它的训练方式是：上文 --> 下一个word

而Word2Vec是专门为了Word Embedding而生的，更加灵活一些，有CBOW和Skip-gram这2种训练方式，CBOW: 上下文 --> 中间的word，Skip-gram：中间的word --> 上下文


## 5.5 Character-Level Model

[Character-Aware Neural Language Models-2015](https://arxiv.org/abs/1508.06615)

#### Code

- <https://github.com/chiragjn/deep-char-cnn-lstm> (Keras)

#### Article

- [Character Level NLP - 2019](https://www.lighttag.io/blog/character-level-NLP/)

    字符级NLP优劣分析：在某些场景中比词向量更好用
