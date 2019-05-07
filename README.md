

# Part I Basic and Overview
## 1. Python + ML Basic
Material:

This part is not my keypoint.

## 2. Neural Network and Deep Learning - Andrew Ng - Coursera
Website: 

Language: English, Chinese, ...

Homework: Jupyter on Coursera

Github: None

## 3. Natural Language Processing - HSE - Coursera
Website: 

Language: English

Homework: Jupyter Locally or Using Google Colab

Github: https://github.com/hse-aml/natural-language-processing

Opinions: 

## 4. Corpus and Data
大规模中文自然语言处理语料Large Scale Chinese Corpus for NLP: https://github.com/brightmart/nlp_chinese_corpus

上万中文姓名及性别：https://pan.baidu.com/s/1hsHTEU4

好的资料：

https://github.com/thunlp


# Part II Basic in Details

## 5. Overview

**要点总结：**

a. 通过计算两个vector的Similarity来获得概率分布：To get possibility distribution by computing 'similarity' of query and hidden state.  该概率分布要么直接输出作为结果，要么用于加权求和。

## 6. Concepts & Methods
### 6.1 Overview

### 6.2 CNN

### 6.3 RNN

### 6.4 LSTM/GRU

### 6.5 Adversarial LSTM
**Paper**

Adversarial Training Methods for Semi-Supervised Text Classification - 2016 (https://arxiv.org/abs/1605.07725)

核心思想：通过对word Embedding上添加噪音生成对抗样本，将对抗样本以和原始样本 同样的形式喂给模型，得到一个Adversarial Loss，通过和原始样本的loss相加得到新的损失，通过优化该新 的损失来训练模型，作者认为这种方法能对word embedding加上正则化，避免过拟合。

**Practices**

a. 文本分类实战（七）—— Adversarial LSTM模型 (https://www.cnblogs.com/jiangxinyang/p/10208363.html)

### 6.6 Boosting & Bagging & Stacking

**Source**

https://github.com/brightmart/text_classification/blob/master/a00_boosting/a08_boosting.py (Tensorflow)

### 6.7 Summary


## 7. Embeddings
### 7.1 Overview

Keywords: Word2Vec  Wiki2Vec  GloVe  Ngram2Vec  Para2Vec  Doc2Vec StarSpace

### 7.2 Word2Vec
**Papers** - OK

Word2Vec - Efficient Estimation of Word Representations in Vector Space-2013(https://arxiv.org/abs/1301.3781)

Hierarchical Softmax & Negative Sampling & Subsampling of Frequent Words & Phrase Learning - Distributed Representations of Words and Phrases and their Compositionality-2013(https://arxiv.org/abs/1310.4546)

**Comment**

Yao: These are not easy to understand, and you'd better learn them by reading some other articles, such as the ones in 'Articles' below.

**Source**

Website: 

Github: https://github.com/danielfrg/word2vec

**Articles** - OK

Skip-Gram: http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/

Negative Sampling & Subsampling of Frequent Words & Phrase Learning: http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/

**Tool/Library**

Gensim:https://radimrehurek.com/gensim/models/word2vec.html

**Practices** - TODO

a. How to Create a Simple Word2Vec Network and What Are the Input and Output?

b. How to Use Word2Vec Tools to Train Your Own Embeddings? Such as gensim.models.word2vec and others. Both English and Chinese.

For English

Wikipeida --> Word Embeddings; Wiki2Vec

For Chinese

People's Daily  --> Word Embeddings

缺点：

一个单词对应一个固定的向量，无法很好处理多义词。

**Further** - TODO

Interesting Job: Applying Word2Vec to Recommenders and Advertising?(http://mccormickml.com/2018/06/15/applying-word2vec-to-recommenders-and-advertising/)

### 7.3 GloVe
**Papers**

GloVe: Global Vectors for Word Representation (https://nlp.stanford.edu/projects/glove/)

**Source**

Github: https://github.com/stanfordnlp/GloVe

**Articles**

a. (https://blog.csdn.net/coderTC/article/details/73864097)

Yao：本质上是想让词向量(经一定的函数处理后)具有共现矩阵的性质，方法上使用了较多脑洞和技巧，有种强行凑出来的感觉，而非严谨的数学推导，不过对于想做研究的人来说，可以是一种有启发的借鉴，借鉴的不是想法，而是做法！

**Practices**

How to Use Pretrained Word2Vec/GloVe Word Embeddings?

To Do Simple Jobs

To Create Embedding Layer for Neural Network

### 7.4 EMLO - TOTODO
**Papers**

Deep contextualized word representations - AllenAI2018 (https://arxiv.org/abs/1802.05365)

**Articles**

b. ELMO模型（Deep contextualized word representation）(https://www.cnblogs.com/jiangxinyang/p/10060887.html)

**Practices**

b. 文本分类实战（九）—— ELMO 预训练模型 (https://www.cnblogs.com/jiangxinyang/p/10235054.html)

### 7.5 BERT - TOTODO
**Papers**

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding - Google2018 (https://arxiv.org/abs/1810.04805)

**Source**

Github: https://github.com/google-research/bert (Tensorflow)

**Practices**

a. https://github.com/brightmart/text_classification/tree/master/a00_Bert (Tensorflow)

b. 文本分类实战（十）—— BERT 预训练模型 (https://www.cnblogs.com/jiangxinyang/p/10241243.html)

GPT

### 7.6 Ngram2Vec
Maybe it's useful for attr-attrval matching!?!

Phrase2Vec???

**Papers**

Ngram2vec: Learning Improved Word Representations from Ngram Co-occurrence Statistics (http://www.aclweb.org/anthology/D17-1023)

**Source**

Github: https://github.com/zhezhaoa/ngram2vec/

### 7.7 Sentence2Vec

Universal Sentence Encoder-2018 

Paper: https://arxiv.org/abs/1803.11175

Source: https://tfhub.dev/google/universal-sentence-encoder/2

Supervised Learning of Universal Sentence Representations from Natural Language Inference Data-2018

Paper: https://arxiv.org/abs/1705.02364

Source: https://github.com/facebookresearch/InferSent

Shortcut-Stacked Sentence Encoders for Multi-Domain Inference-2017

Paper: https://arxiv.org/abs/1708.02312

Source: https://github.com/easonnie/ResEncoder

### 7.8 Doc2Vec & Paragraph2Vec


### 7.9 StarSpace
**Papers**

StarSpace: Embed All The Things! (https://arxiv.org/abs/1709.03856)

损失函数：相似Entity之间相似度较高

**Source**

Github: https://github.com/facebookresearch/StarSpace


### 7.10 Others

Wiki2Vec: https://github.com/idio/wiki2vec


### 7.11 Summary

## 8. Language Models
### 8.1 Overview

Keywords: BOW  NNLM  (Tools: Onehot TFIDF Ngram)

Statistical Language Models

### 8.2 Statistical Language Models

### 8.3 BOW

### 8.4 NNLM

Purpose: To predict N-th word using previous N-1 words

N-gram

### 8.5 Character-Level Models
**Papers**

Character-Aware Neural Language Models-2015 (https://arxiv.org/abs/1508.06615)

**Source**

Github: https://github.com/chiragjn/deep-char-cnn-lstm (Keras)

8.6 Summary

## 9. Text Classification
### 9.1 Overview

Keywords: TextCNN  VDCNN  fastText  DRNN  

Object: Sentiment Analysis, Object Extraction, Categorization

**Github**

a. https://github.com/brightmart/text_classification (Tensorflow)

b. https://github.com/jiangxinyang227/textClassifier (Tensorflow)

**Practices**

a. 文本分类实战系列文章 (https://www.cnblogs.com/jiangxinyang/p/10207482.html) (Tensorflow)

b. 阿里AI工程师教你如何用CNN RNN Attention解决大规模文本分类问题 (https://www.sohu.com/a/130492867_642762)

### 9.2 fastText
**Papers**

Bag of Tricks for Efficient Text Classification (https://arxiv.org/abs/1607.01759)

**Source**

Github: https://github.com/facebookresearch/fastText

Website: https://fasttext.cc/

**Practices**

a. 直接使用Python的fastText库：

```Python
from fastText import train_supervised
model = train_supervised('train.txt', epoch=25, lr=1.0, wordNgrams=2, minCount=1, loss='hs')
N, P, R = model.test('test.txt', k=1)
model.test_label('test.txt', k=1, threshold=0.5)  # Return precision and recall for each label
model.predict('五彩 手机壳', k=1, threshold=0.5)
model.save_model("model.bin")
```

其中，train.txt每行Schema：用__label__表示label，使用空格逗号空格分隔label和text，其中label是类别名字或编号，text是用空格分隔的各个word

> Schema: __label__0 , 贴膜 金刚膜

b. 还有一种自己搭建模型的方式，不确定是否合理：

Structure: inputs-->Embedding-->SpatialDropout1D-->GlobalAveragePooling1D-->Dense-->Softmax/Sigmoid  ???

### 9.3 TextCNN
**Papers**

a. Convolutional Neural Networks for Sentence Classification (https://arxiv.org/abs/1408.5882)

Structure: Input -> Embedding -> (Conv1D -> GlobalMaxPooling1D) * n_filter_size -> Concatenate -> Dropout -> Dense -> Dropout -> Dense，其中GlobalMaxPooling1D与MaxPooling1D-->Flatten功能相同。

要点：

TextCNN的输入Embedding有3种：

1). 随机初始化

2). Static Pre-trained

3). Two-Channel: Static Pre-trained + Dynamic Embedding，其中Dynamic有2种：Pre-trained + Finetuning和随机初始化，Dynamic的作用是，通过对语料的学习，模型可以得到task-specific的信息。

b. 对TextCNN进行调优：A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification(https://arxiv.org/abs/1510.03820)

论文基于one-layer CNNs，研究了Input Word Vectors、Filter Region Size、Number of Feature Maps for Each Filter Region Size、Activation Function、Pooling Strategy和Regularization等对模型性能的影响，有助于我们参考以选择适合自己的参数。

c. charCNN

Paper: Character-level Convolutional Networks for Text Classification-2016 (https://arxiv.org/abs/1509.01626)

Practice: 文本分类实战（三）—— charCNN模型 (https://www.cnblogs.com/jiangxinyang/p/10207686.html)

Source: https://github.com/jiangxinyang227/textClassifier

**Source**

a. Yao(Keras)

```Python
from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, Dense
from keras.models import Model
def TextCNNModel(input_shape, embedding_layer):
    sentences = Input(input_shape, dtype='int32')
    X = embedding_layer(sentences)                          # (None, maxlen, emb_dim)
    Xs = []
    for fsize in [2, 3, 4]:
        Xi = Conv1D(128, fsize, activation='relu')(X)       # (None, maxlen-fsize+1, 128)
        Xi = GlobalMaxPooling1D()(Xi)                       # (None, 128) Equals 2 lines above
        Xs.append(Xi)
    X = Concatenate(axis=-1)(Xs)                            # (None, 128*3)
    X = Dropout(0.5)(X)
    X = Dense(units=32, activation='relu')(X)               # (None, 32)
    X = Dense(units=1, activation='sigmoid')(X)             # (None, 1)
    return Model(sentences, X)
```

b. https://github.com/jiangxinyang227/textClassifier (Tensorflow)

**Practices**


### 9.4 TextRNN

Structure1: Embedding -> BiLSTM -> Average -> Softmax

Structure2: Embedding -> BiLSTM -> Dropout -> LSTM -> Dropout -> Softmax


**Source**

https://github.com/brightmart/text_classification (Tensorflow)


**Practices**

1. 其实就是使用LSTM或BiLSTM按一般方式去处理Sequnce，没什么花样。

### 9.5 TextRCNN

**Papers**

Recurrent Convolutional Neural Networks for Text Classification (https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552)

Structure

Recurrent Structure(Convolutional Layer) -> MaxPooling -> Dense(Softmax)

Representation of Current Word = [ Left_context_vector, Current_word_embedding, Right_context_vecotor]

For Left_context Cl(w4), it uses a recurrent structure, a non-linearity transform of previous word w3('stroll') and left previous context Cl(w3)


**Source**

a. https://github.com/brightmart/text_classification (Tensorflow)

核心代码：生成Left_context_vector和Right_context_vector，注意是双向的，顺序生成各个Left_context_vector，倒序生成各个Right_context_vector

```Python
# left与right相似，本质上都是相乘相加再激活
# Context(n) = Activation(W * Context(n-1) + W_s * Embedding(n-1))

# Size：context_left-[batch_size, embed_size]  W_l-[embed_size, embed_size]  embedding_previous-[batch_size, embed_size]
def get_context_left(self, context_left, embedding_previous):
    left_h = tf.matmul(context_left, self.W_l) + tf.matmul(embedding_previous, self.W_sl)
    context_left = self.activation(left_h)
    return context_left    # [None,embed_size]
    
def get_context_right(self, context_right, embedding_afterward):
    right_h = tf.matmul(context_right, self.W_r) + tf.matmul(embedding_afterward, self.W_sr)
    context_right = self.activation(right_h)
    return context_right    # [None,embed_size]
```

b. https://github.com/jiangxinyang227/textClassifier (Tensorflow)

**Practices**

a. 文本分类实战（六）—— RCNN模型 (https://www.cnblogs.com/jiangxinyang/p/10208290.html) (Tensorflow)

### 9.6 VDCNN

**Papers**

Very Deep Convolutional Networks for Text Classification - Facebook2017 (https://arxiv.org/abs/1606.01781)

**Source**

Github: https://github.com/lethienhoa/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing (Tensorflow)

### 9.7 DRNN


### 9.8 Others

multi_channel_CNN, deep_CNN, LSTM_CNN, Tree-LSTM

**Practice**

a. 详解文本分类之DeepCNN的理论与实践 (https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247485837&idx=1&sn=1b8f6e0f1ec21d3b73871179f0471428&chksm=eb501d1edc2794081a51eba592da28880bb572caf5bd174869211f7c10719b3c59e703f2ef6b&scene=21#wechat_redirect)

### 9.9 Summary


## 10. Seq2seq
### 10.1 Overview

Attention, Transformer, Pointer Network

### 10.2 Seq2Seq & Encoder2Decoder


### 10.3 Attention

**Papers**

Neural Machine Translation by Jointly Learning to Align and Translate-2014 (https://arxiv.org/abs/1409.0473v2)

**Source**

https://github.com/brightmart/text_classification (Tensorflow)

**Articles**

a. Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention) (https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

### 10.4 Hierarchical Attention Network

**Papers**

Hierarchical Attention Networks for Document Classification (https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

Structure: Word Encoder(BiGRU) -> Word Attention -> Sentence Encoder(BiGRU) -> Sentence Attention -> Softmax

共有Word和Sentence这2种level的Encoder + Attention

Encoder: To get rich representation of word/sentence

Attention: To get important word/sentence among words/sentences

巧妙之处：受Attention启发，这种结构不仅可以获得每个Sentence中哪些words较为重要，而且可获得每个document(很多sentence)中哪些sentences较为重要！It enables the model to capture important information in different levels.

**Source**

https://github.com/brightmart/text_classification (Tensorflow)

### 10.5 Transformer - TOTODO

**Papers**

Attention Is All You Need - Google2017 (https://arxiv.org/abs/1706.03762)

**Source**

a. https://github.com/brightmart/text_classification (Tensorflow)

b. The Annotated Transformer (http://nlp.seas.harvard.edu/2018/04/03/attention.html) (PyTorch)

**Articles**

a. The Illustrated Transformer (https://jalammar.github.io/illustrated-transformer/)


## 11. Probabilistic Graphical Models (PGM)
### 11.1 Overview

Hidden Markov Model, Bayesian Network, Conditional Random Field, Maximum Entropy Markov Model, Latent Dirichlet Allocation, LSI

Course: https://www.coursera.org/specializations/probabilistic-graphical-models

### 11.2 Bayesian Network (BN)

### 11.3 Hidden Markov Model (HMM)

### 11.4 Conditional Random Field (CRF)

### 11.5 MEM, MEMM and Others

### 11.6 Expectation-Maximization (EM)

### 11.7 Summary

## 12. Topic Models
### 12.1 Overview

### 12.2 LDA

### 12.3 LSA

### 12.4 Summary

## 13. Transfer Learning and Multi-task Learning
### 13.1 Overview


## 14. Domain Adaption
### 14.1 Overview


## 15. Generative Models
### 15.1 Overview

Generative Adversarial Network (GAN), variational autoencoder, 


## 16. Knowledge Graph
### 16.1 Overview

**Papers**

a. Learning Entity and Relation Embeddings for Knowledge Graph Completion - THU2015(http://nlp.csai.tsinghua.edu.cn/~lzy/publications/aaai2015_transr.pdf)

**Source**

a. https://github.com/thunlp/KB2E (C++)

b. https://github.com/wuxiyu/transE (Python)


# Part III Application

## 17. Application Overview

NLP(Processing)    NLU(Understanding)     NLI(Inference)

**Papers**

Neural Network Models for Paraphrase Identification, Semantic Textual Similarity, Natural Language Inference, and Question Answering (https://arxiv.org/abs/1806.04330)


## 18. Pairwise Input
### 18.1 Overview

NLP领域的任务的输入输出有以下几种：

| 输入 | 输出 | 示例 | 备注 |
| :-: | :-: | :-: | :-: |
| Seq | label |  |  |
| Seq1 | Seq2 | 所有Seq2Seq问题，如翻译、Chatbox、序列生成等 |  |
| Seq1 + Seq2 | label | Pairwise类问题，比如判断2个Seq的关系或相似度、Chatbox等 |  | 

除7.2外，其他模型结构都是或类似于双胞胎网络(Siamese Network)，2个网络的结构是完全一致的，但其参数，有时共享，有时不同？

### 18.2 BiLSTMTextRelation

Structure: Input(Seq EOS Seq) -> Embeddding -> BiLSTM -> Average -> Softmax

Same with TextRNN, but input is special designed.

e.g. input: "How much is the computer ? EOS Price of laptop", where 'EOS' is a special token splitted input1 and input2

**Source**

https://github.com/brightmart/text_classification (Tensorflow)

### 18.3 twoCNNTextRelation - OK

Structure: Seq1(Input1 -> Embedding -> TextCNN) + Seq2(Input2 -> Embedding -> TextCNN) -> Concatenate -> Softmax

产品词关系项目中使用的模型与此类似，在此基础上增加了第3个Input（结构化输入）。

**Source**

https://github.com/brightmart/text_classification (Tensorflow)

### 18.4 BiLSTMTextRelationTwoRNN

又叫 Dual Encoder LSTM Network ?

**Papers**

The Ubuntu Dialogue Corpus: A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems-2016 (https://arxiv.org/abs/1506.08909)

Structure:  Seq1(Input1 -> Embedding -> BiLSTM) + Seq2(Input2 -> Embedding -> BiLSTM) -> Dot Product -> Softmax

Dot Product作用：To measure the similarity of the predicted response r' and the actual response r by taking the dot product of these two vectors. A large dot product means the vectors are similar and that the response should receive a high score. We then apply a sigmoid function to convert this score into a probability. Similarity --> Probability


**Articles**

Deep Learning for Chatbots, Part 2 – Implementing a Retrieval-Based Model in Tensorflow(http://www.wildml.com/2016/07/deep-learning-for-chatbots-2-retrieval-based-model-tensorflow/)

### 18.5 Others

**Papers**

Pairwise relation classification with mirror instances and a combined convolutional neural network - Singapore (https://www.aclweb.org/anthology/C16-1223)

Github: https://github.com/jefferyYu/Pairwise-relation-classification (Torch, Lua)


## 19. Text Similarity
### 19.1 Overview

### 19.2 Papers

a. Siamese Recurrent Architectures for Learning Sentence Similarity-2016 (www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf, https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/view/12195)

b. Learning Text Similarity with Siamese Recurrent Networks-2016 (http://www.aclweb.org/anthology/W16-16)

**Github**

https://github.com/likejazz/Siamese-LSTM (Keras)

https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb (Keras)

https://github.com/dhwajraj/deep-siamese-text-similarity (Tensorflow)

https://github.com/vishnumani2009/siamese-text-similarity (Tensorflow)

https://github.com/aditya1503/Siamese-LSTM (Theano)


## 20. Sentiment Analysis
### 20.1 Overview

## 21. Named Entity Recognition (NER)
### 21.1 Overview

### 21.2 Using HMM

### 21.3 Using RNN

### 21.4 Using RNN + CRF


## 22. Part-of-Speech Tagging (POS)
### 22.1 Overview

### 22.2 Using MEM

Link: https://github.com/yandexdataschool/nlp_course/blob/master/week05_structured/tagger.ipynb

Paper: A Maximum Entrop Model for Part-Of-Speech Tagging (http://www.aclweb.org/anthology/W96-0213)

### 22.3 Using RNN

Using simpleRNN and BiLSTM: https://github.com/yandexdataschool/nlp_course/blob/master/week05_structured/rnn_tagger.ipynb

### 22.4 Using RNN + CRF


## 23. Machine Translation
### 23.1 Overview


## 24. NLU
### 24.1 Overview

### 24.2 Memory Network

**Papers**

a. Memory Networks - Facebook2015 (https://arxiv.org/abs/1410.3916)

b. End-To-End Memory Networks - Facebook2015 (http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)


**Articles**

a. 论文笔记 - Memory Networks 系列 (https://zhuanlan.zhihu.com/p/32257642?edition=yidianzixun&utm_source=yidianzixun&yidian_docid=0HymGR2b)

**Source**

https://github.com/brightmart/text_classification (Tensorflow)

### 24.3 Recurrent Entity Network

**Papers**

Tracking the World State with Recurrent Entity Networks - Facebook2017 (https://arxiv.org/abs/1612.03969)

Structure

Input (Context, Question) -> Encoding (BOW with Position Mask or BiRNN) -> 

Dynamic Memory (Similarity of keys, values-> Gate -> Candidate Hidden State -> Current Hidden State) -> 

Output Module (Similarity of Query and Hidden State -> Possibility Distribution -> Weighted Sum -> Non-linearity Transform -> Predicted Label)

**Source**

a. https://github.com/facebook/MemNN/tree/master/EntNet-babi (Torch, Lua)

b. https://github.com/jimfleming/recurrent-entity-networks (Tensorflow)

c. https://github.com/brightmart/text_classification (Tensorflow)


## 25. Text Matching

Text Matching and Text Entailment both belong to Natural Language Inference (NLI), and sometimes they are very close to each other.

### 25.1 Overview

A curated list of papers dedicated to neural text (semantic) matching.

Github: https://github.com/NTMC-Community/awaresome-neural-models-for-semantic-match

### 25.2 MatchZoo

**Papers**

MatchZoo: A Toolkit for Deep Text Matching-2017 (https://arxiv.org/abs/1707.07270)

**Source**

Github: https://github.com/NTMC-Community/MatchZoo


## 26. Text Entailment
### 26.1 Overview

Textual entailment (TE) in natural language processing is a directional relation between text fragments. The relation holds whenever the truth of one text fragment follows from another text. In the TE framework, the entailing and entailed texts are termed text (t) and hypothesis (h), respectively. Textual entailment is not the same as pure logical entailment — it has a more relaxed definition: 

t entails h (t ⇒ h) if, typically, a human reading t would infer that h is most likely true.

中文名：文本蕴含

Wikipedia: https://en.wikipedia.org/wiki/Textual_entailment

**Articles**

Textual entailment with TensorFlow (https://www.colabug.com/496258.html)

**Books**

Recognizing Textual Entailment: Models and Applications (https://ieeexplore.ieee.org/document/6812786)

### 26.2 ESIM

**Papers**

Enhanced LSTM for Natural Language Inference-2017 (https://arxiv.org/pdf/1609.06038.pdf)

Usage Scenario

**Source**

https://github.com/sdnr1/EBIM-NLI (Keras)

https://blog.csdn.net/wcy23580/article/details/84990923 (Keras)

https://github.com/zsy23/chinese_nlp/tree/master/natural_language_inference (Tensorflow)

https://github.com/allenai/allennlp/blob/master/allennlp/models/esim.py (PyTorch)

https://github.com/pengshuang/Text-Similarity (PyTorch)

**Practices**

Python实战——ESIM 模型搭建（keras版）(https://blog.csdn.net/wcy23580/article/details/84990923)


### 26.3 ABCNN

**Papers**

ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs-2018 (https://arxiv.org/abs/1512.05193)

**Source**

Github: https://github.com/lsrock1/abcnn_pytorch (PyTorch)


## 27. Text Summary
### 27.1 Overview


## 28. Information Extraction
### 28.1 Overview


## 29. Dialogue Systems
### 29.1 Overview

CopyNet


## 30. Adversarial Methods
### 30.1 Overview
