
# 20. Sentiment Analysis

## 20.1 Overview

#### Paper

- [Deep Learning for Sentiment Analysis : A Survey - 2018](https://arxiv.org/abs/1801.07883)

    中文翻译：[就喜欢看综述论文：情感分析中的深度学习](https://cloud.tencent.com/developer/article/1120718)

    **YAO**: 2017年及之前的论文综述，讲了三种粒度级别的情感分析，Related Tasks(包括Aspect Extraction, Opinion Extraction, Sentiment Composition等)，带有词嵌入的情感分析，以及嘲讽Detection，Emotion分析等。


#### Code

- 【Great】<https://github.com/zenRRan/Sentiment-Analysis> (PyTorch)

    **YAO**: Great在于提供了很多分类模型的PyTorch实现，并在权威数据上验证，同时各种方法封装得特别好
    
    - data：TREC, SUBJ, MR, CR, MPQA
    
    - 模型：Pooling, TextCNN, MultiChannel-TextCNN, MultiLayer-TextCNN, Char-TextCNN, GRU, LSTM, LSTM-TextCNN, TreeLSTM,TreeLSTM-rel, BiTreeLSTM, BiTreeLSTM-rel, BiTreeGRU, TextCNN-TreeLSTM, LSTM-TreeLSTM, LSTM-TreeLSTM-rel

    **YAO**: 任务类型是**Sentence-Level的二或多分类**，包括Question Type 六分类，主观客观二分类，电影评论正负二分类，商品评论正负二分类，文章opinion正负二分类

- <https://github.com/ami66/ChineseTextClassifier> (Tensorflow)

    京东商城中文商品评论短文本分类器，可用于情感分析。模型包括：Transformer, TextCNN, FastText, LSTM/GRU, LSTM/GRU+Attention, BiLSTM+Attention

    **YAO**: 任务类型是**Sentence-Leve的正负二分类**

- <https://github.com/anycodes/SentimentAnalysis> (Tensorflow)

    基于深度学习（LSTM）的情感分析(京东商城数据)

    **YAO**: 任务类型是**Sentence-Level的正负二分类**


#### Competition

**AI challenger 2018**:

> 

- 【Great】<https://github.com/BigHeartC/Al_challenger_2018_sentiment_analysis> (Tensorflow & Keras)
  
    细粒度用户评论情感分析，排名17th，基于Aspect Level 思路的解决方案

    **YAO**: 任务类型是在**Document**中进行**Aspect-Level**的情感分析

    **YAO**: 实现并修改了GCAE和SynAtt两个论文并进行了模型融合，特征包括SVD特征、Embedding特征   .......待继续

- <https://github.com/chenghuige/wenzheng> (Tensorflow & PyTorch)
  
    细粒度情感分类第一名解决方案，统一使用tensorflow和pytorch的一个框架

- <https://github.com/pengshuang/AI-Comp> (Tensorflow & Keras)
  
    Baseline 细粒度用户评论情感分析

- <https://github.com/xueyouluo/fsauor2018> (Tensorflow)
  
    Code for Fine-grained Sentiment Analysis of User Reviews


**BDCI 2018 汽车行业用户观点主题及情感识别比赛**:

> 

- <https://github.com/yilifzf/BDCI_Car_2018> (PyTorch)

    决赛一等奖方案

- <https://github.com/nlpjoe/CCF-BDCI-Automotive-Field-ASC-2018> (Tensorflow)

    第6名解决方案


**搜狐2019内容识别算法大赛**:

> 给定若干文章，从文章中识别最多三个核心实体以及对核心实体的情感态度(消极，中立，积极)

- <https://github.com/Fengfeng1024/SOHU-baseline> (Keras)

    **YAO**: 任务类型是在**Document**中进行**Aspect-Level**的情感分析，任务过程为2个子任务--Aspect识别+对Aspect进行情感极性三分类，前者类似于NER，后者使用了Attention机制

- <https://github.com/sys1874/seq2seq-model-for-Sohu-2019> (PyTorch)

    完全端到端的核心实体识别与情感预测

    **YAO**: 任务类型同上，任务过程是End-to-End，模型是Attention(Seq2Seq)+ELMo+ensemble

- <https://github.com/lmhgithi/2019-sohu-competition>

    决赛解决方案ppt、实体LightGBM单模代码

    **YAO**: 任务类型只包括**Document-Level的二分类**，模型是LightGBM，使用了一些常规通用常用的方法，如自己训练word2vec、doc2vec模型、tfidf模型


**搜狐2018内容识别大赛**:

> 对文章和图片乾分类(全部营销、部分营销和无营销)，并从部分营销的文章中抽取属性营销部分的文本片段

- 【Great】<https://github.com/zhanzecheng/SOHU_competition> (Keras)

    第一名解决方案，使用的模型包括：CatBoost, XGBoost, LightGBM, DNN, TextCNN, Capsule, CovLSTM, DPCNN, LSTM+GRU, LSTM+GRU+Attention, 模型技巧为：Stacking, Snapshot Ensemble, Pesudo Labeling.

    **YAO**: 任务类型是**Document-Level**的文本和图片**三分类**，以及对部分营销的文本进行**信息抽取**

    **YAO**: Great在处理了文本和图片特征，把全部特征汇总在一起，使用传统模型和深度模型，并使用模型融合等技巧，非常实用！


## 20.2 Deep Learning

- [Aspect Level Sentiment Classification with Deep Memory Network - HIT2016](https://arxiv.org/abs/1605.08900)

- [Effective LSTMs for Target-Dependent Sentiment Classification - HIT2016](https://arxiv.org/abs/1512.01100)

- [Attention-based LSTM for Aspect-level Sentiment Classification - THU2016](https://www.aclweb.org/anthology/D16-1058)

    **Code**: <http://coai.cs.tsinghua.edu.cn/media/files/atae-lstm_uVgRmdb.rar>

- [Interactive Attention Networks for Aspect-Level Sentiment Classification - PKU2017](https://arxiv.org/abs/1709.00893)

- GCAE: [Aspect Based Sentiment Analysis with Gated Convolutional Networks - FIU2018](https://arxiv.org/abs/1805.07043)

- [IARM: Inter-Aspect Relation Modeling with Memory Networks in Aspect-Based Sentiment Analysis - 2018](https://aclweb.org/anthology/D18-1377)

    **Code**: <https://github.com/SenticNet/IARM> (PyTorch)

- SynATT: [Effective Attention Modeling for Aspect-Level Sentiment Classification - Singapore2018](https://www.aclweb.org/anthology/C18-1096)

- [Exploiting Document Knowledge for Aspect-level Sentiment Classification - Singapore2018](https://arxiv.org/abs/1806.04346)
  
    **Code**: <https://github.com/ruidan/Aspect-level-sentiment> (Keras)


## 20.3 Topic Model

主题模型是一种非监督学习方法

- [Joint sentiment/topic model for sentiment analysis - 2009](http://people.sabanciuniv.edu/berrin/share/LDA/YulanHe-JointSentiment-Topic-2009.pdf)

- [Sentiment Analysis with Global Topics and Local Dependency - 2010](http://www.cs.huji.ac.il/~jeff/aaai10/02/AAAI10-242.pdf)
