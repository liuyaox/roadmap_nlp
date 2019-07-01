
# 20. Sentiment Analysis

## 20.1 Overview

#### Paper

- [Deep Learning for Sentiment Analysis : A Survey - 2018](https://arxiv.org/abs/1801.07883)

    中文翻译：[就喜欢看综述论文：情感分析中的深度学习](https://cloud.tencent.com/developer/article/1120718)

    **YAO**: 2017年及之前的论文综述，讲了三种粒度级别的情感分析，Related Tasks(包括Aspect Extraction, Opinion Extraction, Sentiment Composition等)，以及带有词嵌入的情感分析，以及嘲讽Detection，Emotion分析等。

#### Code

- 【Great】<https://github.com/zenRRan/Sentiment-Analysis> (PyTorch)

    **YAO**: Great在于提供了很多分类模型的PyTorch实现，并在权威数据上验证，同时各种方法封装得特别好
    
    - data：TREC, SUBJ, MR, CR, MPQA
    
    - 模型：Pooling, TextCNN, MultiChannel-TextCNN, MultiLayer-TextCNN, Char-TextCNN, GRU, LSTM, LSTM-TextCNN, TreeLSTM,TreeLSTM-rel, BiTreeLSTM, BiTreeLSTM-rel, BiTreeGRU, TextCNN-TreeLSTM, LSTM-TreeLSTM, LSTM-TreeLSTM-rel

    **YAO**: 任务类型是**Sentence-Level**，包括Question Type 六分类，主观客观二分类，电影评论正负二分类，商品评论正负二分类，文章opinion正负二分类

- <https://github.com/ami66/ChineseTextClassifier> (Tensorflow)

    京东商城中文商品评论短文本分类器，可用于情感分析。模型包括：Transformer, TextCNN, FastText, LSTM/GRU, LSTM/GRU+Attention, BiLSTM+Attention

    **YAO**: 任务类型是**Sentence-Level**的**正负二分类**

- <https://github.com/anycodes/SentimentAnalysis> (Tensorflow)

    基于深度学习（LSTM）的情感分析(京东商城数据)

    **YAO**: 任务类型是**Sentence-Level**的**正负二分类**

#### Competition

- <https://github.com/xueyouluo/fsauor2018> (Tensorflow)
  
    Code for Fine-grained Sentiment Analysis of User Reviews of AI Challenger 2018

- 【Great】<https://github.com/BigHeartC/Al_challenger_2018_sentiment_analysis> (Tensorflow & Keras)
  
    AI Challenger 2018 细粒度用户评论情感分析，排名17th，基于Aspect Level 思路的解决方案

- <https://github.com/chenghuige/wenzheng> (Tensorflow & PyTorch)
  
    AI challenger 2018 细粒度情感分类第一名解决方案,统一使用tensorflow和pytorch的一个框架

- <https://github.com/pengshuang/AI-Comp> (Tensorflow & Keras)
  
    AI Challenger Baseline 细粒度用户评论情感分析

- <https://github.com/yilifzf/BDCI_Car_2018> (PyTorch)

    BDCI 2018 汽车行业用户观点主题及情感识别 决赛一等奖方案

- <https://github.com/nlpjoe/CCF-BDCI-Automotive-Field-ASC-2018> (Tensorflow)

    BDCI 2018 汽车行业用户观点主题及情感识别挑战赛 第6名解决方案

- <https://github.com/Fengfeng1024/SOHU-baseline> (Keras)

    搜狐内容识别算法大赛：给定若干文章，判断文章的核心实体以及对核心实体的情感态度。

- 【Great】<https://github.com/zhanzecheng/SOHU_competition> (Keras)

    搜狐2018内容识别大赛第一名解决方案，使用的模型包括：CatBoost, XGBoost, LightGBM, DNN, TextCNN, Capsule, CovLSTM, DPCNN, LSTM+GRU, LSTM+GRU+Attention, 模型技巧为：Stacking, Snapshot Ensemble, Pesudo Labeling.


## 20.2 Deep Learning

- [Aspect Level Sentiment Classification with Deep Memory Network - HIT2016](https://arxiv.org/abs/1605.08900)

- [Effective LSTMs for Target-Dependent Sentiment Classification - HIT2016](https://arxiv.org/abs/1512.01100)

- [Attention-based LSTM for Aspect-level Sentiment Classification - THU2016](https://www.aclweb.org/anthology/D16-1058)

    **Code**: <http://coai.cs.tsinghua.edu.cn/media/files/atae-lstm_uVgRmdb.rar>

- [IARM: Inter-Aspect Relation Modeling with Memory Networks in Aspect-Based Sentiment Analysis - 2018](https://aclweb.org/anthology/D18-1377)

    **Code**: <https://github.com/SenticNet/IARM> (PyTorch)


## 20.3 Topic Model

主题模型是一种非监督学习方法

- [Joint sentiment/topic model for sentiment analysis - 2009](http://people.sabanciuniv.edu/berrin/share/LDA/YulanHe-JointSentiment-Topic-2009.pdf)

- [Sentiment Analysis with Global Topics and Local Dependency - 2010](http://www.cs.huji.ac.il/~jeff/aaai10/02/AAAI10-242.pdf)


