
# 20. Sentiment Analysis

YAO's: <https://github.com/liuyaox/sentiment_analysis> (Keras & PyTorch)

## 20.1 Overview

Sentiment Analysis 按粒度可分为3种：

- Document Level
- Sentence Level
- Aspect Level

其中, Aspect Level 的 Sentiment Analysis 按 Aspect 类型又可分为2种：

- ATSA: Aspect-Term Sentiment Analysis

    Aspect-Term：不固定，不唯一，有很多Term共同表示同一种 Aspect，如 image,photo,picture 都是 Term
相关任务是 To group the same aspect expressions into a category，如上面三者可都归为 Image 这一 category

- ACSA: Aspect-Category Sentiment Analysis

    Aspect-Category：表示一种 Aspect，固定而唯一，如上例中的 Image

#### Paper

- [Deep Learning for Sentiment Analysis : A Survey - 2018](https://arxiv.org/abs/1801.07883)

    中文翻译：[就喜欢看综述论文：情感分析中的深度学习](https://cloud.tencent.com/developer/article/1120718)

    **YAO**: 2017年及之前的论文综述，讲了三种粒度级别的情感分析，Related Tasks(包括Aspect Extraction, Opinion Extraction, Sentiment Composition等)，带有词嵌入的情感分析，以及嘲讽 Detection，Emotion 分析等。

- <https://github.com/yw411/aspect_sentiment_classification>

    Aspect-Level Sentiment Analysis 论文大全，包括ATSA和ACSA

#### Practice

- 【Great】<https://github.com/zenRRan/Sentiment-Analysis> (PyTorch)

    **YAO**: 
    
    a. Great在于提供了很多分类模型的PyTorch实现，并在权威数据上验证，同时各种方法封装得特别好
    
    - data：TREC, SUBJ, MR, CR, MPQA
    
    - 模型：Pooling, TextCNN, MultiChannel-TextCNN, MultiLayer-TextCNN, Char-TextCNN, GRU, LSTM, LSTM-TextCNN, TreeLSTM,TreeLSTM-rel, BiTreeLSTM, BiTreeLSTM-rel, BiTreeGRU, TextCNN-TreeLSTM, LSTM-TreeLSTM, LSTM-TreeLSTM-rel

    b. 任务类型是 **Sentence-Level 的二或多分类**，包括Question Type 六分类，主观客观二分类，电影评论正负二分类，商品评论正负二分类，文章opinion正负二分类

- <https://github.com/ami66/ChineseTextClassifier> (Tensorflow)

    京东商城中文商品评论短文本分类器，可用于情感分析。模型包括：Transformer, TextCNN, FastText, LSTM/GRU, LSTM/GRU+Attention, BiLSTM+Attention

    **YAO**: 任务类型是 **Sentence-Level 的正负二分类**

- <https://github.com/anycodes/SentimentAnalysis> (Tensorflow)

    基于深度学习（LSTM）的情感分析(京东商城数据)

    **YAO**: 任务类型是 **Sentence-Level 的正负二分类**

- <https://github.com/BUPTLdy/Sentiment-Analysis> (Keras)

    Chinese Shopping Reviews sentiment analysis 

- <https://github.com/liuhuanyong/SentenceSentimentClassifier> (Keras)

    基于机器学习与深度学习方法的情感分析算法实现与对比，包括决策树，贝叶斯，KNN, SVM ,MLP, CNN, LSTM实现

- <https://github.com/maowankuiDji/Word2Vec-sentiment>

    基于Word2Vec+SVM对电商的评论数据进行情感分析

- <https://github.com/hehuihui1994/coarse-fine_emotion_classification>

    结合上下文和篇章特征的多标签情绪分类，微博文本中的句子

    **YAO**: 任务是 **Sentence-Level 的分类**， 使用了 MLKNN，情绪转移，卡方统计特征选择

- 【Great!】<https://github.com/jcsyl/news-analyst> (Keras)

    对舆情事件进行词云展示，对评论进行情感分析和观点抽取。情感分析基于lstm 的三分类，观点抽取基于 AP 算法的聚类和MMR的抽取

    **YAO**: 使用 TFIDF 和 TextRank 提取关键词，使用 Word2Vec 和 LSTM 进行情感三分类，通过 AP 聚类进行观点聚类和抽取！

- <https://github.com/peace195/aspect-based-sentiment-analysis> (Tensorflow)

    Aspect Based Sentiment Analysis

- <https://github.com/Zbored/Chinese-sentiment-analysis>

    gensim-word2vec+svm文本情感分析

    **YAO**: SVM如何与Word Embedding结合: Word Embedding --<求均值>--> Document Embedding --> 直接输入SVM

- <https://github.com/soujanyaporia/aspect-extraction> (Tensorflow)

    Aspect extraction from product reviews - window-CNN+maxpool+CRF, BiLSTM+CRF, MLP+CRF

    **YAO**: 注意，只是Aspect Extraction

- <https://github.com/cedias/Hierarchical-Sentiment> (PyTorch)

    Hierarchical Models for Sentiment Analysis in Pytorch


#### Competition

**AI challenger 2018 餐饮行业细粒度用户评论情感分析**:

> 国内目前为止最大最全的面向餐饮领域的细分情感分析，任务类型是在 **Document** 中进行 **Aspect-Level** 的情感分析，Aspect 预先设定好！但好像又可以修改？！
> 
> 训练数据Schema：<comment, aspect1, polarity1, aspect2, polarity2, ...> 注意，有多个aspect及其polarity！

- <https://github.com/BigHeartC/Al_challenger_2018_sentiment_analysis> (Tensorflow)
  
    基于 Aspect Level 思路的解决方案，排名17th

    **YAO**:

    a. 实现并修改了 GCAE 和 SynATT 两个模型(论文见20.2)并进行了模型融合，特征包括特征化特征(TFIDF+SVD)和 Embedding 特征(Char-level & Word-level)

    b. 关键技术有：词向量和字向量联合表示，GCAE，SynATT

    c. 详细解析和注释见：[forked_repos_with_notes](https://github.com/liuyaox/forked_repos_with_notes/tree/master/Al_challenger_2018_sentiment_analysis-master)

- <https://github.com/chenghuige/wenzheng> (Tensorflow & PyTorch)
  
    第一名解决方案，统一使用tensorflow和pytorch的一个框架

    **YAO**: 看起来很杂乱的样子。。。暂时放弃吧

- <https://github.com/pengshuang/AI-Comp> (Tensorflow & Keras)
  
    Baseline

- <https://github.com/xueyouluo/fsauor2018> (Tensorflow)
  
    Code for Fine-grained Sentiment Analysis of User Reviews

- <https://github.com/foamliu/Sentiment-Analysis> (PyTorch)

    **YAO**: 数据示例和解释还挺全，待看……


**搜狐2019 内容识别算法大赛**:

> 给定若干文章，从文章中识别最多三个核心实体以及对核心实体的情感态度(消极，中立，积极)，任务类型是在 **Document** 中进行 **Aspect-Level** 的情感分析。
> 
> 训练数据Schema：<comment, aspect1, polarity1, aspect2, polarity2, aspect3, polarity3> 注意，有最多三个 aspect 及其 polarity！

- <https://github.com/Fengfeng1024/SOHU-baseline> (Keras)

    Baseline

    **YAO**:

    好像还挺好，可以学习一下！
        
    任务过程为2个子任务--Aspect 识别 + 对 Aspect 进行情感极性三分类，前者类似于 NER，后者使用了 Attention 机制

- <https://github.com/sys1874/seq2seq-model-for-Sohu-2019> (PyTorch)

    完全端到端的核心实体识别与情感预测

    **YAO**:
    
    任务过程是 End-to-End，模型是 Attention(Seq2Seq)+ELMo+ensemble

- <https://github.com/lmhgithi/2019-sohu-competition>

    决赛解决方案ppt、实体 LightGBM 单模代码

    **YAO**:
    
    任务类型只包括**Document-Level的二分类**，模型是 LightGBM，使用了一些常规通用常用的方法，如自己训练 word2vec、doc2vec 模型、tfidf 模型

- <https://github.com/LLouice/Sohu2019> (PyTorch)

    **YAO**:

    挺详细的，待看……
    
    貌似也挺好，继续了BertPretrainedModel

- <https://github.com/rebornZH/2019-sohu-algorithm-competition>

    季军，只有PPT


**搜狐2018 内容识别大赛**:

> 对文章和图片乾分类(全部营销、部分营销和无营销)，并从部分营销的文章中抽取属性营销部分的文本片段

- 【Great】<https://github.com/zhanzecheng/SOHU_competition> (Keras)

    第一名解决方案，使用的模型包括：CatBoost, XGBoost, LightGBM, DNN, TextCNN, Capsule, CovLSTM, DPCNN, LSTM+GRU, LSTM+GRU+Attention, 模型技巧为：Stacking, Snapshot Ensemble, Pesudo Labeling.

    **YAO**: 
    
    任务类型是 **Document-Level** 的文本和图片**三分类**，以及对部分营销的文本进行**信息抽取**

    Great在处理了文本和图片特征，把全部特征汇总在一起，使用传统模型和深度模型，并使用模型融合等技巧，非常实用！


**BDCI 2018 汽车行业用户观点主题及情感识别比赛**:

> 任务类型在 **Aspect-Level** 的情感分析，Aspect 预先设定好！
> 
> 训练数据Schema：<comment, aspect, polarity>，貌似只有一个 aspect！？

- <https://github.com/yilifzf/BDCI_Car_2018> (PyTorch)

    决赛一等奖方案

- <https://github.com/nlpjoe/CCF-BDCI-Automotive-Field-ASC-2018> (Tensorflow)

    第6名解决方案


#### Article

- 【Great】[如何到top5%？NLP文本分类和情感分析竞赛总结](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247486159&idx=1&sn=522345e275df807942c7b56b0054fec9)


## 20.2 Deep Learning

#### Paper

- [Aspect Level Sentiment Classification with Deep Memory Network - HIT2016](https://arxiv.org/abs/1605.08900)

    **Code**: 1

- [Effective LSTMs for Target-Dependent Sentiment Classification - HIT2016](https://arxiv.org/abs/1512.01100)

    **Code**: 1

- [Attention-based LSTM for Aspect-level Sentiment Classification - THU2016](https://www.aclweb.org/anthology/D16-1058)(ACAS)

    **Code**: 1, <http://coai.cs.tsinghua.edu.cn/media/files/atae-lstm_uVgRmdb.rar>

- [Neural Sentiment Classification with User & Product Attention - THU2016](http://nlp.csai.tsinghua.edu.cn/~chm/publications/emnlp2016_NSCUPA.pdf)

    **Code**: <https://github.com/cedias/Hierarchical-Sentiment> (PyTorch)

- [Interactive Attention Networks for Aspect-Level Sentiment Classification - PKU2017](https://arxiv.org/abs/1709.00893)

    **Code**: 1, <https://github.com/lpq29743/IAN> (Tensorflow)

- [Recurrent Attention Network on Memory for Aspect Sentiment Analysis - Tencent2017](http://www.cs.cmu.edu/~lbing/pub/emnlp17_aspect_sentiment.pdf)

    **Code**: 1, <https://github.com/lpq29743/RAM> (Tensorflow)

- [Content Attention Model for Aspect Based Sentiment Analysis - UESTC2018](http://delivery.acm.org/10.1145/3190000/3186001/p1023-liu.pdf)

    **Code**: 1

- GCAE: [Aspect Based Sentiment Analysis with Gated Convolutional Networks - FIU2018](https://arxiv.org/abs/1805.07043)(ACSA & ATSA)

    **Code**: 2

- [IARM: Inter-Aspect Relation Modeling with Memory Networks in Aspect-Based Sentiment Analysis - Mexico2018](https://aclweb.org/anthology/D18-1377)

    **Code**: <https://github.com/SenticNet/IARM> (PyTorch)

- SynATT: [Effective Attention Modeling for Aspect-Level Sentiment Classification - Singapore2018](https://www.aclweb.org/anthology/C18-1096)(ATSA ? ACSA)

    **Code**: 2

- [Exploiting Document Knowledge for Aspect-level Sentiment Classification - Singapore2018](https://arxiv.org/abs/1806.04346)
  
    **Code**: <https://github.com/ruidan/Aspect-level-sentiment> (Keras)

- [Transformation Networks for Target-Oriented Sentiment Classification - CUHK2018](https://arxiv.org/abs/1805.01086)

    **Code**: <https://github.com/lixin4ever/TNet> (Theano)

- [An Interactive Multi-Task Learning Network for End-to-End Aspect-Based Sentiment Analysis - Singapore2019](https://arxiv.org/abs/1906.06906)

    **Code**: <https://github.com/ruidan/IMN-E2E-ABSA> (Keras)

#### Code

- 1. <https://github.com/AlexYangLi/ABSA_Keras> (Keras)

- 2. <https://github.com/BigHeartC/Al_challenger_2018_sentiment_analysis> (Tensorflow)
  

## 20.3 Topic Model

主题模型是一种非监督学习方法

#### Paper

- [Joint sentiment/topic model for sentiment analysis - 2009](http://people.sabanciuniv.edu/berrin/share/LDA/YulanHe-JointSentiment-Topic-2009.pdf)

- [Sentiment Analysis with Global Topics and Local Dependency - 2010](http://www.cs.huji.ac.il/~jeff/aaai10/02/AAAI10-242.pdf)

#### Article

- 【Great】[Twitter数据挖掘及其可视化 - 2017](https://www.hrwhisper.me/twitter-data-mining-and-visualization/)

    使用模型有 LDA, OLDA, WOLDA

    **YAO**: 一直对 LDA/SVD 感兴趣，可以看此文章


## 20.4 Rule

规则 + 数据挖掘

#### Article

- [基于词典的中文情感倾向分析算法设计 - 2016](https://cloud.tencent.com/developer/article/1059360)