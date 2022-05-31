# 2. Corpus, Data and Tool

## 2.1 Data

### 2.1.1 Overview

- <https://github.com/thunlp/THUOCL>

    [THUOCL：清华大学开放中文词库](http://thuocl.thunlp.org/)  包括以下数据及其词频统计DF值：IT 财经 成语 地名 历史名人 诗词 医学 饮食 法律 汽车 动物

- <https://github.com/brightmart/nlp_chinese_corpus>
  
    大规模中文自然语言处理语料Large Scale Chinese Corpus for NLP  包括：维基百科、新闻语料、百科问答、社区问答、翻译语料
    
    用途：预训练通用语料，QA(知识/百科/社区问答)构建，文本生成/抽取(标题/关键词生成)，文本分类(新闻类型分类、问题话题分类)，文本预测(文本相似性、答案评分)，中英文翻译

- <https://github.com/OYE93/Chinese-NLP-Corpus>

    **Article**: [中文自然语言处理医疗、法律等公开数据集整理分享](https://mp.weixin.qq.com/s/EIOGIpWbaeFzOEdHBEUfHw)

- <https://github.com/niderhoff/nlp-datasets>
  
    Alphabetical list of free/public domain datasets with text data for use in NLP

    **Chinese**: [100+个自然语言处理数据集大放送，再不愁找不到数据！](https://www.sohu.com/a/230090656_642762)

- [Datasets for Natural Language Processing - 2017](https://machinelearningmastery.com/datasets-natural-language-processing/)

- <https://dumps.wikimedia.org/zhwiki/>

    中文 Wikipedia

- <https://radimrehurek.com/gensim/corpora/wikicorpus.html>

    Gensim中的Corpus from a Wikipedia dump

- <https://pan.baidu.com/s/1hsHTEU4>

    上万中文姓名及性别

- <https://github.com/insanelife/chinesenlpcorpus>
  
    中文自然语言处理数据集，平时做做实验的材料

- <https://github.com/fighting41love/funNLP>

    中英文敏感词、语言检测、中外手机/电话归属地/运营商查询、名字推断性别、手机号抽取、身份证抽取、邮箱抽取、中日文人名库、中文缩写库、拆字词典、词汇情感值、停用词、反动词表、暴恐词表、繁简体转换、英文模拟中文发音、汪峰歌词生成器、职业名称词库、同义词库、反义词库、否定词库、汽车品牌词库、汽车零件词库、连续英文切割、各种中文词向量、公司名字大全、古诗词库、IT词库、财经词库、成语词库、地名词库、历史名人词库、诗词词库、医学词库、饮食词库、法律词库、汽车词库、动物词库、中文聊天语料、中文谣言数据、百度中文问答数据集、句子相似度匹配算法集合、bert资源、文本生成&摘要相关工具、cocoNLP信息抽取工具、国内电话号码正则匹配、清华大学XLORE:中英文跨语言百科知识图谱、清华大学人工智能技术……

- <https://github.com/chinese-poetry/chinese-poetry>
  
    最全中华古诗词数据库, 接近5.5万首唐诗，26万宋诗，21050首词

- <https://github.com/pwxcoo/chinese-xinhua>

    中华新华字典数据库。包括歇后语，成语，词语，汉字

- <https://github.com/wb14123/couplet-dataset>

    70万条对联数据库

    [Google Drive 链接](https://drive.google.com/file/d/13cJWWp_ST2Xqt76pEr5ZWa6LVJwGYC4-/view?usp=sharing)

- <https://github.com/andy-yangz/midi_lyric_corpus>

    包含250首中文歌的 midi 文件，以及对应的歌词文本

- <https://github.com/yangjianxin1/QQMusicSpider>

    基于Scrapy的QQ音乐爬虫(QQ Music Spider)，爬取歌曲信息、歌词、精彩评论等，并且分享了QQ音乐中排名前6400名的内地和港台歌手的49万+的音乐语料

- <https://github.com/andy-yangz/chinese>

    中文形近字

- <https://github.com/andy-yangz/corpus>

    古典中文語料庫：全唐诗，古籍字频统计，汉语大词典，三国志，中国历史地名辞典，全宋词，全唐五代词，千家诗，唐诗三百首，宋词三百首，日本姓氏，日本战国时代，楚辞，乐府诗集，花间集，诗经，教育部词频统计表

- <https://github.com/shijiebei2009/CEC-Corpus>

    中文突发事件语料库(Chinese Emergency Corpus) - 上海大学-语义智能实验室

- <https://github.com/fate233/toutiao-multilevel-text-classfication-dataset>

    今日头条 中文文本多层分类数据集，共2914000条，分布于1000+个多层的类别中

- <https://github.com/howl-anderson/tools_for_corpus_of_people_daily>

    人民日报语料处理工具集 | Tools for Corpus of People's Daily

    **YAO**: 支持分词、NER 两个任务

- <https://github.com/Roshanson/TextInfoExp>

    自然语言处理实验（sougou数据集），TF-IDF，文本分类、聚类、词向量、情感识别、关系抽取等

- [A Large Chinese Text Dataset in the Wild](https://ctwdataset.github.io/)

    **Paper**: [A Large Chinese Text Dataset in the Wild - THU2018](https://arxiv.org/abs/1803.00085)

- [Fueling the Gold Rush: The Greatest Public Datasets for AI - 2017](https://medium.com/startup-grind/fueling-the-ai-gold-rush-7ae438505bc2#.3x80s6mw4)

    CV,NLP,语音等6大类

- [【数据集汇总（附下载链接）】再也不愁没数据练习机器学习算法啦！ - 2020](https://mp.weixin.qq.com/s/6WkdQiotvqbmqyXMU_8W3Q)


### 2.1.2 Knowledge Graph

- 大词林：<http://www.bigcilin.com>

    **Article**: [重磅！《大词林》V2.0：自动消歧的开放域中文知识图谱 - 2019](https://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=2650795283&idx=1&sn=5c36be2f86b1892401d4588eb4095b9d)

    **Article**: [哈工大《大词林》开放75万核心实体词及相关概念、关系列表 - 2020](https://mp.weixin.qq.com/s/gGig5KFztInrGAmhUhqAkg)


### 2.1.3 Sentiment Analysis

- <http://alt.qcri.org/semeval2014/>

    SemEval-2014 : Semantic Evaluation Exercises, International Workshop on Semantic Evaluation (SemEval-2014).

    - Task 1: Evaluation of Compositional Distributional Semantic Models on Full Sentences through Semantic Relatedness and Entailment
    - Task 3: Cross-Level Semantic Similarity
    - Task 4: Aspect Based Sentiment Analysis (ABSA)
    - Task 6: Supervised Semantic Parsing of Spatial Robot Commands
    - Task 8: Broad-Coverage Semantic Dependency Parsing
    - Task 9: Sentiment Analysis in Twitter
    - Task 10: Multilingual Semantic Textual Similarity

- <http://www.cs.jhu.edu/~mdredze/datasets/sentiment/>

    Multi-Domain Sentiment Dataset

- <http://www.sananalytics.com/lab/twitter-sentiment/>

    twitter 情感分析数据集

- <https://github.com/Roshanson/TextInfoExp>

    自然语言处理实验（sougou数据集），TF-IDF，文本分类、聚类、词向量、情感识别、关系抽取等

- [中文情感分析语料库大全-带下载地址](https://mlln.cn/2018/10/11/%E4%B8%AD%E6%96%87%E6%83%85%E6%84%9F%E5%88%86%E6%9E%90%E8%AF%AD%E6%96%99%E5%BA%93%E5%A4%A7%E5%85%A8-%E5%B8%A6%E4%B8%8B%E8%BD%BD%E5%9C%B0%E5%9D%80/)

    中文对话情绪语料，微博情感分析测评数据，中文情感词汇本体，中文褒贬义词词典，商品评论情感语料库

- <https://github.com/z17176/Chinese_conversation_sentiment>

    A Chinese sentiment dataset: sentiment_XS_test.txt contains 11577 instances labeled manually. sentiment_XS_30k.txt contains almost 30k instances labeled automatically.

- <https://github.com/liuhuanyong/ChineseHumorSentiment>

    中文文本幽默情绪计算项目,项目包括幽默文本语料库的构建,幽默计算模型,包括幽默等级识别,幽默类型识别,隐喻类型识别,隐喻情绪识别等


### 2.1.4 打造数据集

#### Article

- [如何打造高质量的机器学习数据集？ - 2019](https://www.zhihu.com/question/333074061/answer/773825458)


## 2.2 Embedding

- [Tencent AI Lab Embedding Corpus for Chinese Words and Phrases](https://ai.tencent.com/ailab/nlp/embedding.html)

    This corpus provides 200-dimension vector representations, a.k.a. embeddings, for over 8 million Chinese words and phrases.
  
- <https://github.com/cliuxinxin/TX-WORD2VEC-SMALL>

    上面腾讯word2vec模型缩小版，包括 5K, 4.5W, 7W, 10W, 50W, 100W, 200W 词汇量。

- <https://github.com/Embedding/Chinese-Word-Vectors>

    100+ Chinese Word Vectors 上百种预训练中文词向量 from 北师大和人大

- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)

    有Wikipedia+Gigaword, Common Crawl, Twitter三种语料

## 2.3 Tool

- <https://github.com/huyingxi/Synonyms>

    中文近义词工具包，基于Word Embedding和字符串编辑距离等。

- <https://github.com/KimMeen/Weibo-Analyst>

    微博评论分析工具, 实现功能: 1.微博评论数据爬取; 2.分词与关键词提取; 3.词云与词频统计; 4.情感分析; 5.主题聚类

- [45个小众而实用的NLP开源字典和工具 - 2020](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247485681&idx=1&sn=ba5cc6de539571cd9efe60a38e19bf74)


## 2.4 Practice

- [Working with and analyzing Wikipedia Data](https://github.com/WillKoehrsen/wikipedia-data-science)



## 2.5 Data Augmentation

除下文所说方法外，还可以使用**SentencePiece**: sp.nbest_encode_as_pieces对于同一个原始文本，可以有多种编码，每种编码对应一份样本。

#### Paper

- [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks - USA2019](https://arxiv.org/abs/1901.11196)

    **Github**: <https://github.com/jasonwei20/eda_nlp>

    **Article**: [NLP中一些简单的数据增强技术](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650411500&idx=2&sn=76e635526015ccecd14a1436bda55e2c)

#### Github

- <https://github.com/quincyliang/nlp-data-augmentation>

    Data Augmentation for NLP    NLP数据增强

#### Library

- <https://github.com/makcedward/nlpaug>

    **Article**: [NLP中数据增强的综述，快速的生成大量的训练数据](https://mp.weixin.qq.com/s/9fYPPF51RbJi1wNHBrJVzQ)

#### Article

- 【Great】[文本增强技术的研究进展及应用实践 - 2020](https://mp.weixin.qq.com/s/CHSDi2LpDOLMjWOLXlvSAg)

- [These are the Easiest Data Augmentation Techniques in NLP you can think of — and they work - 2019](https://towardsdatascience.com/these-are-the-easiest-data-augmentation-techniques-in-natural-language-processing-you-can-think-of-88e393fd610)

- [赛尔笔记 | 深度学习领域的数据增强 - 2019](https://mp.weixin.qq.com/s?__biz=MzIxMjAzNDY5Mg==&mid=2650796432&idx=1&sn=6253922c918463e3dbc45abe5e0e57e1)

- [数据增强在语音识别中的应用 - 2020](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650412581&idx=3&sn=405e693504f448340ab4a6ad56020f3a)

- [最全面的data augmentation讲解 - 2020](https://mp.weixin.qq.com/s/QkWO-lmZED4sSZQArWsmkg)

    本文更关注利用预训练语言模型来完成数据增强

- 【Great】[Lessons Learned from Applying Deep Learning for NLP Without Big Data - 2018](https://towardsdatascience.com/lessons-learned-from-applying-deep-learning-for-nlp-without-big-data-d470db4f27bf)

    数据增强，同义词替换，等

    **YAO**: 处理所有Data都可以进行这些处理！

- [Automating Data Augmentation: Practice, Theory and New Direction - Stanford AI Lab 2020](https://ai.stanford.edu/blog/data-augmentation/)

    **Chinese**: [自动化数据增强：实践、理论和新方向](https://mp.weixin.qq.com/s/K7aYSuROGgWqaVrC8TaqmQ)

    Automating the Art of Data Augmentation:
    
    - [Part I Overview](https://hazyresearch.stanford.edu/data-aug-part-1)
    - [Part II Practical Methods](https://hazyresearch.stanford.edu/data-aug-part-2)
    - [Part III Theory](https://hazyresearch.stanford.edu/data-aug-part-3)
    - [Part IV New Direction](https://hazyresearch.stanford.edu/data-aug-part-4)

- [文本数据增强 - 2019](https://mp.weixin.qq.com/s/aIydEKcDYWNaczMUi07tIQ)

- [文本数据增强：撬动深度少样本学习模型的性能 - 2020](https://mp.weixin.qq.com/s/LUMYqN-BVBKXMoHRMCyHpA)


## 2.6 Data Labeling

数据标注，训练数据生成等

### 2.6.1 Overview

#### Article


## 2.6.2 Active Learning

主动学习

#### Article

- [主动学习（Active Learning）-少标签数据学习 - 2019](https://zhuanlan.zhihu.com/p/79764678)

- [Active Learning in Machine Learning - 2020](https://towardsdatascience.com/active-learning-in-machine-learning-525e61be16e5)

- [主动学习：标注的一颗救心丸 - 2020](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650412415&idx=3&sn=f4025488aacbb7fe4ae93952917621af)

- [主动学习-Active Learning：如何减少标注代价 - 2019](https://zhuanlan.zhihu.com/p/39367595)

- [【01】主动学习-Active Learning：如何减少标注代价 - 2019](https://zhuanlan.zhihu.com/p/39367595)


## 2.6.3 Confidence Learning

置信学习

#### Article

- [如何判断样本标注的靠谱程度？置信度学习（CL）简述 - 2020](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247485618&idx=1&sn=7016ad51dab33771d51ae7c2d5c8223f)

- [别让数据坑了你！用置信学习找出错误标注（附开源实现）- 2020](https://mp.weixin.qq.com/s/PyyPMsdKaeoZ_3OiSiSBwg)