# 4. Basic Problem

## 4.1 Overview


## 4.2 Vocabulary & OOV

#### Article

- [非主流自然语言处理：大规模语料词库自动生成 - 2017](http://www.sohu.com/a/157426068_609569)

- [word2vec缺少单词怎么办？](https://www.zhihu.com/question/329708785)


## 4.3 Segmentation

#### Library

- 【Great】<https://github.com/lancopku/PKUSeg-python>

    中文分词工具包，准确度远超 Jieba

#### Paper

- [A Simple and Effective Neural Model for Joint Word Segmentation and POS Tagging - HLJU2018]()

    **Code**: <https://github.com/bamtercelboo/pytorch_Joint-Word-Segmentation-and-POS-Tagging> (PyTorch)

    **Code**: <https://github.com/zhangmeishan/NNTranJSTagger> (C++)

- [Effective Neural Solution for Multi-Criteria Word Segmentation - Canada2018](https://arxiv.org/abs/1712.02856)

    **Code**: <https://github.com/hankcs/multi-criteria-cws> (Dynet)

#### Practice

- <https://github.com/FanhuaandLuomu/BiLstm_CNN_CRF_CWS> (Tensorflow & Keras)

    BiLstm+CNN+CRF 法律文档（合同类案件）领域分词   [中文解读](https://www.jianshu.com/p/373ce87e6f32)

- <https://github.com/liweimin1996/CwsPosNerCNNRNNLSTM>

    基于字向量的CNN池化双向BiLSTM与CRF模型的网络，可能一体化的完成中文和英文分词，词性标注，实体识别。主要包括原始文本数据，数据转换,训练脚本,预训练模型,可用于序列标注研究.

#### Article

- [深度学习时代，分词真的有必要吗 - 2019](https://zhuanlan.zhihu.com/p/66155616)

- [从why到how的中文分词详解，从算法原理到开源工具 - 2019](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247488211&idx=2&sn=3adb96ff316dfc67663f72a9595eda5d)


## 4.4 Dependency Parsing

#### Practice

- <https://github.com/Samurais/text-dependency-parser>

    依存关系分析


## 4.5 Subword

### 4.5.1 Overview

#### Article

- [子词技巧：The Tricks of Subword - 2019](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650411766&idx=3&sn=c5f92645737b469d386bf303bcbcf71f)

- [深入理解NLP Subword算法：BPE、WordPiece、ULM - 2020](https://zhuanlan.zhihu.com/p/86965595)


### 4.5.2 WordPiece


### 4.5.3 SentencePiece

#### Code

- <https://github.com/google/sentencepiece>

    SentencePiece is an unsupervised text tokenizer and implements subword units (BPE) and unigram language model with the extension of direct training from raw sentences.

- <https://github.com/yoheikikuta/bert-japanese>

    BERT with SentencePiece for Japanese text.

- <https://github.com/wannaphong/thai-word-segmentation-sentencepiece>

    thai word segmentation using sentencepiece

- <https://github.com/bheinzerling/bpemb>

    A collection of pre-trained subword embeddings in 275 languages, based on BPE and trained on Wikipedia. It can be as input for neural models in NLP.


## 4.6 Phrase Mining & Keyword Extraction

#### Article

- 【Great】[谈谈医疗健康领域的Phrase Mining - 2020](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650412407&idx=2&sn=2e96b4456afb1a208aab475ebf9fa1b8)

- 【Great】医疗健康领域的短文本解析探索 - 2020: [一](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650412504&idx=2&sn=09f7f5783a48c009744997714b52eec4) and [二](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650412504&idx=3&sn=37f03898a549e89e1da0a093b67e9925)

- 【Great】[知识图谱如何应用到文本标签化算法中 - 2020](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650412450&idx=2&sn=3c01ae8525dbab5202433b19fc0cc5db)

#### Practice

- <https://github.com/bigzhao/Keyword_Extraction>

    神策杯2018高校算法大师赛（中文关键词提取）第二名代码方案


## 4.7 New Word

#### Practice

- <https://github.com/xylander23/New-Word-Detection>

    新词发现算法(NewWordDetection)

- <https://github.com/zhanzecheng/Chinese_segment_augment>

    python3实现互信息和左右熵的新词发现
    
- <https://github.com/Rayarrow/New-Word-Discovery>

    新词发现 基于词频、凝聚系数和左右邻接信息熵


## 4.9 Disambiguation

词语纠错和语义消歧

**YAO**: 特别重要！几乎可以作为所有 NLP 任务最最开始时的处理！比分词还要靠前！

#### Practice

- 【Great】<https://github.com/taozhijiang/chinese_correct_wsd>

    简易的中文纠错和消歧

- <https://github.com/beyondacm/Autochecker4Chinese>

    中文文本错别字检测以及自动纠错 / Autochecker & autocorrecter for chinese

- <https://github.com/liuhuanyong/WordMultiSenseDisambiguation>

    基于百科知识库的中文词语多词义/义项获取与特定句子词语语义消歧

- <https://github.com/ccheng16/correction>

    Chinese "spelling" error correction

- <https://github.com/apanly/proofreadv1>

    中文文本自动纠错

#### Article

- [中文文本纠错算法--错别字纠正的二三事 - 2019](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650411874&idx=3&sn=f78fa6e6ba3493086503cbb7d11a7ff2)

    **YAO**: 对当前一些Library/Tool的概述和测试

    主要技术：错别字词典，编辑距离，语言模型(NGram, DNN)

    三个关键点：分词质量，领域相关词典质量，语言模型的种类和质量

    语言模型：计算序列的合理性；
    
    规则约束：优先选择拼音相同的候选字词，其次是形似的；纠正后分词个数更少；


## 4.10 Synonym

#### Article

- [如何扩充知识图谱中的同义词 - 2019](https://mp.weixin.qq.com/s?__biz=MzUyMDY0OTg3Nw==&mid=2247484011&idx=1&sn=8f6e6ae9e3d34b7a1dc2cd55812f55ca)

    同义词挖掘，以医学术语为例

