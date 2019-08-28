# 4. Basic Problem

## 4.1 Overview


## 4.2 Vocabulary


#### Article

- [非主流自然语言处理：大规模语料词库自动生成 - 2017](http://www.sohu.com/a/157426068_609569)


## 4.3 Out of Vocabulary (OOV)

#### Article

- [word2vec缺少单词怎么办？](https://www.zhihu.com/question/329708785)



## 4.4 Segmentation

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


## 4.5 Dependency Parsing

#### Practice

- <https://github.com/Samurais/text-dependency-parser>

    依存关系分析


## 4.6 Small Data

#### Article

- 【Great!!!】[Lessons Learned from Applying Deep Learning for NLP Without Big Data - 2018](https://towardsdatascience.com/lessons-learned-from-applying-deep-learning-for-nlp-without-big-data-d470db4f27bf)

    数据增强，同义词替换，等

    **YAO**: 虽然针对Small Data，但我觉得处理所有Data时都可以进行这些处理！


## 4.7 New Word

#### Practice

- <https://github.com/xylander23/New-Word-Detection>

    新词发现算法(NewWordDetection)

- <https://github.com/zhanzecheng/Chinese_segment_augment>

    python3实现互信息和左右熵的新词发现
    
- <https://github.com/Rayarrow/New-Word-Discovery>

    新词发现 基于词频、凝聚系数和左右邻接信息熵


## 4.8 Disambiguation

词语纠错和语义消歧

#### Practice

- 【Great!!!】<https://github.com/taozhijiang/chinese_correct_wsd>

    简易的中文纠错和消歧

    **YAO**: 特别重要！几乎可以作为所有 NLP 任务最最开始时的处理！比分词还要靠前！

- <https://github.com/beyondacm/Autochecker4Chinese>

    中文文本错别字检测以及自动纠错 / Autochecker & autocorrecter for chinese

- <https://github.com/liuhuanyong/WordMultiSenseDisambiguation>

    基于百科知识库的中文词语多词义/义项获取与特定句子词语语义消歧

- <https://github.com/ccheng16/correction>

    Chinese "spelling" error correction

- <https://github.com/apanly/proofreadv1>

    中文文本自动纠错


## 4.9 打造NLP数据集

#### Article

- [如何打造高质量的机器学习数据集？ - 2019](https://www.zhihu.com/question/333074061/answer/773825458)

