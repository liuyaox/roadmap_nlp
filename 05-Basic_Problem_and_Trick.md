

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


## 4.5 Dependency Parsing


## 4.6 Small Data

#### Article

- 【Great!!!】[Lessons Learned from Applying Deep Learning for NLP Without Big Data - 2018](https://towardsdatascience.com/lessons-learned-from-applying-deep-learning-for-nlp-without-big-data-d470db4f27bf)

    数据增强，同义词替换，等

    **YAO**: 虽然针对Small Data，但我觉得处理所有Data时都可以进行这些处理！


## 4.7 New Word

#### Practice

- <https://github.com/xylander23/New-Word-Detection>

    新词发现算法(NewWordDetection)