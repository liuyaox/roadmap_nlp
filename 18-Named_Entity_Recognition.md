# 18. Named Entity Recognition (NER)

YAO's: <https://github.com/liuyaox/named_entity_recognition> (Keras)

## 18.1 Overview

#### Paper

- 【Great】[零资源跨领域命名实体识别 - 2020](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247491946&idx=3&sn=e7ec58926b860fe09d93a298698d09a8)

#### Article

- 【Great】[Named entity recognition serie - 2018](https://www.depends-on-the-definition.com/introduction-named-entity-recognition-python/) (Keras)

    包括NER各种模型：CRF, Seq2Seq, LSTM+CRF, LSTM + Char Embedding, Residual LSTM + ELMo, NER with Bert

    **YAO**: 

- [浅析深度学习在实体识别和关系抽取中的应用 - 2017](https://blog.csdn.net/u013709270/article/details/78944538)

- 【Great】[命名实体识别 NER 论文综述：那些年，我们一起追过的却仍未知道的花名(一) - 2020](https://mp.weixin.qq.com/s/_V9dDurEIi1BG14yjE9LfQ)

- [序列标注的方法中有多种标注方式BIO、BIOSE、IOB、BILOU、BMEWO的异同 - 2020](https://mp.weixin.qq.com/s/Cpn87G1JBFYWVHmY5Mqp5A)

- [流水的NLP铁打的NER：命名实体识别实践与探索 - 2020](https://mp.weixin.qq.com/s/0cW4SXqEGcL8F-zj1zGABA)

- [工业界求解NER问题的12条黄金法则 - 2020](https://mp.weixin.qq.com/s/TmA4BcxPPz94gwknCBe69Q)


#### Practice

- 【Great】[NLP命名实体识别(NER)开源实战教程 - 2019](https://blog.csdn.net/xiaosongshine/article/details/99622170) (Keras)

    介绍模型：BiLSTM + CRF, IDCNN + CRF    实现模型：BiLSTM (无CRF)

    **YAO**: 

- 【Great】<https://github.com/qiufengyuyi/sequence_tagging>
  
    **Article**: [中文NER任务实验小结报告——深入模型实现细节](https://zhuanlan.zhihu.com/p/103779616)

- [美团搜索中NER技术的探索与实践 - 2020](https://mp.weixin.qq.com/s/o5X8ck30zPprEZptbS4HUw)


#### Competition

**[2019 “达观杯”文本智能信息抽取挑战赛](https://www.biendata.com/competition/datagrand/)**

> 信息抽取：抽取出特定的事件或事实信息，帮助我们将海量内容自动分类、提取和重构

- [冠亚季军分享：预训练模型彻底改变了NLP，但也不能忽略传统方法带来的提升 - 2019](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651674340&idx=2&sn=9a7b74e461a0d716ba150798f3f5f597)

    冠军：GloVe + Pretrained(Flair/ELMo/BERT/XLNet) + LSTM + CNN + CRF   
    亚军：Bert + BiLSTM + CRF   
    季军：Transformer + BiLSTM + CRF

- 【Great】<https://github.com/cdjasonj/datagrand> (Keras & Tensorflow)

    Rank 6
    
    输入：Embedding有2*3+1=7种：(char, bichar) * (Word2Vec, GloVe, FastText) + \<char, ELMo>，每种有4种维度dim=(150, 200, 250, 300)
    
    中间：
    - 模型1：BiLSTMs + SelfAttention  
    - 模型2：BiLSTM + CNNs(kernel: 3, 5, 7) + SelfAttention  
    - 模型3：(BiGRU + LocalAttention)s  
    - 模型4：BiONLSTMs + SelfAttention (未使用)  
    
    输出：TimeDistributed(Dense) + CRF

    **YAO**: [Detailed Notes](https://github.com/liuyaox/forked_repos_with_notes/tree/master/datagrand-master)

    - 特点1：**数据不Padding！也没有使用Masking！** 按Batch训练模型，每一Batch内输入具有相同的Seq_length；应用时同样，相同Seq_length的输入一并应用。

    - 特点2：使用ELMo，且融合了Word2Vec, GloVe, FastText三种静态Embedding的特点。

    HERE HERE HERE HERE HERE

- <https://github.com/lonePatient/daguan_2019_rank9> (Pytorch)

    Rank 9
    
    模型1：BERT + LSTM + CRF  
    模型2：BERT + LSTM + MDP + CRF  
    模型3：BERT + LSTM + SPAN 

- 【Great】<https://github.com/renjunxiang/daguan_2019> (PyTorch)

    **自行训练BERT**，而非直接使用训练好的BERT，过程详细值得学习


#### Data

- <https://github.com/LG-1/video_music_book_datasets>

    9632条视频/音乐/书籍标注数据


## 18.2 HMM & CRF


## 18.3 RNN + CRF

主要以 BiLSTM + CRF 为主

关于CRF和Loss推导和实现细节，以及解码细节，请参考：[04_Probabilistic_Graphical_Model](https://github.com/liuyaox/Machine_Learning_Awesome/blob/master/04_Probabilistic_Graphical_Model.md)


#### Paper

- [Neural Architectures for Named Entity Recognition - CMU2016](https://arxiv.org/abs/1603.01360)

    BiLSTM + CRF

- HSCRF: [Hybrid semi-Markov CRF for Neural Sequence Labeling - USTC2018](https://arxiv.org/abs/1805.03838)

    **Code**: <https://github.com/ZhixiuYe/HSCRF-pytorch> (PyTorch)


#### Practice

##### Keras

- <https://github.com/stephen-v/zh-NER-keras> (Keras)

    **Chinese**：[基于keras的BiLstm与CRF实现命名实体标注 - 2018](https://www.cnblogs.com/vipyoumay/p/ner-chinese-keras.html)

    **YAO**: OK
    
    - 基本信息：中文实体  **基于字，采用BIO标注集**，实体有Person/Location/Organization，则tags共有3*2+1=7个，模型结构: Embedding -> BiLSTM -> CRF

    - 数据处理：对于X，as normal，向量化编码，补零截断；对于Y，向量化编码(不同tag转化为0-6)，随后也要**补零截断**！注意，Padding的mask_value，X与Y要相同（待研究？）

    - Library版本：当(tensorflow=1.10.0, keras=2.2.0, keras-contrib=0.0.2)时CRF没问题

- <https://github.com/UmasouTTT/keras_bert_ner> (Keras)

    中文实体，模型结构：BERT + BiLSTM + CRF，数据和其他信息同上(zh-NER-keras)

    **YAO**: OK  [Detailed Notes](https://github.com/liuyaox/forked_repos_with_notes/tree/master/keras_bert_ner-master)
    
    使用keras-bert，基于chinese_L-12_H-768_A-12加载BERT模型，直接融入模型并且参与训练(Finetuning)

- 【Great】<https://github.com/AidenHuen/BERT-BiLSTM-CRF> (Keras)

    BERT-BiLSTM-CRF的Keras版实现  预训练模型为chinese_L-12_H-768_A-12.zip，使用BERT客户端和服务器bert-serving-server和bert-serving-client

    **YAO**: HERE HERE HERE HERE HERE

##### PyTorch

- <https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling> (PyTorch)

    模型结构：Language Model + BiLSTM + CRF，使用了Highway Networks

    **Article**: [一文完全搞懂序列标注算法](https://mp.weixin.qq.com/s?__biz=MzU0MDQ1NjAzNg==&mid=2247488405&idx=1&sn=c7fc862082f6e6d432912b2d70d5b5ee)

- <https://github.com/fangwater/Medical-named-entity-recognition-for-ccks2017> (PyTorch)

    A LSTM+CRF model for the seq2seq task for Medical named entity recognition in ccks2017

    **YAO**: PyTorch实现的CRF

- <https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Sequence-Labeling> (PyTorch)

    LM + BiLSTM + CRF

    **Artichle**: [NLP | 一文完全搞懂序列标注算法](https://mp.weixin.qq.com/s/RsPTqmtp4tBTsTaKH0-5YA)

- <https://github.com/llcing/BiLSTM-CRF-ChineseNER.pytorch> (PyTorch)

    PyTorch implement of BiLSTM-CRF for Chinese NER

- <https://github.com/yanwii/ChinsesNER-pytorch> (PyTorch)

    基于 BiLSTM + CRF 的中文命名实体识别

- <https://github.com/chenxiaoyouyou/Bert-BiLSTM-CRF-pytorch> (PyTorch)

    基于BERT做字嵌入的BiLSTM-CRF序列标注模型

- [Pytorch BiLSTM + CRF做NER - 2019](https://zhuanlan.zhihu.com/p/59845590) (PyTorch)

- [如何使用BERT来做命名实体识别 - 2019](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247490099&idx=3&sn=8416ee9aeb0453e0b1de67abb057f0a0)

- [NLP实战-中文命名实体识别 - 2019](https://zhuanlan.zhihu.com/p/61227299) (PyTorch)

##### Tensorflow

- <https://github.com/phychaos/transformer_crf> (Tensorflow)

    Transformer + CRF

- <https://github.com/Determined22/zh-NER-TF> (Tensorflow)

    A very simple BiLSTM-CRF model for Chinese Named Entity Recognition 中文命名实体识别
    
    **Article**: [序列标注：BiLSTM-CRF模型做基于字的中文命名实体识别 - 2017](https://blog.csdn.net/liangjiubujiu/article/details/79674606)
    
- <https://github.com/shiyybua/NER> (Tensorflow)

    BiRNN + CRF

    **Article**: [基于深度学习的命名实体识别详解 - 2017](https://zhuanlan.zhihu.com/p/29412214)

- <https://github.com/pumpkinduo/KnowledgeGraph_NER> (Tensorflow)

    中文医学知识图谱命名实体识别，模型有：BiLSTM+CRF, Transformer+CRF

- 【Great】<https://github.com/baiyyang/medical-entity-recognition> (Tensorflow)

    包含传统的基于统计模型(CRF)和基于深度学习(Embedding-Bi-LSTM-CRF)下的医疗数据命名实体识别

- <https://github.com/Nrgeup/chinese_semantic_role_labeling> (Tensorflow)

    基于 Bi-LSTM 和 CRF 的中文语义角色标注

- <https://github.com/dkarunakaran/entity_recoginition_deep_learning> (Tensorflow)

    **Article**: [Entity extraction using Deep Learning based on Guillaume Genthial work on NER - 2018](https://medium.com/intro-to-artificial-intelligence/entity-extraction-using-deep-learning-8014acac6bb8)


#### Article

- 【Great】[CRF Layer on the Top of BiLSTM 1-8 - 2017](https://github.com/createmomo/CRF-Layer-on-the-Top-of-BiLSTM) (Chainer)

    **YAO**: 详细介绍Emission Score和Transition Score，以及Path Score, All Path Score和Loss

- [bi-LSTM + CRF with character embeddings for NER and POS - 2017](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)
    
    **Code**: <https://github.com/guillaumegenthial/tf_ner> (Tensorflow)

    **Chinese**: [命名实体识别（biLSTM+crf）](https://blog.csdn.net/xxzhix/article/details/81514040)

- [如何理解LSTM后接 CRF？](https://www.zhihu.com/question/62399257)

    **YAO**: 

    接CRF是为了Model label sequence **jointly**, instead of decoding each label independently.

    CRF所需要的各种特征：转移概率矩阵G，直接是模型待学习的参数；发射概率矩阵H，由模型编码部分(如LSTM)完成对X的编码

- [CRF 和 LSTM 模型在序列标注上的优劣？](https://www.zhihu.com/question/46688107)


## 18.4 CNN

- <https://github.com/nlpdz/Medical-Named-Entity-Rec-Based-on-Dilated-CNN>

    基于膨胀卷积神经网络（Dilated Convolutions）训练好的医疗命名实体识别工具


## 18.5 Others



#### Article

- [用腻了 CRF，试试 LAN 吧 - Xihu2019](https://zhuanlan.zhihu.com/p/91031332)

    LAN: 逐层改进的基于标签注意力机制的网络

    **Code**: <https://github.com/Nealcly/BiLSTM-LAN> (PyTorch)


