# 16. Knowledge Graph

## 16.1 Overview

#### Paper

- [Learning Entity and Relation Embeddings for Knowledge Graph Completion - THU2015](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/aaai2015_transr.pdf)

- [COMET: Commonsense Transformers for Automatic Knowledge Graph Construction - Allen2019](https://arxiv.org/abs/1906.05317)

    **Chinese**: [ACL 2019 | AI2等提出自动知识图谱构建模型COMET，接近人类表现](https://mp.weixin.qq.com/s/TKGQxPBA1XVNxR4nVtl8mg)

- [How much is a Triple? Estimating the Cost of Knowledge Graph Creation - Germany2018](http://ceur-ws.org/Vol-2180/ISWC_2018_Outrageous_Ideas_paper_10.pdf)

    **Chinese**: [67 亿美金搞个图，创建知识图谱的成本有多高你知道吗？](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650757216&idx=1&sn=aea53429d390c783a7bf2a961da05c63)

- <https://github.com/liuhuanyong/KnowledgeGraphSlides>

    中文知识图谱计算会议CCKS报告合集，涵盖从2013年至2018年，共48篇，从中可以看出从谷歌2012年推出知识图谱以来，中国学术界及工业界这6年来知识图谱的主流思想变迁。

- [Distant supervision for relation extraction without labeled data - Stanford2009](http://www.stanfordlibrary.us/~jurafsky/mintz.pdf)

    远程监督 = 监督学习 + Bootstrapping

    **Article**: [关系抽取之远程监督算法 - 2019](https://www.cnblogs.com/Luv-GEM/p/11598294.html)

#### Data

- <https://www.ownthink.com/>

    思知 智能时代：史上最大规模1.4亿中文知识图谱开源下载

    **Code**:<https://github.com/ownthink/KnowledgeGraphData>


#### Code

- <https://github.com/thunlp/KB2E> (C++)

- <https://github.com/wuxiyu/transE> (Python)

- <https://github.com/liuhuanyong/CausalityEventExtraction>

    基于因果关系知识库的因果事件图谱实验项目，本项目罗列了因果显式表达的几种模式，基于这种模式和大规模语料，再经过融合等操作，可形成因果事件图谱。

- <https://github.com/liuhuanyong/HyponymyExtraction>

    基于知识概念体系，百科知识库，以及在线搜索结构化方式的词语上下位抽取与可视化展示


#### Competition

**瑞金医院MMC人工智能辅助构建知识图谱大赛**

- <https://github.com/zhpmatrix/tianchi-ruijin> (Keras)

    baseline, BiLSTM+CRF, 只做了第一阶段实体识别，作为NER来做，第二阶段是关系抽取，其实可以End-to-End一起做


**2019百度信息抽取比赛**

> 抽取满足约束的SPO三元组知识, <http://lic2019.ccf.org.cn/kg>

- <https://github.com/bojone/kg-2019> (Keras)

    Rank 7   基于CNN + Attenton + 自行设计的标注结构的信息抽取模型

- <https://github.com/zhengyima/kg-baseline-pytorch> (PyTorch)

    使用Pytorch实现苏神的模型，F1在dev集可达到0.75，联合关系抽取，Joint Relation Extraction.

- <https://github.com/wangpeiyi9979/IE-Bert-CNN> (PyTorch)

    BERT + CNN


#### Article

- [知识图谱从哪里来：实体关系抽取的现状与未来 - 2019](https://zhuanlan.zhihu.com/p/91762831)


## 16.2 Representation Learning

- <https://github.com/chenbjin/RepresentationLearning>

    知识表示相关学习算法

- <https://github.com/liuhuanyong/ProductKnowledgeGraph>

    基于京东网站的商品上下级概念、商品品牌之间关系和商品描述维度等知识库，可以支持商品属性库构建、商品销售问答、品牌物品生产等知识查询服务，也可用于情感分析等下游应用。


## 16.3 Entity

- <https://github.com/hendrydong/Chinese-Event-Extraction>

    ?


## 16.4 Relation

#### Practice

- <https://github.com/thunlp/JointNRE> (Tensorflow)

    Joint Neural Relation Extraction with Text and KGs

    **Paper**: 1

- <https://github.com/shiningliang/CCKS2019-IPRE> (Tensorflow)

    CCKS2019-人物关系抽取

- <https://github.com/mengxiaoxu/entity_relation_extraction> (Java)

    基于依存分析的实体关系抽取简单实现，即抽取事实三元组

- <https://github.com/lawlietAi/relation-classification-via-attention-model> (PyTorch)

    code of Relation Classification via Multi-Level Attention CNNs

- <https://github.com/hadyelsahar/CNN-RelationExtraction> (Tensorflow)

    CNN for relation extraction between two given entities

- <https://github.com/lemonhu/open-entity-relation-extraction>

    Knowledge triples extraction and knowledge base construction based on dependency syntax for open domain text.

#### Paper

- 1. [Neural Knowledge Acquisition via Mutual Attention between Knowledge Graph and Text - THU2018](https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/16691/16013)

- 2. [Chinese Relation Extraction with Multi-Grained Information and External Linguistic Knowledge - THU2019](http://nlp.csai.tsinghua.edu.cn/~lzy/publications/acl2019_nre4chinese.pdf)

    **Code**: <https://github.com/thunlp/Chinese_NRE> (PyTorch)

#### Article

- [关系提取简述 - 2019](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650412040&idx=1&sn=8d580639fa1b65d948a187ac41340087)

- 【论文】Awesome Relation Classification Paper - 2019

    [PART I](https://blog.csdn.net/Kaiyuan_sjtu/article/details/89877420)

    [PART II](https://blog.csdn.net/Kaiyuan_sjtu/article/details/89961647)

- 【论文】Awesome Relation Extraction Paper - 2019

    [PART I](https://blog.csdn.net/Kaiyuan_sjtu/article/details/89961674)

    [PART II](https://blog.csdn.net/Kaiyuan_sjtu/article/details/90071703)


## 16.5 End-to-end

#### Practice

- <https://github.com/gswycf/Joint-Extraction-of-Entities-and-Relations-Based-on-a-Novel-Tagging-Scheme> (PyTorch)

    Joint Extraction of Entities and Relations Based on cnn+rnn

    **Paper**: paper1, 2

- <https://github.com/zsctju/triplets-extraction> (Keras)

    Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme

    **Paper**: paper1

- <https://github.com/WindChimeRan/pytorch_multi_head_selection_re> (PyTorch)

    reproduce "Joint entity recognition and relation extraction as a multi-head selection problem" for Chinese IE

    **Paper**: paper3

- <https://github.com/sanmusunrise/NPNs>

    **Paper**: paper4

#### Paper

- 1. [Joint Extraction of Entities and Relations Based on a Novel Tagging Scheme - CAS2017](https://arxiv.org/abs/1706.05075)

- 2. [Deep Active Learning for Named Entity Recognition - UTEXAS2018](https://arxiv.org/abs/1707.05928)

- 3. [Joint entity recognition and relation extraction as a multi-head selection problem - Belgium2018](https://arxiv.org/abs/1804.07847)

- 4. [Nugget Proposal Networks for Chinese Event Detection - CAS2018](https://arxiv.org/abs/1805.00249)

- 5. [ointly Multiple Events Extraction via Attention-based Graph Information Aggregation - BIT2018](https://arxiv.org/abs/1809.09078)
