
# 7. Pretrained Model

## 7.1 Overview

#### Paper

- 【Great】<https://github.com/thunlp/PLMpapers>

    Must-read Papers on pre-trained language models  各预训练模型层次衍生关系图

#### Article

- [8 Excellent Pretrained Models to get you Started with NLP](https://www.analyticsvidhya.com/blog/2019/03/pretrained-models-get-started-nlp/)

    包括：ULMFiT, Transformer, BERT, Transformer-XL, GPT-2, ELMo, Flair, StanfordNLP

    **Chinese**：[8种优秀预训练模型大盘点，NLP应用so easy！](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651669109&idx=2&sn=29b4e45291eac659af2967a1e246aa03)

- [Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html)

    包括：CoVe, ELMo, Croww-View Training, ULMFiT, GPT, BERT, GPT-2

    **Chinese**：[上下文预训练模型最全整理：原理、应用、开源代码、数据分享](https://mp.weixin.qq.com/s?__biz=MzIxNDgzNDg3NQ==&mid=2247485551&idx=1&sn=de0a04647870543fe0b36d024f58429e)

- [Language Models and Contextualised Word Embeddings](http://www.davidsbatista.net/blog/2018/12/06/Word_Embeddings/)
  
    对 ELMo, BERT 及其他模型进行了一个简单的综述

- [从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史 - 张俊林](https://zhuanlan.zhihu.com/p/49271699)

- [NLP's ImageNet moment has arrived - 2018](https://thegradient.pub/nlp-imagenet/)

    词嵌入已死，语言模型当立

- [nlp中的预训练语言模型总结(单向模型、BERT系列模型、XLNet) - 2019](https://zhuanlan.zhihu.com/p/76912493)

- [NLP的游戏规则从此改写？从word2vec, ELMo到BERT - 2018](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247484985&idx=1&sn=30075de882e862081f8d8d972f150a70)

- [就最近看的paper谈谈预训练语言模型发展 - 2019](https://zhuanlan.zhihu.com/p/79371603)

- [李宏毅-ELMO、BERT、GPT视频笔记 - 2019](https://mp.weixin.qq.com/s?__biz=MzIwODI2NDkxNQ==&mid=2247484834&idx=3&sn=0951ac8c768ad7e078754f8baba8e65c)

    **Video**: <https://www.bilibili.com/video/av46561029/?p=61>

#### Code

- <https://github.com/PaddlePaddle/LARK> (PaddlePaddle)

    BERT, EMLo and ERNIE Implementation with PaddlePaddle

#### Practice

- [北大、人大联合开源工具箱UER，3 行代码完美复现BERT、GPT - 2019](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247490122&idx=3&sn=a2413923ce3e620f26a00edb4d89d878)

#### Library

- 【Great】<https://github.com/huggingface/pytorch-transformers> (PyTorch)

    A library of SOTA pretrained models for NLP

    包含8个主流预训练模型(BERT, OpenAIGPT, GPT2, TransfoXL, XLNet, XLM, RoBERTa, OpenCLaP)，提供整套API：Tokenize, 转化为字符的ID, 计算隐藏向量表征等

    **Chinese**: [BERT、GPT-2这些顶尖工具到底该怎么用到我的模型里？ - 2019](https://baijiahao.baidu.com/s?id=1626146900426049013)

- <https://github.com/zalandoresearch/flair> (PyTorch)

    Flair: 混合了BERT, EMLo, GPT-2

    **Article**: [Text Classification with State of the Art NLP Library — Flair - 2018](https://towardsdatascience.com/text-classification-with-state-of-the-art-nlp-library-flair-b541d7add21f)

    **Chinese**: [简单易用NLP框架Flair发布新版本 - 2018](https://www.jiqizhixin.com/articles/2018-12-27-12)


## 7.2 Chinese

中文预训练语言模型

#### Code

- <https://github.com/brightmart/roberta_zh> (Tensorflow)

    RoBERTa for Chinese   目前只是base版训练数据：10G文本，包含新闻、社区问答、百科数据等

    **Article**: [RoBERTa中文预训练模型，你离中文任务的SOTA只差个它](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650769391&idx=2&sn=b5f90a2a60c7929469f622db37ef4b1e)

- <https://github.com/thunlp/OpenCLaP> (PyTorch)

    OpenCLaP：多领域开源中文预训练语言模型仓库   可被PyTorch-Transformers直接使用   包含：百度百科ERT, 民事文书BERT, 刑事文书BERT

- 【Great】<https://github.com/ymcui/Chinese-BERT-wwm> (Tensorflow & PyTorch)

    Pre-Training with Whole Word Masking for Chinese BERT（中文BERT-wwm预训练模型）
  
    **Paper**: [Pre-Training with Whole Word Masking for Chinese BERT - HIT2019](https://arxiv.org/abs/1906.08101)

    **Article**: [中文最佳，哈工大讯飞联合发布全词覆盖中文BERT预训练模型](https://mp.weixin.qq.com/s/88OwaHqnrVMQ7vH98INA3w)

#### Practice

- <https://github.com/cdj0311/keras_bert_classification> (Keras)

    使用 chinese_L-12_H-768_A-12，模型为BERT + FC/LSTM

- 【Great】<https://github.com/songyingxin/bert-textclassification> (PyTorch)

    Implemention some Baseline Model upon Bert for Text Classification

- <https://github.com/YC-wind/embedding_study> (Tensorflow)

    中文预训练模型生成字向量学习，测试BERT，ELMO的中文效果

- <https://github.com/renxingkai/BERT_Chinese_Classification> (Tensorflow)

    用BERT进行中文情感分类的详细操作及完整程序


## 7.3 EMLo - TOTODO

EMLo 是第一个使用预训练模型进行词嵌入的方法，将句子输入ELMO，可以得到句子中每个词的向量表示。

[Deep contextualized word representations - AllenAI2018](https://arxiv.org/abs/1802.05365)

#### Code

- <https://github.com/allenai/allennlp> (PyTorch)

- <https://github.com/allenai/bilm-tf> (Tensorflow)

#### Article

- [ELMO模型(Deep contextualized word representation)](https://www.cnblogs.com/jiangxinyang/p/10060887.html)

- [对 ELMo 的视频介绍](https://vimeo.com/277672840)

#### Practice

- [文本分类实战（九）—— ELMO 预训练模型](https://www.cnblogs.com/jiangxinyang/p/10235054.html)

- [A Step-by-Step NLP Guide to Learn ELMo for Extracting Features from Text - 2019](https://www.analyticsvidhya.com/blog/2019/03/learn-to-use-elmo-to-extract-features-from-text/)

    NLP详细教程：手把手教你用ELMo模型提取文本特征


## 7.4 BERT - TOTODO

### 7.4.1 Overview

#### Article

- [8篇论文梳理BERT相关模型进展与反思 - 2019](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651673864&idx=4&sn=703a1271f3cd40afe85130c80df90cc9)

- 站在BERT肩膀上的NLP新秀们 - 2019
  
  - [PART I](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247489437&idx=4&sn=d1d7ca7e3b4b0a1710252e8d52affe4d)
  
    给 BERT 模型增加外部知识信息，使其能更好地感知真实世界，主要讲了 ERNIE from Baidu 和 ERNIE from THU

  - [PART II](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650409996&idx=1&sn=ddf837339e50001be4514fee743bfe9d)
    
    主要讲了 XLMs from Facebook, LASER from Facebook, MASS from Microsoft 和 UNILM from Microsoft

  - [PART III](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650410110&idx=1&sn=310f675cf0cc1e2a1f4cc7b919743bc4)
    
    主要看看预训练模型中的增强训练（多任务学习/数据增强）以及BERT多模态应用： MT-DNN from Microsoft, MT-DNN-2 from Microsoft, GPT-2 from OpenAI 和 VideoBERT from Google

- [BERT 瘦身之路：Distillation，Quantization，Pruning - 2019](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247490372&idx=7&sn=7fb9c5060796f3f9a92c3f817afc080f)

- [Understanding searches better than ever before - 2019](https://blog.google/products/search/search-language-understanding-bert)

    **Chinese**: [谷歌搜索用上BERT，10%搜索结果将改善](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650772610&idx=2&sn=8770bdfbf950b3651910488722f6873d)


### 7.4.2 BERT

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding - Google2019](https://arxiv.org/abs/1810.04805)

#### Code

- <https://github.com/google-research/bert> (Tensorflow)

#### Library

- <https://github.com/CyberZHG/keras-bert> (Keras)

- <https://github.com/codertimo/BERT-pytorch> (PyTorch)

#### Article

- 编码器: [Dissecting BERT Part 1: The Encoder](https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3)

- 网络结构: [Understanding BERT Part 2: BERT Specifics](https://medium.com/dissecting-bert/dissecting-bert-part2-335ff2ed9c73)

- 解码器: [Dissecting BERT Appendix: The Decoder](https://medium.com/dissecting-bert/dissecting-bert-appendix-the-decoder-3b86f66b0e5f)

- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/)

- [彻底搞懂BERT](https://www.cnblogs.com/rucwxb/p/10277217.html)

- [理解BERT每一层都学到了什么 - 2019](https://zhuanlan.zhihu.com/p/74515580)

- [关于最近实践 Bert 的一些坑 - 2019](https://zhuanlan.zhihu.com/p/69389583)

- [BERT fintune 的艺术 - 2019](https://zhuanlan.zhihu.com/p/62642374)

#### Practice

- 【Great】<https://github.com/xmxoxo/BERT-train2deploy> (Tensorlfow)

    BERT模型从训练到部署

- [文本分类实战（十）—— BERT 预训练模型](https://www.cnblogs.com/jiangxinyang/p/10241243.html) (Tensorflow)

- [Multi-label Text Classification using BERT – The Mighty Transformer](https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d)

    **Chinese**:[手把手教你用BERT进行多标签文本分类](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651667790&idx=1&sn=c051c993ee561c7ada8c03b58679f305)

- <https://github.com/bamtercelboo/PyTorch_Bert_Text_Classification> (PyTorch)

    PyTorch Bert Text Classification

- <https://github.com/AidenHuen/BERT-BiLSTM-CRF> (Keras)

    BERT-BiLSTM-CRF的Keras版实现  预训练模型为chinese_L-12_H-768_A-12.zip，使用BERT客户端和服务器bert-serving-server和bert-serving-client

- <https://github.com/llcing/BiLSTM-CRF-ChineseNER.pytorch> (PyTorch)

    PyTorch implement of BiLSTM-CRF for Chinese NER
    

### 7.4.3 RoBERTa

[RoBERTa: A Robustly Optimized BERT Pretraining Approach - Washington2019](https://arxiv.org/abs/1907.11692)


## 7.5 GPT

GPT1: [Improving Language Understanding by Generative Pre-Training - OpenAI2018](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

GPT2: [Language Models are Unsupervised Multitask Learners - OpenAI2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

#### Code

- GPT1: <https://github.com/huggingface/pytorch-openai-transformer-lm> (PyTorch)

- GPT1: <https://github.com/openai/finetune-transformer-lm> (Tensorflow)

- GPT2: <https://github.com/CyberZHG/keras-gpt-2> (Keras)

- GPT2: <https://github.com/morizeyao/gpt2-chinese> (PyTorch)

    Chinese version of GPT2 training code, using BERT tokenizer

- GPT2: <https://github.com/openai/gpt-2> (Tensorflow)

    **Data*: <https://github.com/openai/gpt-2-output-dataset>

#### Practice

- <https://github.com/imcaspar/gpt2-ml>

    **Article**: [只需单击三次，让中文GPT-2为你生成定制故事 - 2019](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650773965&idx=4&sn=c974e222235d79af62c83c74bc5251b3)

#### Article

- GPT1: [Improving Language Understanding with Unsupervised Learning](https://openai.com/blog/language-unsupervised/)

- GPT2: [Better Language Models and Their Implications](https://www.openai.com/blog/better-language-models/)

- GPT2: [The Illustrated GPT-2 (Visualizing Transformer Language Models) - 2019](https://jalammar.github.io/illustrated-gpt2/)

    **Chinese**: [完全图解GPT-2：看完这篇就够了（一）](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650768689&idx=2&sn=ff46397819b544a19c3200297f180dea)


## 7.6 ULMFit

[Universal Language Model Fine-tuning for Text Classification - fastAI2018](https://arxiv.org/abs/1801.06146)

在 Kaggle 和其他竞赛中，ULMFit 的效果都超越了其他模型。

#### Code

- <https://github.com/fastai/fastai/tree/ulmfit_v1> (PyTorch)

#### Article

- [fastAI 课程《Cutting Edge DeepLearning course》第10课对 ULMFit 的介绍](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)

- [fastAI 发布的课程](http://course18.fast.ai/lessons/lesson10.html)


## 7.7 ERNIE (Baidu & THU)

### 7.7.1 ERNIE - Baidu

百度提出知识增强的语义表示模型 ERNIE（Enhanced Representation from kNowledge IntEgration），并发布了基于 PaddlePaddle 的开源代码与模型，在语言推断、语义相似度、命名实体识别、情感分析、问答匹配等自然语言处理（NLP）各类中文任务上的验证显示，模型效果全面超越 BERT。

[ERNIE: Enhanced Representation through Knowledge Integration - Baidu2019](https://arxiv.org/abs/1904.09223)

[ERNIE 2.0: A Continual Pre-training Framework for Language Understanding - Baidu2019](https://arxiv.org/abs/1907.12412v1)

#### Code

- <https://github.com/PaddlePaddle/ERNIE> (PaddlePaddle)

#### Article

- [中文任务全面超越BERT：百度正式发布NLP预训练模型ERNIE](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247489050&idx=2&sn=0474c58819363b84b99d0f9ffe868f6a)

- [ERNIE Tutorial（论文笔记 + 实践指南）- 2019](https://yam.gift/2019/08/02/Paper/2019-08-02-Baidu-ERNIE-Tutorial/)

- [如何评价百度新发布的NLP预训练模型ERNIE？](https://www.zhihu.com/question/316140575/answer/719617103)


### 7.7.2 ERNIE - THU

[ERNIE: Enhanced Language Representation with Informative Entities - THU2019](https://arxiv.org/abs/1905.07129)

#### Article

- [Bert 改进： 如何融入知识 - 2019](https://zhuanlan.zhihu.com/p/69941989)

    BERT, 百度ERNIE, 清华ERNIE


## 7.8 CoVe

[Learned in Translation: Contextualized Word Vectors - Salesforce2017](https://arxiv.org/abs/1708.00107)

#### Code

- <https://github.com/salesforce/cove> (PyTorch)

- <https://github.com/rgsachin/CoVe> (Keras)


## 7.9 XLM

XLNet其实本质上还是ELMO, GPT, Bert这一系列两阶段模型的进一步延伸

[Cross-lingual Language Model Pretraining - Facebook2019](https://arxiv.org/abs/1901.07291)

#### Code

- <https://github.com/facebookresearch/XLM> (PyTorch)


## 7.10 XLNet

[XLNet: Generalized Autoregressive Pretraining for Language Understanding - CMU2019](https://arxiv.org/abs/1906.08237)

#### Code

- <https://github.com/CyberZHG/keras-xlnet> (Keras)

- <https://github.com/zihangdai/xlnet> (Tensorflow)

#### Article

- [XLNet:运行机制及和Bert的异同比较](https://zhuanlan.zhihu.com/p/70257427)

- [XLNet太贵？这位小哥在PyTorch Wrapper上做了微缩版的](https://github.com/graykode/xlnet-pytorch)


## 7.11 T5 Model: NLP Text-to-Text

#### Article

- [如何评价 Google 提出的预训练模型 T5](https://www.zhihu.com/question/352227934/answer/868639851)

- [T5 模型：NLP Text-to-Text 预训练模型超大规模探索 - 2019](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650411701&idx=2&sn=f253b2cde92e0be27e4cdb010f8f957a)


## 7.12 Application

- [基于LSTM与TensorFlow Lite，kika输入法是如何造就的 - 2018](https://cloud.tencent.com/developer/article/1118053)

