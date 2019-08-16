
# 8. Pretrained Model

## 8.1 Overview

#### Article

- [8 Excellent Pretrained Models to get you Started with NLP](https://www.analyticsvidhya.com/blog/2019/03/pretrained-models-get-started-nlp/)

    包括：ULMFiT, Transformer, BERT, Transformer-XL, GPT-2, ELMo, Flair, StanfordNLP

    中文解读：[8种优秀预训练模型大盘点，NLP应用so easy！](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651669109&idx=2&sn=29b4e45291eac659af2967a1e246aa03)

- [Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html)

    包括：CoVe, ELMo, Croww-View Training, ULMFiT, GPT, BERT, GPT-2

    中文解读：[上下文预训练模型最全整理：原理、应用、开源代码、数据分享](https://mp.weixin.qq.com/s?__biz=MzIxNDgzNDg3NQ==&mid=2247485551&idx=1&sn=de0a04647870543fe0b36d024f58429e)

- [BERT, EMLo and ERNIE Implementation with PaddlePaddle](https://github.com/PaddlePaddle/LARK)

- [Language Models and Contextualised Word Embeddings](http://www.davidsbatista.net/blog/2018/12/06/Word_Embeddings/)
  
  对 ELMo, BERT 及其他模型进行了一个简单的综述

- [从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史 - 张俊林](https://zhuanlan.zhihu.com/p/49271699)

- [NLP's ImageNet moment has arrived - 2018](https://thegradient.pub/nlp-imagenet/)

    词嵌入已死，语言模型当立

- [nlp中的预训练语言模型总结(单向模型、BERT系列模型、XLNet) - 2019](https://zhuanlan.zhihu.com/p/76912493)


## 8.2 EMLo - TOTODO

EMLo 是第一个使用预训练模型进行词嵌入的方法，将句子输入ELMO，可以得到句子中每个词的向量表示。

#### Paper

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


## 8.3 BERT - TOTODO

#### Paper

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding - Google2019](https://arxiv.org/abs/1810.04805)

#### Code

- <https://github.com/CyberZHG/keras-bert> (Keras)

- 【Great】<https://github.com/huggingface/pytorch-pretrained-BERT> (PyTorch)

    **Chinese**: [BERT、GPT-2这些顶尖工具到底该怎么用到我的模型里？ - 2019](https://baijiahao.baidu.com/s?id=1626146900426049013)

- <https://github.com/google-research/bert> (Tensorflow)

- <https://github.com/codertimo/BERT-pytorch> (PyTorch)


#### Article

- 编码器: [Dissecting BERT Part 1: The Encoder](https://medium.com/dissecting-bert/dissecting-bert-part-1-d3c3d495cdb3)

- 网络结构: [Understanding BERT Part 2: BERT Specifics](https://medium.com/dissecting-bert/dissecting-bert-part2-335ff2ed9c73)

- 解码器: [Dissecting BERT Appendix: The Decoder](https://medium.com/dissecting-bert/dissecting-bert-appendix-the-decoder-3b86f66b0e5f)

- [The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning)](http://jalammar.github.io/illustrated-bert/)

- [彻底搞懂BERT](https://www.cnblogs.com/rucwxb/p/10277217.html)


#### Practice

- [Pretrained PyTorch models for BERT, OpenAI GPT & GPT-2, Google/CMU Transformer-XL](https://github.com/huggingface/pytorch-pretrained-bert) (PyTorch)

- Good!!! [Implemention some Baseline Model upon Bert for Text Classification](https://github.com/songyingxin/bert-textclassification) (PyTorch)

- [BERT模型从训练到部署](https://github.com/xmxoxo/BERT-train2deploy)

- <https://github.com/brightmart/text_classification/tree/master/a00_Bert> (Tensorflow)

- [文本分类实战（十）—— BERT 预训练模型](https://www.cnblogs.com/jiangxinyang/p/10241243.html)

- [Multi-label Text Classification using BERT – The Mighty Transformer](https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d)

    [中文解读](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651667790&idx=1&sn=c051c993ee561c7ada8c03b58679f305)

- <https://github.com/bamtercelboo/PyTorch_Bert_Text_Classification> (PyTorch)

    PyTorch Bert Text Classification

- [关于最近实践 Bert 的一些坑 - 2019](https://zhuanlan.zhihu.com/p/69389583)

- [BERT fintune 的艺术 - 2019](https://zhuanlan.zhihu.com/p/62642374)


#### Further

- [哈工大讯飞联合实验室发布基于全词覆盖的中文BERT预训练模型](https://github.com/ymcui/Chinese-BERT-wwm)
  
  **Paper**: [Pre-Training with Whole Word Masking for Chinese BERT - HIT2019](https://arxiv.org/abs/1906.08101)

- [站在BERT肩膀上的NLP新秀们（PART I）](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247489437&idx=4&sn=d1d7ca7e3b4b0a1710252e8d52affe4d)
  
  给 BERT 模型增加外部知识信息，使其能更好地感知真实世界，主要讲了 ERNIE from Baidu 和 ERNIE from THU

- [站在BERT肩膀上的NLP新秀们（PART II）](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650409996&idx=1&sn=ddf837339e50001be4514fee743bfe9d)
  
  主要讲了 XLMs from Facebook, LASER from Facebook, MASS from Microsoft 和 UNILM from Microsoft

- [站在BERT肩膀上的NLP新秀们（PART III）](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650410110&idx=1&sn=310f675cf0cc1e2a1f4cc7b919743bc4)
  
  主要看看预训练模型中的增强训练（多任务学习/数据增强）以及BERT多模态应用： MT-DNN from Microsoft, MT-DNN-2 from Microsoft, GPT-2 from OpenAI 和 VideoBERT from Google

- [Bert 改进： 如何融入知识 - 2019](https://zhuanlan.zhihu.com/p/69941989)

    BERT, 百度ERNIE, 清华ERNIE, 


## 8.4 GPT

#### Paper

- GPT1: [Improving Language Understanding by Generative Pre-Training - OpenAI2018](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

- GPT2: [Language Models are Unsupervised Multitask Learners - OpenAI2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

#### Code

- GPT1: <https://github.com/huggingface/pytorch-openai-transformer-lm> (PyTorch)

- GPT1: <https://github.com/openai/finetune-transformer-lm> (Tensorflow)

- GPT2: <https://github.com/CyberZHG/keras-gpt-2> (Keras)

- GPT2: <https://github.com/openai/gpt-2> (Tensorflow)

#### Article

- GPT1: [Improving Language Understanding with Unsupervised Learning](https://openai.com/blog/language-unsupervised/)

- GPT2: [Better Language Models and Their Implications](https://www.openai.com/blog/better-language-models/)


## 8.5 ULMFit

在 Kaggle 和其他竞赛中，ULMFit 的效果都超越了其他模型。

#### Paper

[Universal Language Model Fine-tuning for Text Classification - fastAI2018](https://arxiv.org/abs/1801.06146)

#### Code

- <https://github.com/fastai/fastai/tree/ulmfit_v1> (PyTorch)

#### Article

- [fastAI 课程《Cutting Edge DeepLearning course》第10课对 ULMFit 的介绍](http://nlp.fast.ai/classification/2018/05/15/introducting-ulmfit.html)

- [fastAI 发布的课程](http://course18.fast.ai/lessons/lesson10.html)


## 8.6 Flair

混合了BERT, EMLo, GPT-2，其实是一个Framework，一个library.

#### Code

- <https://github.com/zalandoresearch/flair> (PyTorch)


## 8.7 ERNIE

百度提出知识增强的语义表示模型 ERNIE（Enhanced Representation from kNowledge IntEgration），并发布了基于 PaddlePaddle 的开源代码与模型，在语言推断、语义相似度、命名实体识别、情感分析、问答匹配等自然语言处理（NLP）各类中文任务上的验证显示，模型效果全面超越 BERT。

#### Paper

- [ERNIE: Enhanced Representation through Knowledge Integration - Baidu2019](https://arxiv.org/abs/1904.09223)

- [ERNIE 2.0: A Continual Pre-training Framework for Language Understanding - Baidu2019](https://arxiv.org/abs/1907.12412v1)

#### Code

- <https://github.com/PaddlePaddle/ERNIE> (PaddlePaddle)

#### Article

- [中文任务全面超越BERT：百度正式发布NLP预训练模型ERNIE](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247489050&idx=2&sn=0474c58819363b84b99d0f9ffe868f6a)


## 8.8 CoVe

#### Paper

[Learned in Translation: Contextualized Word Vectors - Salesforce2017](https://arxiv.org/abs/1708.00107)

#### Code

- <https://github.com/salesforce/cove> (PyTorch)

- <https://github.com/rgsachin/CoVe> (Keras)


## 8.9 XLM

XLNet其实本质上还是ELMO, GPT, Bert这一系列两阶段模型的进一步延伸。

#### Paper

[Cross-lingual Language Model Pretraining - Facebook2019](https://arxiv.org/abs/1901.07291)

#### Code

- <https://github.com/facebookresearch/XLM> (PyTorch)


## 8.10 XLNet

#### Paper

[XLNet: Generalized Autoregressive Pretraining for Language Understanding - CMU2019](https://arxiv.org/abs/1906.08237)

#### Code

- <https://github.com/CyberZHG/keras-xlnet> (Keras)

- <https://github.com/zihangdai/xlnet> (Tensorflow)

#### Article

- [XLNet:运行机制及和Bert的异同比较](https://zhuanlan.zhihu.com/p/70257427)

- [XLNet太贵？这位小哥在PyTorch Wrapper上做了微缩版的](https://github.com/graykode/xlnet-pytorch)


## 8.11 Application

- [基于LSTM与TensorFlow Lite，kika输入法是如何造就的 - 2018](https://cloud.tencent.com/developer/article/1118053)