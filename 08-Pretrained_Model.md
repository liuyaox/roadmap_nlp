
# 8. Pretrained Model

## 8.1 Overview

#### Article

- [8 Excellent Pretrained Models to get you Started with NLP](https://www.analyticsvidhya.com/blog/2019/03/pretrained-models-get-started-nlp/)

    包括：ULMFiT, Transformer, BERT, Transformer-XL, GPT-2, ELMo, Flair, StanfordNLP

    中文解读：[8种优秀预训练模型大盘点，NLP应用so easy！](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651669109&idx=2&sn=29b4e45291eac659af2967a1e246aa03&chksm=bd4c65e68a3becf0fdbb58b02a4c517c4dc62a6715763c9997b5139e4f6f96baab3ea850b96a&mpshare=1&scene=1&srcid=&key=10bce1f8c56039bd343495c51671ce60ec99a527555e1ae881a1c0ec560163d18b152a00dc0cac877c76ef4971469ad25afa2728e4e521188ed7a3995c25b21640e3ba391d2425525173a4d3060ea29c&ascene=1&uin=MjcwMjE1Nzk1&devicetype=Windows+7&version=62060833&lang=en&pass_ticket=vdfRef2%2FTgBj6IL%2Bi547vWc7twm2xFzEDaoX9%2Bj2dp9PWcoVadURLaEObEVcUU%2Ff)

- [Generalized Language Models](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html)

    包括：CoVe, ELMo, Croww-View Training, ULMFiT, GPT, BERT, GPT-2

    中文解读：[上下文预训练模型最全整理：原理、应用、开源代码、数据分享](https://mp.weixin.qq.com/s?__biz=MzIxNDgzNDg3NQ==&mid=2247485551&idx=1&sn=de0a04647870543fe0b36d024f58429e&chksm=97a0c3bba0d74aade31178413de56de5d130620f8c05b3185ed45d14bcae67bfed1aae129952&mpshare=1&scene=1&srcid=&key=cb3ec8ead4de2b57b1bd58e48635b5ab74e3d34071cef9919ce843b038e2a37413d241edb643bd67abadc9549d0c614eaf71148f71da475e3c5834fa8ba715a665e4b68a5742b2fc25aec426bbca3eda&ascene=1&uin=MjcwMjE1Nzk1&devicetype=Windows+7&version=62060833&lang=en&pass_ticket=vdfRef2%2FTgBj6IL%2Bi547vWc7twm2xFzEDaoX9%2Bj2dp9PWcoVadURLaEObEVcUU%2Ff)

- [BERT, EMLo and ERNIE Implementation with PaddlePaddle](https://github.com/PaddlePaddle/LARK)

- [Language Models and Contextualised Word Embeddings](http://www.davidsbatista.net/blog/2018/12/06/Word_Embeddings/)
  
  对 ELMo, BERT 及其他模型进行了一个简单的综述


## 8.2 EMLO - TOTODO

EMLO 是第一个使用预训练模型进行词嵌入的方法，将句子输入ELMO，可以得到句子中每个词的向量表示。

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


## 8.3 BERT - TOTODO

#### Paper

[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding - Google2019](https://arxiv.org/abs/1810.04805)

#### Code

- <https://github.com/huggingface/pytorch-pretrained-BERT> (PyTorch)

- <https://github.com/google-research/bert> (Tensorflow)

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

    [中文解读](https://mp.weixin.qq.com/s?__biz=MjM5MTQzNzU2NA==&mid=2651667790&idx=1&sn=c051c993ee561c7ada8c03b58679f305&chksm=bd4c1add8a3b93cbd06bea5c0885ec21943ad89b597c195512fa9ae4701c385ff5254b90c0f8&mpshare=1&scene=1&srcid=#rd)

#### Further

- [哈工大讯飞联合实验室发布基于全词覆盖的中文BERT预训练模型](https://github.com/ymcui/Chinese-BERT-wwm)
  
  **Paper**: [Pre-Training with Whole Word Masking for Chinese BERT - HIT2019](https://arxiv.org/abs/1906.08101)

- [站在BERT肩膀上的NLP新秀们（PART I）](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247489437&idx=4&sn=d1d7ca7e3b4b0a1710252e8d52affe4d&chksm=ebb42f49dcc3a65ffbf86a6016db944a04911a17bd22979cfe17f2da52c2aa0d68833bf5eda8&mpshare=1&scene=1&srcid=&key=67a728f6339a0a1e70d20293822ea56da128bf66230228b7602acf0e31e1e1d7235379d978254cf7172f1639e89d6c2cd984a2205d92963f471a6c99238e89225aa2efc53febb55162ee78dd20023973&ascene=1&uin=MjcwMjE1Nzk1&devicetype=Windows+7&version=62060833&lang=en&pass_ticket=UYja1qL4FjsIjJwYJqT7vvLoCJho0%2Bf7%2FxcTCgiuCuAokcVpCfGb7MuLVdYj2QHK)
  
  给 BERT 模型增加外部知识信息，使其能更好地感知真实世界，主要讲了 ERNIE from Baidu 和 ERNIE from THU

- [站在BERT肩膀上的NLP新秀们（PART II）](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650409996&idx=1&sn=ddf837339e50001be4514fee743bfe9d&chksm=becd8a5689ba03405b3e11c882e376effc407b1ecde745a1df0295008329b5d83c4a6ab5fc1c&scene=21#wechat_redirect)
  
  主要讲了 XLMs from Facebook, LASER from Facebook, MASS from Microsoft 和 UNILM from Microsoft

- [站在BERT肩膀上的NLP新秀们（PART III）](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650410110&idx=1&sn=310f675cf0cc1e2a1f4cc7b919743bc4&chksm=becd8a2489ba033262daff98227a9887f5a80fba45590b75c9328567258f9420f98423af6e19&mpshare=1&scene=1&srcid=&key=67a728f6339a0a1ede767062b26eef6e4f4dc12c1a81bf2f24d459516816478c0c05b3e3a6625e2e074c356aecc75e63253a21de6491e01d23be22709b9e32605da5ae8c545820a5d0c02a0428e7ed3a&ascene=1&uin=MjcwMjE1Nzk1&devicetype=Windows+7&version=62060833&lang=en&pass_ticket=SWBCv%2Bah0eIEISXOXsPuddmJM8%2Bvbzjxrwkg2kH2Il116bWpQYmtXQht1D9khSa%2B)
  
  主要看看预训练模型中的增强训练（多任务学习/数据增强）以及BERT多模态应用： MT-DNN from Microsoft, MT-DNN-2 from Microsoft, GPT-2 from OpenAI 和 VideoBERT from Google


## 8.4 GPT

#### Paper

- GPT1: [Improving Language Understanding by Generative Pre-Training - OpenAI2018](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

- GPT2: [Language Models are Unsupervised Multitask Learners - OpenAI2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

#### Code

- GPT1: <https://github.com/huggingface/pytorch-openai-transformer-lm> (PyTorch)

- GPT1: <https://github.com/openai/finetune-transformer-lm> (Tensorflow)

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

#### Code

- <https://github.com/PaddlePaddle/LARK/tree/develop/ERNIE>

#### Article

- [中文任务全面超越BERT：百度正式发布NLP预训练模型ERNIE](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247489050&idx=2&sn=0474c58819363b84b99d0f9ffe868f6a&chksm=ebb42ecedcc3a7d811f4c96598c7d171df4f615963d6016e90e0e50666bde56f36f249c3476c&mpshare=1&scene=1&srcid=&key=59b86e7114b344c3fd931cb3fd1b5a3b6217cf806b6cf11f896b6e50a8837099c84b0619661c3a018aa63edb361d3a9640353699afa2b94ea96a4fbc9ab72df7085fc72c96afdeaaaf5d266e4175924f&ascene=1&uin=MjcwMjE1Nzk1&devicetype=Windows+7&version=62060833&lang=en&pass_ticket=vdfRef2%2FTgBj6IL%2Bi547vWc7twm2xFzEDaoX9%2Bj2dp9PWcoVadURLaEObEVcUU%2Ff)


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

- <https://github.com/zihangdai/xlnet> (Tensorflow)

#### Article

- [XLNet:运行机制及和Bert的异同比较](https://zhuanlan.zhihu.com/p/70257427)

- [XLNet太贵？这位小哥在PyTorch Wrapper上做了微缩版的](https://github.com/graykode/xlnet-pytorch)