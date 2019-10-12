
# 21. Machine Comprehension

## 21.1 Overview

#### Paper

- <https://github.com/thunlp/RCPapers>

    Must-read papers on Machine Reading Comprehension.

- [Neural Machine Reading Comprehension: Methods and Trends - NUDT2019](https://arxiv.org/abs/1907.01118)

    综述：总结机器阅读理解领域已经提出的方法和近期发展趋势。


#### Data

- <https://github.com/ymcui/Chinese-Cloze-RC>

    A Chinese Cloze-style Reading Comprehension Dataset: People Daily & Children's Fairy Tale (CFT)


#### Competition

**AI Challenger 2018 观点型问题阅读理解**

- <https://github.com/yuhaitao1994/AIchallenger2018_MachineReadingComprehension> (Tensorflow)

    复赛第8名 解决方案


**法研杯 2019 阅读理解**

- <https://github.com/circlePi/2019Cail-A-Bert-Joint-Baseline-for-Machine-Comprehension> (PyTorch)

    A pytorch implement of bert joint baseline for machine comprehension task in 2019 cail


#### Project

- <https://github.com/chineseGLUE/chineseGLUE>

    中文语言理解基准测评：Language Understanding Evaluation benchmark for Chinese: datasets, baselines, pre-trained models,corpus and leaderboard


## 21.2 Memory Network

[Memory Networks - Facebook2015](https://arxiv.org/abs/1410.3916)

[End-To-End Memory Networks - Facebook2015](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)

#### Article

- [论文笔记 - Memory Networks 系列](https://zhuanlan.zhihu.com/p/32257642)

#### Code

- <https://github.com/brightmart/text_classification> (Tensorflow)


## 21.3 R-NET

[R-Net: Machine Reading Comprehension with Self-Matching- MSRA2017](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/05/r-net.pdf )

#### Code

- <https://github.com/NLPLearn/R-net> (Tensorflow)

#### Article

- [Understanding R-Net: Microsoft’s ‘superhuman’ reading AI](https://codeburst.io/understanding-r-net-microsofts-superhuman-reading-ai-23ff7ededd96)

- [《R-NET：MACHINE READING COMPREHENSION》阅读笔记](https://zhuanlan.zhihu.com/p/61502862)


## 21.4 Recurrent Entity Network ?

[Tracking the World State with Recurrent Entity Networks - Facebook2017](https://arxiv.org/abs/1612.03969)

**Structure**:

Input (Context, Question) -> Encoding (BOW with Position Mask or BiRNN) -> 

Dynamic Memory (Similarity of keys, values-> Gate -> Candidate Hidden State -> Current Hidden State) -> 

Output Module (Similarity of Query and Hidden State -> Possibility Distribution -> Weighted Sum -> Non-linearity Transform -> Predicted Label)

![recurrent_entity_network_structure](./image/recurrent_entity_network_01.png)

#### code

- <https://github.com/facebook/MemNN/tree/master/EntNet-babi> (Torch, Lua)

- <https://github.com/jimfleming/recurrent-entity-networks> (Tensorflow)

- <https://github.com/brightmart/text_classification> (Tensorflow)


## 21.5 Code

- <https://github.com/laddie132/Match-LSTM> (PyTorch)

    **Paper**: 1, 2, R-NET


## 21.6 Paper

- 1. [Machine Comprehension Using Match-LSTM and Answer Pointer - Singapore2016](https://arxiv.org/abs/1608.07905)

- 2. [Reinforced Mnemonic Reader for Machine Reading Comprehension - NUDT2018](https://arxiv.org/abs/1705.02798)

