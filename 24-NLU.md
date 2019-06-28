
# 24. NLU

## 24.1 Overview

## 24.2 Memory Network

#### Paper

- [Memory Networks - Facebook2015](https://arxiv.org/abs/1410.3916)

- [End-To-End Memory Networks - Facebook2015](http://papers.nips.cc/paper/5846-end-to-end-memory-networks.pdf)

#### Article

- [论文笔记 - Memory Networks 系列](https://zhuanlan.zhihu.com/p/32257642?edition=yidianzixun&utm_source=yidianzixun&yidian_docid=0HymGR2b)

#### Code

- <https://github.com/brightmart/text_classification> (Tensorflow)

## 24.3 Recurrent Entity Network

#### Paper

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
