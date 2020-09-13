
# 14. Text Generation

## 14.1 Overview

#### Article

- [文本生成12：4种融合知识的text generation（推荐收藏） - 2020](https://zhuanlan.zhihu.com/p/133266258)

- [文本生成评价指标的进化与推翻 - 2020](https://mp.weixin.qq.com/s/oOqkVLZejWeFYAbZULbdyg)


#### Practice

- ["自动作诗机"上线，代码和数据都是公开的 - 2019](ttps://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650410297&idx=1&sn=cda7099455083fbd412d0fdcb41acbea&scene=21)

- [风云三尺剑，花鸟一床书---对联数据集和自动对联机器人 - 2019](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650408997&idx=1&sn=93395c083d85cf15490cf36cb5251a0f&scene=21)

- [用 GPT-2 自动写诗，从五言绝句开始 - 2020](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650412442&idx=1&sn=4e874b9598f49b7cf89267d561addf73)


## 14.2 Text Summary

### 14.2.1 Overview

#### Paper

- [文本摘要(text summarization)最新研究热点、发展趋势，里程碑论文推荐 - 2020](https://zhuanlan.zhihu.com/p/111266615)

#### Article

- [NLP中自动生产文摘（auto text summarization）- 2017](https://www.sohu.com/a/197255751_609569)

    类似于综述，较为全面！

- [当深度学习遇见自动文本摘要 - 2018](hts://cloud.tencent.com/developer/article/1029101)

- [Comprehensive Guide to Text Summarization using Deep Learning in Python - 2019](https://www.analyticsvidhya.com/blog/2019/06/comprehensive-guide-text-summarization-using-deep-learning-python/)

    **Chinese**: [Python利用深度学习进行文本摘要的综合指南](https://mp.weixin.qq.com/s/gDZyTbM1nw3fbEnU--y3nQ)

- [文本摘要简述 - 2019](https://www.jiqizhixin.com/articles/2019-03-25-7)

    **YAO**: OK

    单/多文档 + 抽取/生成式 + 无/有监督 + 传统/DL方法

    抽取式: 无监督-图方法、聚类；有监督-DL(序列标注、句子排序)

    - Lead-3: 抽取文章前三句
    - TextRank: 依照PageRank，句子为节点，使用句子相似度，构建无向有权图，使用边上的权重迭代更新节点值，最后选择TopK个得分最高的节点(句子)
    - 聚类: 句子为节点，得到句子向量表示，使用K-means和Mean-shift完成聚类，得到K个类别，最后从每个类别中选择距离质心最近的句子，得到K个句子
    - 序列标注: 为每个句子打一个二分类标签(0,1)，最后选择所有标签为1的句子。关键在于获得句子的向量表示。如使用BiGRU分别建模word和sentence的向量表示，训练数据中要通过启发式规则获得句子标签
    - 句子排序: 针对每个句子，输出其是否是摘要句的概率，最后依据概率，选择TopK个句子

    生成式: Seq2Seq + Attention + Copy + Coverage

    - Decoder隐层状态与Encoder隐层状态计算权重，得到Context向量，利用Context和Decoder隐层状态计算输出概率Pgen
    - Copy: 在解码的每一步计算拷贝或生成的概率，因为词表是固定的，可以选择从原文中拷贝词语到摘要中，有效缓解OOV
    - Coverage: 在解码的每一步考虑之前步的Attention权重，结合Coverage损失，避免继续考虑已经获得高权重的部分，可有效缓解生成重复的问题
    - 外部信息: 使用真实摘要来指导文本摘要的生成
    - 生成对抗方式: 生成模型G来生成摘要，判别模型D来区分真实摘要和生成摘要

    抽取生成式: 结合抽取式和生成式，首先选择重要内容，然后基于重要内容生成摘要

    - 内容选择: 建模为词语级别序列标注任务，或者直接使用TextRank算法获得关键词，这属于hard方式，显式利用文本级别信息，也可使用门控机制，从文本编码向量中选择有用信息用于之后摘要生成，这属于Soft方式
    - 摘要生成: 如使用Pointer-generator网络，使用内容选择部分计算的概率修改原本Attention概率，使得Decoder仅关注选择的内容
    - 其他: 其他模型方法

- [文本自动摘要任务的“不完全”心得总结 - 2019](https://zhuanlan.zhihu.com/p/83596443)

- [知识图谱如何助力文本摘要生成 - 2019](https://mp.weixin.qq.com/s?__biz=MzUyMDY0OTg3Nw==&mid=2247483982&idx=1&sn=8740d862216be0d4cbe81f650aedf8d0)

- [赛尔笔记 | 事实感知的生成式文本摘要 - 2020](https://mp.weixin.qq.com/s/jwYZ27nqi1eAocqDn7-t4A)


#### Practice

- <https://github.com/ztz818/Automatic-generation-of-text-summaries> (Tensorflow)

    使用两种方法（抽取式Textrank和概要式seq2seq）自动提取文本摘要

- [How To Create Data Products That Are Magical Using Sequence-to-Sequence Models](https://towardsdatascience.com/how-to-create-data-products-that-are-magical-using-sequence-to-sequence-models-703f86a231f8)

    对Github项目进行文本摘要总结

    **Chinese**: [手把手教你用seq2seq模型创建数据产品 - 2018](https://yq.aliyun.com/articles/560596)

- <https://github.com/bojone/seq2seq> (Keras)

    Auto Title


#### Data

- <https://github.com/wonderfulsuccess/chinese_abstractive_corpus>

    教育行业新闻 自动文摘 语料库 自动摘要


### 14.2.2 Deep Learning

#### Paper

- 1. [Regularizing and Optimizing LSTM Language Models - 2017](https://arxiv.org/abs/1708.02182)

- 2. [Global Encoding for Abstractive Summarization - PKU2018](https://arxiv.org/abs/1805.03989)

- 3. [Fast Abstractive Summarization with Reinforce-Selected Sentence Rewriting - UNC2018](https://arxiv.org/abs/1805.11080)

    **Code**: <https://github.com/ChenRocks/fast_abs_rl> (PyTorch)

- 4. [Fine-tune BERT for Extractive Summarization - Edinburgh2019](https://arxiv.org/abs/1903.10318)

    **Code**: <https://github.com/nlpyang/bertsum> (PyTorch)



## 14.3 Title Generation

### 14.3.1 Overview



#### Competition

**Byte Cup 2018 国际机器学习竞赛: 英文文章标题自动生成**

- <https://github.com/iwangjian/ByteCup2018> (PyTorch)

    Rank 6

    **Paper**: Paper1, 2, 3




## 14.4 Data2Text

Data包括：表格（天气预报、人物百科等）、图片、知识图谱三元组等

对Encoder的修改：捕捉结构上的联系，比如表格会加入记录类型，知识图谱会进行BFS等

对Decoder的修改：建立输入输出之间的Alignment


