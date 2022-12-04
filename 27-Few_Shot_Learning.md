# 27. Few Shot Learning

少样本学习，其研究领域与迁移学习有一大部分交集部分，即在源域有足够多样本，而在目标域样本不足。


## Paper

- <https://github.com/tata1661/FewShotPapers>

  **Article**: [什么是小样本学习？这篇综述文章用166篇参考文献告诉你答案](https://mp.weixin.qq.com/s/jzo8kyh0qBCObvFQhiZePg)

## Article

- [深度文本的表征与聚类在小样本场景中的探索 - 2020](https://mp.weixin.qq.com/s/TSFxYQdjjHuyrIJLM2fdcw)

- [Few-shot learning 少样本学习 - 2019](https://zhuanlan.zhihu.com/p/66552960)

- [NLP中的少样本困境问题探究 - 2020](https://mp.weixin.qq.com/s/S9EWQZsKlww8GC8jCf4QvA)


## Course

### Deepshare课程

少样本学习：人类基于以往的学习能力和先验知识，在只看几张企鹅图片的情况下，学会了识别企鹅。对上述能力的模仿，就是少样本学习：在数据匮乏、标注极少的情况下，学会对样本类别的识别（该类别在模型训练时不曾出现）。

Support Set: 标注数据，不同于训练集，样本量很少

Query Set: 相似于测试集（相似但有别）

N-way, K-show：Support Set中有N个类别，每个类别有K个样本。


Few Shot Learning经典思路：
- Data: 补充扩充更多数据，比如数据增强
- Model: 基于先验知识简化模型，比如Metric Based Model
- Algorithm: 基于先验知识，比如


Model:

Metric Based Model: