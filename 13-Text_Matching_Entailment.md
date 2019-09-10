
# 13. Text Matching & Entailment

## 13.1 Text Matching

Text Matching and Text Entailment both belong to Natural Language Inference (NLI), and sometimes they are very close to each other.

### 13.1.1 Overview

#### Paper

-  <https://github.com/NTMC-Community/awaresome-neural-models-for-semantic-match>

    A curated list of papers dedicated to neural text (semantic) matching.

- <https://github.com/Accagain2014/TextMatching>

    Papers, Methods, Books, Tutorial, Datasets, Competitions, Pretrained Models, Courses

#### Code

- <https://github.com/pengming617/text_matching> (Tensorflow)

    文本匹配的相关模型DSSM, ESIM, ABCNN, BIMPM等，数据集为LCQMC官方数据


### 13.1.2 MatchZoo

[MatchZoo: A Toolkit for Deep Text Matching - CAS2017](https://arxiv.org/abs/1707.07270)

#### Code

- <https://github.com/NTMC-Community/MatchZoo>


## 13.2 Text Entailment

### 13.2.1 Overview

Textual entailment (TE) in natural language processing is a directional relation between text fragments. The relation holds whenever the truth of one text fragment follows from another text. In the TE framework, the entailing and entailed texts are termed text (t) and hypothesis (h), respectively. Textual entailment is not the same as pure logical entailment — it has a more relaxed definition: 

t entails h (t ⇒ h) if, typically, a human reading t would infer that h is most likely true.

中文名：文本蕴含

[Wikipedia](https://en.wikipedia.org/wiki/Textual_entailment)

#### Code

- <https://github.com/liuhuanyong/ChineseTextualInference>

    中文文本推断项目,包括88万文本蕴含中文文本蕴含数据集的翻译与构建,基于深度学习的文本蕴含判定模型构建.

#### Article

- [Textual entailment with TensorFlow - 2017](https://www.colabug.com/496258.html)

#### Book

- [Recognizing Textual Entailment: Models and Applications](https://ieeexplore.ieee.org/document/6812786)


### 13.2.2 ESIM

[Enhanced LSTM for Natural Language Inference - USTC2017](https://arxiv.org/abs/1609.06038)

#### Code

- <https://github.com/sdnr1/EBIM-NLI> (Keras)

- <https://blog.csdn.net/wcy23580/article/details/84990923> (Keras)

- <https://github.com/zsy23/chinese_nlp/tree/master/natural_language_inference> (Tensorflow)

- <https://github.com/allenai/allennlp/blob/master/allennlp/models/esim.py> (PyTorch)

- <https://github.com/pengshuang/Text-Similarity> (PyTorch)

    **Article**: [短文本匹配的利器-ESIM](https://zhuanlan.zhihu.com/p/47580077)

#### Practice

- [Python实战——ESIM 模型搭建（keras版）- 2018](https://blog.csdn.net/wcy23580/article/details/84990923)


### 13.2.3 ABCNN

[ABCNN: Attention-Based Convolutional Neural Network for Modeling Sentence Pairs - Germany2018](https://arxiv.org/abs/1512.05193)

#### Code

- <https://github.com/lsrock1/abcnn_pytorch> (PyTorch)
