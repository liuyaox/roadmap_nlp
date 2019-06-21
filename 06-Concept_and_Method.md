

# 6. Concept and Method

## 6.1 Overview

## 6.2 CNN

## 6.3 RNN

## 6.4 LSTM/GRU

### 6.4.1 LSTM

#### Paper

[Long short-term memory](https://www.bioinf.jku.at/publications/older/2604.pdf)

#### Article

- [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

- [Deep Learning for NLP Best Practices blog](http://ruder.io/deep-learning-nlp-best-practices/)

### 6.4.2 AWD-LSTM

AWD-LSTM 对 LSTM 模型进行了改进，包括在隐藏层间加入dropout ，加入词嵌入 dropout ，权重绑定等。建议**使用AWD-LSTM 来替代 LSTM 以获得更好的模型效果**。

#### Paper

[Regularizing and Optimizing LSTM Language Models - 2017](https://arxiv.org/abs/1708.02182)

#### Code

- [Github by Salesforce](https://github.com/salesforce/awd-lstm-lm)
  
- [Github by fastAI](https://github.com/fastai/fastai/blob/master/fastai/text/models/awd_lstm.py)

### 6.4.3 Adversarial LSTM

#### Paper

[Adversarial Training Methods for Semi-Supervised Text Classification - 2016](https://arxiv.org/abs/1605.07725)

核心思想：通过对word Embedding上添加噪音生成对抗样本，将对抗样本以和原始样本 同样的形式喂给模型，得到一个Adversarial Loss，通过和原始样本的loss相加得到新的损失，通过优化该新 的损失来训练模型，作者认为这种方法能对word embedding加上正则化，避免过拟合。

#### Practice

- [文本分类实战（七）—— Adversarial LSTM模型](https://www.cnblogs.com/jiangxinyang/p/10208363.html)

### 6.4.4 GRU

## 6.5 Boosting & Bagging & Stacking

#### Code

- <https://github.com/brightmart/text_classification/blob/master/a00_boosting/a08_boosting.py> (Tensorflow)

## 6.6 Summary
