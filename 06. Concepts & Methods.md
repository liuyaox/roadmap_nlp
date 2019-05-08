

## 6. Concepts & Methods
### 6.1 Overview

### 6.2 CNN

### 6.3 RNN

### 6.4 LSTM/GRU

### 6.5 Adversarial LSTM
**Paper**

Adversarial Training Methods for Semi-Supervised Text Classification - 2016 (https://arxiv.org/abs/1605.07725)

核心思想：通过对word Embedding上添加噪音生成对抗样本，将对抗样本以和原始样本 同样的形式喂给模型，得到一个Adversarial Loss，通过和原始样本的loss相加得到新的损失，通过优化该新 的损失来训练模型，作者认为这种方法能对word embedding加上正则化，避免过拟合。

**Practices**

a. 文本分类实战（七）—— Adversarial LSTM模型 (https://www.cnblogs.com/jiangxinyang/p/10208363.html)

### 6.6 Boosting & Bagging & Stacking

**Source**

https://github.com/brightmart/text_classification/blob/master/a00_boosting/a08_boosting.py (Tensorflow)

### 6.7 Summary
