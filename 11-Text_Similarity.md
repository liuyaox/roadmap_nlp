

# 19. Text Similarity

## 19.1 Overview

#### Tool

- [百度短文本相似度](http://ai.baidu.com/tech/nlp/simnet)


## 19.2 Paper

- [Siamese Recurrent Architectures for Learning Sentence Similarity - MIT2016](https://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf)

    **Code**: <https://github.com/LuJunru/Sentences_Pair_Similarity_Calculation_Siamese_LSTM> (Keras)

    **Code**: [基于Simaese LSTM的句子相似度计算](https://blog.csdn.net/android_ruben/article/details/78427068) (Keras)

    **Code**: <https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb> (Keras)

    **Article**: [How to predict Quora Question Pairs using Siamese Manhattan LSTM - 2017](https://medium.com/mlreview/implementing-malstm-on-kaggles-quora-question-pairs-competition-8b31b0b16a07)

    **Article**: [中文翻译](https://www.jianshu.com/p/f3d0d94a4913?utm_campaign)

- [Learning Text Similarity with Siamese Recurrent Networks - 2016](http://www.aclweb.org/anthology/W16-16)

    **Code**: 
    
    <https://github.com/likejazz/Siamese-LSTM> (Keras)

    <https://github.com/eliorc/Medium/blob/master/MaLSTM.ipynb> (Keras)

    <https://github.com/dhwajraj/deep-siamese-text-similarity> (Tensorflow)

    <https://github.com/vishnumani2009/siamese-text-similarity> (Tensorflow)

    <https://github.com/aditya1503/Siamese-LSTM> (Theano)


## 19.3 Practice

- 【Great】<https://github.com/RandolphVI/Text-Pairs-Relation-Classification> (Tensorflow)

    Text Pairs (Sentence Level) Classification (Similarity Modeling) Based on Neural Network

    模型有：ABCNN, ANN, CNN, CRNN, FastText, HAN, RCNN, RNN, SANN

    **YAO**:

    模型很丰富，且具有结构可视化结果，待看……

- <https://github.com/yanqiangmiffy/sentence-similarity> (Keras)

    问题句子相似度计算，即给定客服里用户描述的两句话，用算法来判断是否表示了相同的语义。

    **YAO**: 里面提到了5个文本相似度计算的比赛

- <https://github.com/liuhuanyong/SentenceSimilarity>

    基于同义词词林，知网，指纹，字词向量，向量空间模型的**句子**相似度计算

- <https://github.com/ashengtx/CilinSimilarity>

    **Word** similarity computation based on Tongyici Cilin

- <https://github.com/BiLiangLtd/WordSimilarity>

    基于哈工大同义词词林扩展版的**单词**相似度计算方法

    **Article**: [基于同义词词林扩展版的词语相似度计算](http://codepub.cn/2015/08/04/Based-on-the-extended-version-of-synonyms-Cilin-word-similarity-computing/)

- <https://github.com/PengboLiu/Doc2Vec-Document-Similarity>

    利用Doc2Vec计算文本相似度

- <https://github.com/cjymz886/sentence-similarity>

    对四种句子/文本相似度计算方法进行实验与比较: cosine, cosine+idf, bm25, jaccard

- <https://github.com/liuhuanyong/SiameseSentenceSimilarity>

    SiameseSentenceSimilarity,个人实现的基于Siamese bilstm模型的相似句子判定模型,提供训练数据集和测试数据集.

- <https://github.com/fssqawj/SentenceSim>

    中文短文句相似读, 2016年项目，比较传统，方法有：基于知网、onehot向量模型、基于Word2Vec、基于哈工大SDP、融合算法、LSTM


## 19.4 Competition

- <https://github.com/Leputa/CIKM-AnalytiCup-2018> (Tensorflow)

    CIKM AnalytiCup 2018 – 阿里小蜜机器人跨语言短文本匹配算法竞赛 – Rank12方案

    判断不同语言的两个**问句**语义是否相同。

- <https://github.com/ziweipolaris/atec2018-nlp> (Keras, PyTorch)

    ATEC2018 NLP赛题，判断两个**问句**是否意思相同

- <https://github.com/zake7749/CIKM-AnalytiCup-2018> (Tensorflow & Keras)

    [ACM-CIKM] 2nd place solution at CIKM AnalytiCup 2018, a task for determining **short text** similarities

**2018atec蚂蚁金服NLP智能客服比赛**:

> 给定客服里用户描述的两句话，判断**问句**相似度

- <https://github.com/zle1992/atec> (Keras)

    Rank 16/2631

- <https://github.com/Lapis-Hong/atec-nlp> (PyTorch)


## 9.5 Traditional Method

### 9.5.1 Simhash

#### Article

- [simhash与重复信息识别 - 2011](https://grunt1223.iteye.com/blog/964564simhash与重复信息识别)