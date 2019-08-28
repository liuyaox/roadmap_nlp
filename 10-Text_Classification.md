
# 10. Text Classification

YAO's: <https://github.com/liuyaox/text_classification> (Keras & PyTorch)


## 10.1 Overview

#### Paper

- [A Brief Survey of Text Mining: Classification, Clustering and Extraction Techniques - UGA2017](https://arxiv.org/abs/1707.02919)

- [Towards Explainable NLP: A Generative Explanation Framework for Text Classification - PKU2019](https://arxiv.org/abs/1811.00196)

    **Chinese**: [面向可解释的NLP：北大、哈工大等提出文本分类的生成性解释框架](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247489745&idx=5&sn=44e8bfeb5d9d9c1bb2906d941dfc063e)


#### Code

- 【Great】<https://github.com/yongzhuo/Keras-TextClassification> (Keras)

    中文长文本分类、短句子分类（Chinese Text Classification of Keras NLP, or sentence classify, long or short），字词句向量嵌入层（embeddings）和网络层（graph）构建基类
    
    模型：FastText, TextCNN，CharCNN，TextRNN, RCNN, DCNN, DPCNN, VDCNN, CRNN, Bert, Attention, DeepMoji, HAN, CapsuleNet, Transformer-encode, Seq2seq, ENT, DMN

    **Article**: 【Great!】[中文短文本分类实例1-12](https://blog.csdn.net/rensihui)

- <https://github.com/ShawnyXiao/TextClassification-Keras> (Keras)

    Text classification models implemented in Keras, including: FastText, TextCNN, TextRNN, TextBiRNN, TextAttBiRNN, HAN, RCNN, RCNNVariant, etc.

- <https://github.com/AlexYangLi/TextClassification> (Keras)

    All kinds of neural text classifiers implemented by Keras: TextCNN, DCNN, RCNN, HAN, DPCNN, VDCNN, MultiTextCNN, BiLSTM, RNNCNN, CNNRNN.

- 【Great】<https://github.com/wabyking/TextClassificationBenchmark> (PyTorch)

    A Benchmark of Text Classification in PyTorch

    模型：FastText, BasicCNN(KimCNN, MultiLayerCNN, MultiPerspectiveCNN), InceptionCNN, LSTM(BiLSTM, StackLSTM), LSTM with Attention(Self Attention, Quantum Attention), Hybrids between CNN and RNN (RCNN, C-LSTM), Transformer, Capsule, Quantum-inspired NN

- 【Great】<https://github.com/649453932/Chinese-Text-Classification-Pytorch> (PyTorch)

    中文文本分类，TextCNN，TextRNN，FastText，TextRCNN，BiLSTM_Attention，DPCNN，基于pytorch，开箱即用

- <https://github.com/songyingxin/TextClassification-Pytorch> (PyTorch)

    Pytorch + NLP, 一份友好的项目实践仓库

- <https://github.com/brightmart/text_classification> (Tensorflow)

    模型：FastText, TextCNN, TextRNN, TextRCNN, BERT, HAN, Attention, Transformer, Dynamic Memory Network, EntityNetwork, BiLSTMTextRelation, TwoCNNTextRelation, BiLSTMTextRelationTwoRNN

- <https://github.com/brightmart/ai_law> (Tensorflow)

    All kinds of baseline models for long text classificaiton (text categorization): HAN, TextCNN, DPCNN, CNN-GRU, GRU-CNN, Simple Pooling, Transformer(todo)

- <https://github.com/Edward1Chou/Textclassification> (Tensorflow)

    分别构建基于传统机器学习(LR, RF)的文本分类和基于深度学习(CNN)的文本分类系统，并在同一数据集上进行测试。

- <https://github.com/jiangxinyang227/textClassifier> (Tensorflow)

    模型：TextCNN, CharCNN, BiLSTM, BiLSTM+Attention, TextRCNN, adversarialLSTM, Transformer, ELMo, BERT

- <https://github.com/pengming617/text_classification> (Tensorflow)

    模型：TextCNN，TextRNN, LEAM, Transformer，Attention, fasttext, HAN等


#### Practice

- [文本分类实战系列文章](https://www.cnblogs.com/jiangxinyang/p/10207482.html) (Tensorflow)

- [阿里AI工程师教你如何用CNN RNN Attention解决大规模文本分类问题](https://www.sohu.com/a/130492867_642762)


#### Article

- 【Great】[用深度学习（CNN RNN Attention）解决大规模文本分类问题 - 综述和实践 - 2017](https://zhuanlan.zhihu.com/p/25928551)

    **YAO**: 类似于综述，有传统方法和深度学习方法，还有经验思考

- 【Great】[在文本分类任务中，有哪些论文中很少提及却对性能有重要影响的tricks](https://www.zhihu.com/question/265357659/answer/578944550)

- 【Great】[如何到top5%？NLP文本分类和情感分析竞赛总结](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247486159&idx=1&sn=522345e275df807942c7b56b0054fec9)

    **YAO**:

    | Layer | Function |
    | :-: | :-: |
    | LSTM/GRU | 解决依赖问题，适合第一层建模，缺点是慢 |
    | CNN | 配合池化抓取关键词特征 |
    | Capsules | 可以替代CNN，效果一般优于CNN |
    | MaxPooling | 只要关键特征，其他全部过滤 |
    | Attention | 突出关键特征，但难以过滤掉所有非重点特征，类似于池化机制 |

    由此，可得出以下若干小结论：

    - 对于短文本，LSTM/GRU+Capsules是一个不错的模型

    - 对于长文本，只使用CNN几乎没法用，最好在前面加一层LSTM/GRU，以及Attention机制


#### Library

- NeuralNLP-NeuralClassifier: <https://github.com/Tencent/NeuralNLP-NeuralClassifier>

    An Open-source Neural Hierarchical Multi-label Text Classification Toolkit from Tencent.


#### Competition

**2018-DC-“达观杯”文本智能处理挑战赛**:

> **单标签多分类**问题，共有19种分类，原始数据是字id序列和词id序列

- <https://github.com/ShawnyXiao/2018-DC-DataGrand-TextIntelProcess>

    Top1    任务是通过长文本的字和词的序列数据，判断文本类别。

    特征构建：TFIDF特征，LDA特征，LSI特征，Word2Vec特征

    传统模型：MultinomialNB, BernoulliNB, LinearSVC, GBDT, MLP, LightGBM, LR等 (对模型融合的提升是巨大的)

    深度学习模型：Enhanced GRU(GRU+Capsule), Reinforced Embedding(Capsule+Attention) + Reinforced GRU(GRU+Capsule+Attention), Reinforced CNN(CNN+Capsule+Attention) + Reinforce GRU

    模型技巧：Stacking

    **YAO**: 4个特征如何加工，如何整合？

- 【Great！】<https://github.com/nlpjoe/daguan-classify-2018> (Keras)

    Rank 18   使用模型：Attention, Attention+RNN, Capsule, ConvLSTM, DPCNN, LSTM-GRU, RCNN, SimpleCNN, TextCNN, TextRNN, LightGBM

    **YAO**: OK
    
    - Baseline：LR和LinearSVC，**特征只有TFIDF特征**，评估指标是Accuracy和F1值，都有非CV版和CV版

    - 模型：首先定义BasicModel, 基于此定义BasicDeepModel和BasicStaticModel，并实现通用方法如计算评估指标、模型训练与评估等；随后继承BasicDeepModel以实现各种深度学习模型，继承BasicStaticModel以实现XGBoost和LightGBM模型

    - 特征：自己训练Word2Vec, TFIDF特征，LSA特征，对于深度模型，未使用TFIDF,LSA这些结构化特征！

    - Trick: **数据增强(打乱原样本序列顺序生成新样本)**， 模型融合Stacking

- <https://github.com/hecongqing/2018-daguan-competition> (Keras)

    Rank 4

- <https://github.com/Rowchen/Text-classifier> (Tensorflow)

    Rank 8/3462

- <https://github.com/moneyDboat/data_grand> (PyTorch)

    Rank 10/3830


## 10.2 Multi-label Classification

#### Code

- <https://github.com/linjian93/toxic_comment_classification> (PyTorch)

    采用LSTM/C-LSTM/CNN等方法，对评论进行多标签分类

    **YAO**: 使用了PyTorch中的 Dataset 和 Dataloader，可以学习一下

- <https://github.com/junyu-Luo/Multi-label_classification> (Tensorflow)

    使用GRU+attention进行多标签二分类


#### Competition

**2017知乎看山杯 多标签 文本分类**:

> 300W个训练数据，每个样本标注为1个或多个Label，共1999个Label，20W测试数据，每个样本需要打5个Label

> 官方总结：[「2017知乎·看山杯机器学习挑战赛」结束，谁获奖了？知乎还会做什么？](https://zhuanlan.zhihu.com/p/28912353)

- <https://github.com/chenyuntc/PyTorchText> (PyTorch)

    Rank 1   模型和技术：TextCNN, TextRNN, TextRCNN, TextInception, MultiModel, 概率等权重融合

    **Article**: [知乎看山杯 夺冠记](https://zhuanlan.zhihu.com/p/28923961)

    **YAO**: 直接处理**1999标签二分类**，模型输出结构类似于**Dense(1999, 'sigmoid')**，注意激活是Sigmoid，而非Softmax！Loss使用torch.nn.MultiLabelSoftMarginLoss，Summary处的c.1类做法。

- <https://github.com/Magic-Bubble/Zhihu> (PyTorch)

    Rank 2  模型和技术：FastText, TextCNN, TextRNN, TextRCNN, HAN, 核心思路是：**使用类似AdaBoost的方式来修复单模型** + 模型融合

    **Article**: [2017知乎看山杯 从入门到第二](https://zhuanlan.zhihu.com/p/29020616)

    **YAO**: 问题定义、模型输出结构、Loss同Rank1

- <https://github.com/yongyehuang/zhihu-text-classification> (Tensorflow)

    Rank 6  模型和技术：TextCNN, BiGRU, HAN, FastText, RNN+CNN

    **Article**: [2017知乎看山杯总结(多标签文本分类)](https://blog.csdn.net/Jerr__y/article/details/77751885)

    **YAO**: 问题定义、模型输出结构同Rank1，Loss使用tf.nn.sigmoid_cross_entropy_with_logits

- <https://github.com/coderSkyChen/zhihu_kanshan_cup_2017> (Keras)

    Rank 9  模型和技术：TextCNN, VDCNN, LSTM, C-LSTM, BiLSTM, RCNN, RCNN+Attention(单模型得分最高), Multi-Channel, Multi-Loss

    **Article**: [大规模文本分类实践-知乎看山杯总结](http://coderskychen.cn/2017/08/20/zhihucup/)

    **YAO**: 问题转化：**1999标签二分类**-->**1999分类**，模型输出结构为**Dense(1999, 'softmax')**，Loss为categorical_crossentropy，Label编码为长度1999的01向量，应用时取概率值Top5，Summary处的c.2类做法。 但似有不妥：Softmax会过于突出1999中的某一个值？Sigmoid似乎更适合Multi-label类问题？

- <https://github.com/chenzhi1992/Multi-Label-Text-Classification> (Tensorflow)

    bilstm+attention, multi label text classify


#### Practice

- [Guide To Multi-Class Multi-Label Classification With Neural Networks In Python - 2017](https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/)

    **Chinese**: [keras解决多标签分类问题](https://blog.csdn.net/somtian/article/details/79614570)

    **YAO**: 讲述为什么多标签多分类问题要使用Sigmoid激活而非Softmax

- [Multi-label classification with Keras - 2018](https://www.pyimagesearch.com/2018/05/07/multi-label-classification-with-keras/)

    **Code**: <https://github.com/ItchyHiker/Multi_Label_Classification_Keras> (Keras)

    **Chinese**: [手把手教你用Keras进行多标签分类](https://blog.csdn.net/tMb8Z9Vdm66wH68VX1/article/details/81090757)

    **YAO**: 对于**3标签多分类**问题，比如标签1是衣服，有裙子、衬衫、卫衣3种，标签2是颜色，有红、蓝、黑、白4种，标签3是质地，有棉、丝2种，需要使用mlb=MultiLabelBinarizer处理这种标签取值，处理后mlb.classes_=3+4+2=9，即问题转化为**9标签二分类**问题！！！(本质理念：把多标签多分类问题，转化为**在每个标签上的二分类问题**！)
    
    - 模型搭建：activation='sigmoid'，类别数量=mlb.classes_

    - 模型编译：loss='binary_crossentropy', metrics=['accuracy']，因为目标是将每个输出标签视作一个独立伯努利分布，我们需要独立地惩罚每个输出节点！(而Softmax用于单标签多分类)

    - 模型应用：得到长度为9的概率向量prob，np.argsort(prob)[::-1][:3]即为最有可能的前3个标签(所对应的标签索引)。

    - 思考：3个标签及取值完全平铺开了，各新标签之间独立和并列，但**不用担心Top3标签中会有多个衣服或颜色或质地**，因为训练数据中压根就没有这种模式的数据，模型自然也不会凭空学到这种模式！Top3标签一定是衣服、颜色和质地各有一个！

- [Keras Issue: How to train a multi-label Classifier - 2015](https://github.com/keras-team/keras/issues/741)

- [Multi-label Text Classification using BERT – The Mighty Transformer - 2019](https://medium.com/huggingface/multi-label-text-classification-using-bert-the-mighty-transformer-69714fa3fb3d)

    搞定NLP领域的“变形金刚”！手把手教你用BERT进行多标签文本分类


#### Summary

分类问题总结：

- a. 单标签二分类：模型输出结构为Dense(1, activation='sigmoid')，Label编码为取值01的标量，应用时结果为1个概率值，判断其与阈值(一般为0.5)大小关系

- b. 单标签N分类：模型输出结构为Dense(N, activation='softmax')，Label编码为Onehot编码(长度为N取值01且有且只有1个1的一维向量, by LabelBinarizer/to_categorical)，应用时结果为N个概率值，取Top1

- c. N标签二分类

    - c.1 一个输出: **直接处理这类问题，模型输出结构为Dense(N, activation='sigmoid')**，Label编码为Multi-Onehot编码(长度为N取值01且不限01个数的一维向量, by MultiLabelBinarizer)，应用时结果为N个概率值，取TopK

    - c.2 一个输出：问题转化为N分类，类似于b，模型输出结构、Label编码同b，应用时结果同c.1

    - c.3 N个输出：每个输出都是a，模型输出结构、Label编码、应用时结果都同a  目前没见过有人这样处理，待尝试？

- d. N标签M分类

    - d.1 一个输出：问题转化为NM标签二分类，同c.1

    - d.2 一个输出：问题转化为NM分类，同c.2

    - d.3 N个输出：每个输出都是b，模型输出结构、Label编码、应用时结果都同b  目前没见过有人这样处理，待尝试？

疑问：使用Sigmoid处理多标签分类问题，前提假设是各标签之间是并列且独立的，这合理么？相同的模型主体，只是输出层开始不同，那么在模型主体处会学习各标签之间隐含的关系么？


## 10.3 FastText

#### Paper

[Bag of Tricks for Efficient Text Classification - Facebook2016](https://arxiv.org/abs/1607.01759)

#### Code

- <https://github.com/facebookresearch/fastText>

- <https://fasttext.cc/>

#### Library

- gensim: <https://radimrehurek.com/gensim/models/fasttext.html>

- skift: <https://github.com/shaypal5/skift>

    scikit-learn wrappers for Python fastText

#### Practice

- 直接使用Python的fastText库
  
  ```Python
  from fastText import train_supervised

  model = train_supervised('train.txt', epoch=25, lr=1.0, wordNgrams=2, minCount=1, loss='hs')
  N, P, R = model.test('test.txt', k=1)
  model.test_label('test.txt', k=1, threshold=0.5)  # Return precision and recall for each label
  model.predict('五彩 手机壳', k=1, threshold=0.5)
  model.save_model("model.bin")
  ```

  其中，train.txt每行Schema：用__label__表示label，使用**空格逗号空格**分隔label和text，其中label是类别名字或编号，text是用**空格**分隔的各个word

  > Schema: __label__0 , 贴膜 金刚膜

- 自己搭建模型，不确定是否合理([Reference](https://github.com/ShawnyXiao/TextClassification-Keras/blob/master/model/FastText/fast_text.py))

  **Structure**: inputs-->Embedding-->SpatialDropout1D-->GlobalAveragePooling1D-->Dense-->Softmax/Sigmoid  ???

- <https://github.com/panyang/fastText-for-AI-Challenger-Sentiment-Analysis>

    AI Challenger 2018 Sentiment Analysis Baseline with fastText

#### Article

- [文本分类需要CNN？No！fastText完美解决你的需求 - 2017](https://blog.csdn.net/weixin_36604953/article/details/78324834)

- [一行代码自动调参，支持模型压缩指定大小，Facebook升级FastText - 2019](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650768854&idx=2&sn=39fcf30127c3291ec4ab90904bd32b45)


## 10.4 TextCNN

#### Paper

- [Convolutional Neural Networks for Sentence Classification - NYU2014](https://arxiv.org/abs/1408.5882)

  **Structure**: Input -> Embedding -> (Conv1D -> GlobalMaxPooling1D) * n_filter_size -> Concatenate -> Dropout -> Dense -> Dropout -> Dense，其中GlobalMaxPooling1D与MaxPooling1D-->Flatten功能相同。

  ![textcnn_structure](./image/textcnn_01.png)

  **Key Points**：

  TextCNN的输入Embedding有3种：

  - 随机初始化

  - Static Pre-trained

  - Two-Channel: Static Pre-trained + Dynamic Embedding，其中Dynamic有2种：Pre-trained + Finetuning和随机初始化，Dynamic的作用是，通过对语料的学习，模型可以得到task-specific的信息。

- 对TextCNN进行调优
  
    **Paper**: [A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification - UTEXAS2016](https://arxiv.org/abs/1510.03820)

    论文基于one-layer CNNs，研究了Input Word Vectors、Filter Region Size、Number of Feature Maps for Each Filter Region Size、Activation Function、Pooling Strategy和Regularization等对模型性能的影响，有助于我们参考以选择适合自己的参数。

#### Code

- Yao(Keras)

    ```Python
    from keras.layers import Input, Conv1D, GlobalMaxPooling1D, Concatenate, Dropout, Dense
    from keras.models import Model

    def TextCNNModel(input_shape, embedding_layer):
        sentences = Input(input_shape, dtype='int32')
        X = embedding_layer(sentences)                          # (None, maxlen, emb_dim)
        X = BatchNormalization()(X)
        Xs = []
        for fsize in [2, 3, 4]:
            Xi = Conv1D(128, fsize, activation='relu')(X)       # (None, maxlen-fsize+1, 128)
            # Xi = Conv2D(128, (fsize, emb_dim), activation='relu')(X)      # 同上行
            Xi = GlobalMaxPooling1D()(Xi)                       # (None, 128)
            # Xi = Flatten()(MaxPool1D(pool_size=(maxlen-fsize+1, ))(Xi))   # 同上行
            Xs.append(Xi)
        X = Concatenate(axis=-1)(Xs)                            # (None, 128*3)
        X = Dropout(0.5)(X)
        X = Dense(units=32, activation='relu')(X)               # (None, 32)
        X = Dense(units=1, activation='sigmoid')(X)             # (None, 1)
        return Model(sentences, X)
    ```

- <https://github.com/alexander-rakhlin/CNN-for-Sentence-Classification-in-Keras> (Keras)

- <https://github.com/jiangxinyang227/textClassifier> (Tensorflow)

#### Practice

- [tf18: 根据姓名判断性别](https://blog.csdn.net/u014365862/article/details/53869732)


## 10.5 CharCNN

#### Paper

[Character-level Convolutional Networks for Text Classification - NYU2016](https://arxiv.org/abs/1509.01626)

#### Code

- <https://github.com/jiangxinyang227/textClassifier> (Tensorflow)

- <https://github.com/mhjabreel/CharCNN> (Tensorflow)

#### Practice

[文本分类实战（三）—— charCNN模型](https://www.cnblogs.com/jiangxinyang/p/10207686.html)


## 10.6 TextRNN

其实就是使用LSTM或BiLSTM按一般方式去处理Sequnce，没什么花样。

**Structure1**: Embedding -> BiLSTM -> Average -> Softmax

![textrnn_structure1](./image/textrnn_01.png)

**Structure2**: Embedding -> BiLSTM -> Dropout -> LSTM -> Dropout -> Softmax

![textrnn_structure2](./image/textrnn_02.png)

**Structure3**: Embedding -> BiLSTM -> Attention -> Softmax

#### Code

<https://github.com/brightmart/text_classification> (Tensorflow)

#### Practice

- <https://github.com/cjymz886/text_rnn_attention> (Tensorflow)

    嵌入Word2vec词向量的RNN+ATTENTION中文文本分类


## 10.7 TextRCNN

#### Paper

[Recurrent Convolutional Neural Networks for Text Classification - CAS2015](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/view/9745/9552)

**Structure**

Recurrent Structure(Convolutional Layer) -> MaxPooling -> Dense(Softmax)

Representation of Current Word = [Left_context_vector, Current_word_embedding, Right_context_vecotor]

For Left_context Cl(w4), it uses a recurrent structure, a non-linearity transform of previous word w3('stroll') and left previous context Cl(w3)

![textrcnn_structure](./image/textrcnn_01.png)

#### Code

- <https://github.com/brightmart/text_classification> (Tensorflow)

    核心代码：生成Left_context_vector和Right_context_vector，注意是双向的，顺序生成各个Left_context_vector，倒序生成各个Right_context_vector

    ```Python
    # left与right相似，本质上都是：前Context与前Embedding相加后激活
    # Context(i) = Activation(W * Context(i-1) + W_s * Embedding(i-1))

    # Size：context_left-[batch_size, embed_size]  W_l-[embed_size, embed_size]  embedding_previous-[batch_size, embed_size]
    def get_context_left(self, context_left, embedding_previous):
        left_h = tf.matmul(context_left, self.W_l) + tf.matmul(embedding_previous, self.W_sl)
        context_left = self.activation(left_h)
        return context_left    # [None, embed_size]

    def get_context_right(self, context_right, embedding_afterward):
        right_h = tf.matmul(context_right, self.W_r) + tf.matmul(embedding_afterward, self.W_sr)
        context_right = self.activation(right_h)
        return context_right    # [None, embed_size]
    ```

- <https://github.com/jiangxinyang227/textClassifier> (Tensorflow)

   解读：[文本分类实战（六）—— RCNN模型](https://www.cnblogs.com/jiangxinyang/p/10208290.html) (Tensorflow)

- <https://github.com/roomylee/rcnn-text-classification> (Tensorflow)

- <https://github.com/airalcorn2/Recurrent-Convolutional-Neural-Network-Text-Classifier> (Keras)

    Slightly modified Keras implementation of TextRCNN

    **YAO**: 论文里公式1和2实现得似乎不对，处理left和right时没有考虑word embedding ???


## 10.8 VDCNN

#### Paper

[Very Deep Convolutional Networks for Text Classification - Facebook2017](https://arxiv.org/abs/1606.01781)

#### Code

- <https://github.com/lethienhoa/Very-Deep-Convolutional-Networks-for-Natural-Language-Processing> (Tensorflow)


## 10.9 DRNN

#### Paper

[Disconnected Recurrent Neural Networks for Text Categorization - iFLYTEK2018](https://aclweb.org/anthology/P18-1215)

#### Code

- <https://github.com/zepingyu0512/disconnected-rnn> (Keras)

- <https://github.com/liuning123/DRNN> (Tensorflow)


## 10.10 DPCNN

#### Paper

[Deep Pyramid Convolutional Neural Networks for Text Categorization - Tencent2017](https://ai.tencent.com/ailab/media/publications/ACL3-Brady.pdf)

单词级别的深层CNN模型，来捕捉文本的全局语义表征，该模型在不增加太多的计算开销的情况下，通过增加网络深度可以获得最佳的性能

#### Code

- <https://github.com/Cheneng/DPCNN> (PyTorch)

    A simple version for DPCNN


## 10.11 multiChannelCNN & DeepCNN & LSTM-CNN & Tree-LSTM

multiChannelCNN: 同一个Sequence，用多个Embedding，其一是随机初始化，其二是预训练Embedding如Word2Vec或GloVe等，然后Concatenate

DeepCNN: 其实是multiLayerCNN

LSTM-CNN:

Tree-LSTM:

### 10.11.1 multiChannelCNN

#### Code

- <https://github.com/zenRRan/Sentiment-Analysis/blob/master/models/multi_channel_CNN.py> (PyTorch)

#### Article

- [详解文本分类之多通道CNN的理论与实践](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247485817&idx=1&sn=fc0c3a4a5f6afc111b57045ff929cc2b)


### 10.11.2 DeepCNN

#### Code

- <https://github.com/zenRRan/Sentiment-Analysis/blob/master/models/Multi_layer_CNN.py> (PyTorch)

#### Article

- [详解文本分类之DeepCNN的理论与实践](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247485837&idx=1&sn=1b8f6e0f1ec21d3b73871179f0471428)


### 10.11.3 LSTM-CNN


### 10.11.4 Tree-LSTM


## 10.12 Others

#### Paper

- [Document Modeling with Gated Recurrent Neural Network for Sentiment Classification (EMNLP 2015)](https://www.aclweb.org/anthology/D15-1167)

    利用GRU对文档进行建模的情感分类模型：首先将文本映射为向量，然后利用CNN/LSTM进行句子向量表示，输入到GRU中，得到文档向量表示，最后输送给Softmax层，得到标签的概率分布。

- [Recurrent Neural Network for Text Classification with Multi-Task Learning (IJCAI 2016)](https://arxiv.org/abs/1605.05101)

    针对文本多分类任务，提出了基于RNN的三种不同的共享信息机制对具有特定任务和文本进行建模：所有任务共享同1个LSTM层，每个任务具有自己独立的LSTM层，除1个共享的BLSTM层外每个任务有自己独立的LSTM层。

- DeepMoji - [Using millions of emoji occurrences to learn any-domain representations for detecting sentiment, emotion and sarcasm - MIT2017](https://arxiv.org/abs/1708.00524)

    使用数以百万计的表情符号来学习任何领域的表情符号来检测情绪、情绪和讽刺，提出了DeepMoji模型，并取得了具有竞争性的效果。同时，DeepMoji模型在文本分类任务上也可以取得不错的结果。

    **Github**: <https://github.com/bfelbo/DeepMoji> (Keras)

- [Investigating Capsule Networks with Dynamic Routing for Text Classification - CAS2018)](https://arxiv.org/abs/1804.00538)

    一种基于胶囊网络的文本分类模型，并改进了Sabour等人提出的动态路由，提出了三种稳定动态路由。

    **Github**: <https://github.com/andyweizhao/capsule_text_classification> (Tensorflow, Keras)

- RNN-Capsule - [Sentiment Analysis by Capsules -THU2018](http://coai.cs.tsinghua.edu.cn/hml/media/files/p1165-wang.pdf)

    RNN-Capsule首先使用RNN捕捉文本上下文信息，然后将其输入到capsule结构中：用注意力机制计算capsule表征 --> 用capsule表征计算capsule状态的概率 --> 用capsule表征以及capsule状态概率重构实例的表征

- GCN - [Graph Convolutional Networks for Text Classification - Northwestern2018)](https://arxiv.org/abs/1809.05679)

    一种基于graph convolutional networks(GCN)进行文本分类，构建了一个包含word节点和document节点的大型异构文本图，显式地对全局word利用co-occurrence信息进行建模，然后将文本分类问题看作是node分类问题。

    **Github**: <https://github.com/yao8839836/text_gcn> (Tensorflow)


## 10.13 Traditional Method
