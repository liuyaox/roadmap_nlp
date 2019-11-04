# 25. Tricks and Tips

## 25.1 Overview

#### Article

- [Rules of Machine Learning](https://developers.google.cn/machine-learning/rules-of-ml/)

    谷歌机器学习43条规则：机器学习工程的最佳实践经验

- [深度学习的一些经验总结和建议| To do v.s Not To Do - 2019](https://mp.weixin.qq.com/s?__biz=MzUyOTU2MjE1OA==&mid=2247485683&idx=2&sn=759fb3d6bcc3e43dfb9b967e83dbe7dc)

- 【Great】[分分钟带你杀入Kaggle Top 1% - 2017](https://zhuanlan.zhihu.com/p/27424282)

- [Ten Techniques Learned From fast.ai](https://blog.floydhub.com/ten-techniques-from-fast-ai/)

    **Chinese**：[称霸Kaggle的十大深度学习技巧](https://mp.weixin.qq.com/s?__biz=MzU3NjE4NjQ4MA==&mid=2247484049&idx=1&sn=56bb5d502b2ed6e1cbc8b7405b52ad20)

- [训练神经网络的方法分享-Andrej Karpathy](https://mp.weixin.qq.com/s?__biz=MzIxNDgzNDg3NQ==&mid=2247485845&idx=1&sn=13620bb17dc0fd75d71100dd84cce59d)

- [12 Key Lessons from ML researchers and practitioners](https://towardsml.com/2019/04/09/12-key-lessons-from-ml-researchers-and-practitioners/)

    **Chinese**：[关于机器学习实战，那些教科书里学不到的 12 个“民间智慧”](https://mp.weixin.qq.com/s/jTdhb00HYhfLYiW1q14gtg)

- [Kaggle 数据挖掘比赛经验分享 - 2017](https://zhuanlan.zhihu.com/p/26820998)

- [Three pitfalls to avoid in machine learning - 2019](https://www.nature.com/articles/d41586-019-02307-y)

    **Chinese**: [谷歌高级研究员Nature发文：避开机器学习三大坑](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650766995&idx=4&sn=2417570839b7ccb2e630cef36a363e0a)

- [37 Reasons why your Neural Network is not working - 2017](https://medium.com/@slavivanov/4020854bd607)

    **Chinese**: [训练的神经网络不工作？一文带你跨过这37个坑](https://baijiahao.baidu.com/s?id=1573879911456703)

- 【Great】[如何训练一个性能不错的深度神经网络 - 2018](http://www.sohu.com/a/222666303_609569)

    **YAO**：HEREHEREHEREHEREHEREHEREHEREHEREHEREHERE

- [Ten Techniques Learned From fast.ai - 2018](https://blog.floydhub.com/ten-techniques-from-fast-ai/)

    **Chinese**: [10大称霸Kaggle的深度学习技巧](https://blog.csdn.net/yH0VLDe8VG8ep9VGe/article/details/81463870)

- [Weight Initialization in Neural Networks: A Journey From the Basics to Kaiming - 2019](https://towardsdatascience.com/weight-initialization-in-neural-networks-a-journey-from-the-basics-to-kaiming-954fb9b47c79)

    神经网络中的权重初始化一览：从基础到Kaiming


#### Practice

- 【Great!!】[How I made top 0.3% on a Kaggle competition](https://www.kaggle.com/lavanyashukla01/how-i-made-top-0-3-on-a-kaggle-competition)

    **YAO**: 完整展示了算法项目的流程及各种细节，强烈建议过一遍！！！

- [万字长文总结机器学习的模型评估与调参 - 2019](https://mp.weixin.qq.com/s?__biz=MzIwOTc2MTUyMg==&mid=2247492923&idx=2&sn=15fd5960ca20f1bd81916e625f929448)


#### Library

- <https://www.scikit-yb.org>

    Yellowbrick: 结合了Scikit-Learn和Matplotlib并且最好得传承了Scikit-Learn文档，对你的模型进行可视化！

    **Chinese**: <http://www.scikit-yb.org/zh/latest/>


## 25.2 Parameter Tuning

#### Article

- 【Great】[听说你不会调参？以TextCNN为例的优化经验Tricks汇总 - 2019](https://mp.weixin.qq.com/s/8aTd8xNRQeIUWTxfqxyp-A)

    **YAO**：OK

    - 调优基本方法：用模型**在Test上做Badcase分析**以发现**共性原因**。一般从2个角度来解决，一是特征，二是模型结构。对于特征，要做一些特征分析，以找到对某些类别有区分度的特征。要考虑特征与特征之间的结合，如何判断与测试集结果是否相关。

    - Badcase原因：一般有4种，分别是**标注错误、数据分布不平衡、缺乏先验知识、过于依赖A特征**

    - 原因归类和排序：按照Badcase原因进行归类，并按频次从高到低排列，也是需要解决问题的优先级。假设依次是：
      - 训练集与测试集部分特征提取方式不一致(前者是离线使用Python，后者在线上使用Java，占比40%)
      - 结果过于依赖A特征(但A特征本身的准确率只有88%左右，导致模型有瓶颈，占比30%)
      - 泛化能力差('我要看黄渤的电影'预测正确，'我想看周星驰的电影'预测错误，占比20%)
      - 缺乏先验知识(比如青花瓷更容易出现在音乐中，而非baike中，但训练数据中并没有特征可体现这一先验知识，占比10%)

    - 特征提取方式不一致：统一训练集和测试集的代码，共用一套，并且使用的数据源也要一样，提升了0.4%
      
    - 过于依赖A特征：基本思想是**减少A特征**，dropout和数据预处理时都可以
      - 在模型结构的FC后增加dropout(keep_rate=0.7)，**随机扔掉30%**的所有特征(包括A特征)，让训练结果与A特征不强相关，提升0.11%
      - 在数据预处理时，随机扔掉10%的A特征(数据增广时设置一概率值，若<0.1则不输出A特征，否则输出)，相比dropout，此处只扔掉了A特征，更有针对性，提升了0.29%

    - 泛化能力差：针对部分文本，增加**槽位抽取**，比如把'黄渤'和'周星驰'都映射为artist，则两句话都变成了'我要看artist的电影'和'我想看artist的电影'，则就不存在泛化能力问题了，当然前提是槽位抽取的准确率要过关！

    - 缺乏先验知识：引入200万**外部词表**，同时计算'青花瓷'在整体语料中出现在其他类别的频率来部分解决，提升了0.5%

    - 模型优先实操记录：
      - Baseline: 默认结构和参数的TextCNN，test_acc=85.14%
      - embed_dim: 128-->64，希望了解不同dim对结果的影响，diff=-0.17%
      - dropout: 在FC后添加一个dropout(0.7)，diff=0.11%；改为dropout(0.5)，diff=-0.04%，说明不宜扔掉太多特征
      - 预处理: 随机扔掉10%的A特征，diff=0.29%；改为随机扔掉20%，diff=0.28%，并不优于10%，故选择10%
      - 训练集和测试集统一：在预处理基础上，统一两者特征抽取的代码和数据源，diff=0.64%
      - 增加特征：在统一基础上，进一步分析B特征对Badcase有很好的区分度，将B引入训练，diff=1.14%

- [Complete Machine Learning Guide to Parameter Tuning in Gradient Boosting (GBM) in Python - 2016](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)

- [深度学习中训练参数的调节技巧 - 2018](https://www.jianshu.com/p/4eff17d9c4ce)

- [step-by-step: 夕小瑶版神经网络调参指南（上）- 2018](https://mp.weixin.qq.com/s?src=11&timestamp=1572781504&ver=1952&signature=bH5H*UEOjeFIKeFi-0-eRWIw6HdnWjwNCKjCJat4BVktOECHS-s-U8cJQOUbTC8PpwxBpJ2QteN9CioKKa8XYEH3qf*Arh2HysXPxRbA7c*zT2rskeybxOlhkUjFaepa)

    **YAO**: HEREHEREHEREHEREHEREHEREHEREHEREHERE

- [Practical guide to hyperparameters search for deep learning models - 2018](https://blog.floydhub.com/guide-to-hyperparameters-search-for-deep-learning-models/)

    超参数搜索不够高效？这几大策略了解一下

- [深度学习网络调参技巧](https://zhuanlan.zhihu.com/p/24720954)

- [调参是否能对深度学习模型的性能有极大提升？](https://www.zhihu.com/question/339287819)

- [是否有办法提高深度学习的调参效率？](https://www.zhihu.com/question/339102039)

- [有哪些deep learning（rnn、cnn）调参的经验](https://www.zhihu.com/question/41631631)

    **YAO**: HEREHEREHEREHEREHEREHEREHEREHEREHERE

- [神经网络训练trick](https://zhuanlan.zhihu.com/p/59918821)

- [你在训练RNN的时候有哪些特殊的trick？](https://www.zhihu.com/question/57828011)

    **YAO**: HEREHEREHEREHEREHEREHEREHEREHEREHERE


## Library

- <http://hyperopt.github.io/>

    hyperopt: 机器学习调参神器

    **Article**: [Optimizing hyperparams with hyperopt](http://fastml.com/optimizing-hyperparams-with-hyperopt/)

    **Article**: [python调参神器hyperopt](https://blog.csdn.net/qq_34139222/article/details/60322995)


## 25.3 Finetuning

#### Article

- A Comprehensive guide to Fine-tuning Deep Learning Models in Keras -2016
  - [Part1](https://flyyufelix.github.io/2016/10/03/fine-tuning-in-keras-part1.html)
  - [Part2](https://flyyufelix.github.io/2016/10/03/fine-tuning-in-keras-part2.html)


## 25.4 Learning Rate

#### Article

- [Improving the way we work with learning rate - 2017](https://techburst.io/improving-the-way-we-work-with-learning-rate-5e99554f163b)

- [Using Learning Rate Schedules for Deep Learning Models in Python with Keras - 2016](https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/)

- [Setting the learning rate of your neural network - 2018](https://www.jeremyjordan.me/nn-learning-rate/)

- [Understanding Learning Rates and How It Improves Performance in Deep Learning - 2018](https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10)


## 25.5 Model Ensembling

### 25.5.1 Overview

#### Paper

- [Snapshot Ensembles: Train 1, get M for free - 2017](https://arxiv.org/abs/1704.00109)

#### Article

- [KAGGLE ENSEMBLING GUIDE - 2015](https://mlwave.com/kaggle-ensembling-guide/)

- [Introduction to Python Ensembles - 2018](https://www.dataquest.io/blog/introduction-to-ensembles/)

#### Library

- <https://github.com/yzhao062/combo>

    A Python Toolbox for Machine Learning Model Combination

    **Doc**: <https://pycombo.readthedocs.io/en/latest/>

    **Article**: [大部分机器学习算法具有随机性，只需多次实验求平均值即可吗？](https://www.zhihu.com/question/328157418/answer/746533382)


### 25.5.2 Boosting

#### Code

- <https://github.com/brightmart/text_classification/blob/master/a00_boosting/a08_boosting.py> (Tensorflow)


### 25.5.3 Bagging


### 25.5.4 Stacking 

对训练好的基学习器的应用结果进行**非线性融合**(输入并训练一个新的学习器)

偷懒的话，可直接使用 **Out-of-fold(OOF)** 做 Stacking

**解读方式1：先循环KFold，然后再循环各基模型**

![](https://raw.githubusercontent.com/liuyaox/ImageHosting/master/for_markdown/Stacking.png)

定义训练集P={(x1, y1), (x2, y2), ..., (xm, ym)}，测试集Q，基模型E1, E2, ..., ET，主模型E，则伪代码如下：

```
for P1, P2 in KFold(P) do       # 训练集拆分：训练集分成P1(k-1 folds)和P2(1 fold)，分别用于训练和验证，要遍历所有KFold
    for t = 1, 2, ..., T do
        Et.train(P1)            # 基模型训练：使用P1训练每个基模型
        prt = Et.apply(P2)      # 基模型应用：基模型应用于P2生成prt
        qrt = Et.apply(Q)       # 基模型应用：基模型应用于Q 生成qrt
    
    pr = [pr1, pr2, ..., prT]   # 收集每个基模型应用结果prt，用于主模型训练，只是1/K份的训练数据
    qr = [qr1, qr2, ..., qrT]   # 收集每个基模型应用结果qrt，用于主模型测试，是全份的测试数据

PR = concate(pr)                # 训练数据纵向堆叠，共同组成总训练数据，共T列，表示T个新特征
QR = average(qr)                # 测试数据纵向求均值，当作最终测试数据，列同上

E.train(PR)                     # 主模型训练
score = E.evaluate(QR)          # 主模型测试

Y(x) = E.apply(E1.apply(x), E2.apply(x), ..., ET.apply(x))  # 全流程应用
```

注意：各Fold的结果是**纵向堆叠或纵向均值**

**解读方式2：先循环各基模型，然后再循环KFold**

每个基模型内先做交叉验证，交叉验证的valid结果纵向拼接为1列，test结果求均值，共同组成该基模型的结果，**表示1列特征**，最后各基模型的结果**横向拼接成T列**

#### Article

- [模型融合之stacking&blending](https://zhuanlan.zhihu.com/p/42229791)

    **Code**: <https://github.com/InsaneLife/MyPicture/blob/master/ensemble_stacking.py>

- [图解Blending&Stacking](https://blog.csdn.net/sinat_35821976/article/details/83622594)

- [为什么做stacking ensemble的时候需要固定k-fold？](https://www.zhihu.com/question/61467937/answer/188191424)

- [A Kaggler's Guide to Model Stacking in Practice](http://blog.kaggle.com/2016/12/27/a-kagglers-guide-to-model-stacking-in-practice/)


### 25.5.5 Blending

理念与Stacking比较类似，模型分为两层，不过比Stacking简单一些，不需要通过KFold这种CV策略来获得主模型的特征，而是建立一个Holdout集，直接使用不相交的数据集用于两层的训练。以两层Blending为例，详细来说为：

Step1: 训练集划分为两部分P1和P2，测试集为Q

Step2: 用P1训练每个基模型

Step3: 基模型应用于P2的结果，训练主模型

Step4: 基模型应用于Q的结果，测试主模型，模型应用时与测试流程一样。

#### Article

- [Blending 和 Stacking - 2018](https://blog.csdn.net/u010412858/article/details/80785429)


## 25.6 Pseudo Labeling

训练集训练好的模型应用于测试集后的结果(即伪标签)，与训练集混合在一起后重新训练模型。？

一般模型能忍受10%的噪音，所以不要把所有测试集与训练集混合，建议保持比例在10:1左右。？

#### Article

- [A Simple Pseudo-Labeling Function Implementation in Keras - 2017](https://shaoanlu.wordpress.com/2017/04/10/a-simple-pseudo-labeling-function-implementation-in-keras/)

- [标签传播算法（Label Propagation）及Python实现 - 2015](https://blog.csdn.net/zouxy09/article/details/49105265)


## 25.7 Engineering

#### Article

- [机器学习框架上的一些实践 - 2019](https://zhuanlan.zhihu.com/p/76541337)


## 25.8 Data Augmentation

#### Paper

- [EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks - USA2019](https://arxiv.org/abs/1901.11196)

    **Article**: [NLP中一些简单的数据增强技术](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650411500&idx=2&sn=76e635526015ccecd14a1436bda55e2c)

#### Article

- [These are the Easiest Data Augmentation Techniques in NLP you can think of — and they work - 2019](https://towardsdatascience.com/these-are-the-easiest-data-augmentation-techniques-in-natural-language-processing-you-can-think-of-88e393fd610)


## 25.9 GPU

#### Article

- [深度学习的GPU：深度学习中使用GPU的经验和建议 - 2018](https://blog.csdn.net/sinat_36458870/article/details/78946030)

- [给训练踩踩油门——Pytorch加速数据读取 - 2019](https://mp.weixin.qq.com/s/jXaItEwH10-reaiH2pKztw)

#### Practice

- [在深度学习中喂饱gpu - 2019](https://zhuanlan.zhihu.com/p/77633542)

    **Code**: <https://github.com/tanglang96/DataLoaders_DALI> (PyTorch)

- [GPU 显存不足怎么办？- 2019](https://zhuanlan.zhihu.com/p/65002487)


## 25.10 Others

#### Article

- [9个大数据竞赛思路分享 - 2016](https://blog.csdn.net/Bryan__/article/details/51713596)

