# 25. Engineering and Tricks

## 25.1 Overview

#### Article

- [深度学习的一些经验总结和建议| To do v.s Not To Do - 2019](https://mp.weixin.qq.com/s?__biz=MzUyOTU2MjE1OA==&mid=2247485683&idx=2&sn=759fb3d6bcc3e43dfb9b967e83dbe7dc)

- [分分钟带你杀入Kaggle Top 1% - 2017](https://zhuanlan.zhihu.com/p/27424282)

- [Ten Techniques Learned From fast.ai - 2018](https://blog.floydhub.com/ten-techniques-from-fast-ai/)

    **Chinese**：[称霸Kaggle的十大深度学习技巧](https://mp.weixin.qq.com/s?__biz=MzU3NjE4NjQ4MA==&mid=2247484049&idx=1&sn=56bb5d502b2ed6e1cbc8b7405b52ad20)

- [训练神经网络的方法分享-Andrej Karpathy - 2019](https://mp.weixin.qq.com/s?__biz=MzIxNDgzNDg3NQ==&mid=2247485845&idx=1&sn=13620bb17dc0fd75d71100dd84cce59d)

- [37 Reasons why your Neural Network is not working - 2017](https://medium.com/@slavivanov/4020854bd607)

    **Chinese**: [训练的神经网络不工作？一文带你跨过这37个坑](https://baijiahao.baidu.com/s?id=1573879911456703)

- [如何训练一个性能不错的深度神经网络 - 2018](http://www.sohu.com/a/222666303_609569)

    **YAO**：OK 归一化推荐一看，其他just soso

    - 零均值化：可避免参数梯度过大

    - 归一化：可使不同维度的数据具有大致相同的分布规模
        
        原因：**S=W1\*x1+W2\*x2+b => dS/dW1=x1,dS/dW2=x2参与形成参数的梯度** => x1与x2本身差异巨大会导致梯度差异巨大，影响梯度下降的效果和速度
        
        进一步讲，之所以会影响，是因为在实际操作中为了方便，**所有维度共用同一套更新策略，即设置相同的步长**，随着迭代进行，步长的缩减也是同步的，这就要求不同维度数据的分布规模大致相同。BTW，理论上讲，**不同维度需要设置不同的迭代方案**，则不受以上影响了。

    - 初始化：使用非常接近零的随机数初始化参数，以打破网络的对称性

    - 模型集成
      - 常见的，省略……
      - 相同模型 + 相同超参 + 不同参数初始化： 缺点是模型多样性仅仅取决于初始化
      - 单个模型的不同Checkpoint

- [Ten Techniques Learned From fast.ai - 2018](https://blog.floydhub.com/ten-techniques-from-fast-ai/)

    **Chinese**: [10大称霸Kaggle的深度学习技巧](https://blog.csdn.net/yH0VLDe8VG8ep9VGe/article/details/81463870)

- 【Great】[半天2k赞火爆推特！李飞飞高徒发布33条神经网络训练秘技 - 2019](https://zhuanlan.zhihu.com/p/63841572)

    **YAO**: HERE HERE HERE HERE HERE HERE

- [笔记之Troubleshooting Deep Neural Networks - 2019](https://zhuanlan.zhihu.com/p/89566632)

- [Practical Advice for Building Deep Neural Networks](https://pcc.cs.byu.edu/2017/10/02/practical-advice-for-building-deep-neural-networks/)

    **Chinese**: [构建神经网络的一些实战经验和建议](https://mp.weixin.qq.com/s/c00PXpJHctdm_YEbk4NTQA)

- 【Great】[如何优雅地训练大型模型 - 2020](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650412541&idx=1&sn=f17415894641f9b555c4ac7f7af4e42a)


## 25.2 Parameter Tuning

#### Article

- 【Great】[别再喊我调参侠！夕小瑶“科学炼丹”手册了解一下 - 2020](https://mp.weixin.qq.com/s/HOgdhAzyti0N2TZBmt_4Iw)

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

- [step-by-step: 夕小瑶版神经网络调参指南 上篇 - 2018](https://mp.weixin.qq.com/s?__biz=MzIwNzc2NTk0NQ==&mid=2247484836&idx=1&sn=bafd0e43515d8f29105decf6481af61e)

    **YAO**: GREAT GREAT GREAT GREAT GREAT GREAT TO BE CONTINUED ……   做NER项目时再阅读

    - 调参前请务必
      - **做好可视化**：可视化每个Step(Batch)的loss和metrics(如准确率或f1-score)，推荐使用tensorboard；建议既输出到屏幕也写入文件，推荐使用logging
      - **关闭正则化**：如L2, Dropout等，它们很可能会极大影响loss曲线，主要在初始调参前

    - learning_rate & num_steps

    - batch_size & momentum

    - learning_rate衰减策略

- [有哪些deep learning（rnn、cnn）调参的经验 - 2019](https://www.zhihu.com/question/41631631)

    **YAO**: HEREHEREHEREHEREHEREHEREHEREHEREHERE

    - 参数初始化：

    - 数据预处理：

    - 训练技巧之梯度：

- [你在训练RNN的时候有哪些特殊的trick？ - 2017](https://www.zhihu.com/question/57828011)

    **YAO**: HEREHEREHEREHEREHEREHEREHEREHEREHERE

- [神经网络训练trick - 2019](https://zhuanlan.zhihu.com/p/59918821)

- [深度学习中训练参数的调节技巧 - 2018](https://www.jianshu.com/p/4eff17d9c4ce)

- [深度学习网络调参技巧 - 2017](https://zhuanlan.zhihu.com/p/24720954)

- [调参是否能对深度学习模型的性能有极大提升？ - 2019](https://www.zhihu.com/question/339287819)

- [是否有办法提高深度学习的调参效率？ - 2019](https://www.zhihu.com/question/339102039)

- [Practical guide to hyperparameters search for deep learning models - 2018](https://blog.floydhub.com/guide-to-hyperparameters-search-for-deep-learning-models/)

    超参数搜索不够高效？这几大策略了解一下

- [Complete Machine Learning Guide to Parameter Tuning in Gradient Boosting (GBM) in Python - 2016](https://www.analyticsvidhya.com/blog/2016/02/complete-guide-parameter-tuning-gradient-boosting-gbm-python/)

## Library

- <http://hyperopt.github.io/>

    hyperopt: 机器学习调参神器

    **Article**: [Optimizing hyperparams with hyperopt](http://fastml.com/optimizing-hyperparams-with-hyperopt/)

    **Article**: [python调参神器hyperopt](https://blog.csdn.net/qq_34139222/article/details/60322995)


## 25.3 Learning Rate

最重要的超参，没有之一。

- [Cyclical Learning Rates for Training Neural Networks - USNavy2017](https://arxiv.org/abs/1506.01186)

    CLR: Cyclic Learning Rate

    **Code**: <https://github.com/bckenstler/CLR>

#### Article

- [Improving the way we work with learning rate - 2017](https://techburst.io/improving-the-way-we-work-with-learning-rate-5e99554f163b)

- [Using Learning Rate Schedules for Deep Learning Models in Python with Keras - 2016](https://machinelearningmastery.com/using-learning-rate-schedules-deep-learning-models-python-keras/)

- [Setting the learning rate of your neural network - 2018](https://www.jeremyjordan.me/nn-learning-rate/)

- [Understanding Learning Rates and How It Improves Performance in Deep Learning - 2018](https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10)


## 25.4 Finetuning

#### Article

- A Comprehensive guide to Fine-tuning Deep Learning Models in Keras -2016
  - [Part1](https://flyyufelix.github.io/2016/10/03/fine-tuning-in-keras-part1.html)
  - [Part2](https://flyyufelix.github.io/2016/10/03/fine-tuning-in-keras-part2.html)


## 25.5 GPU & Hardware

#### Article

- [深度学习的GPU：深度学习中使用GPU的经验和建议 - 2018](https://blog.csdn.net/sinat_36458870/article/details/78946030)

    **YAO**: HERE HERE HERE HERE HERE HERE

- [给训练踩踩油门——Pytorch加速数据读取 - 2019](https://mp.weixin.qq.com/s/jXaItEwH10-reaiH2pKztw)

- [不止显卡！这些硬件因素也影响着你的深度学习模型性能 - 2019](https://zhuanlan.zhihu.com/p/67785062)

#### Practice

- [在深度学习中喂饱gpu - 2019](https://zhuanlan.zhihu.com/p/77633542)

    **Code**: <https://github.com/tanglang96/DataLoaders_DALI> (PyTorch)

- [GPU 显存不足怎么办？- 2019](https://zhuanlan.zhihu.com/p/65002487)

- [Speed Up your Algorithms Part 1 — PyTorch - 2018](https://towardsdatascience.com/speed-up-your-algorithms-part-1-pytorch-56d8a4ae7051)

    **Chinese**: [PyTorch 算法加速指南](https://mp.weixin.qq.com/s?__biz=MzU3NjE4NjQ4MA==&mid=2247486517&idx=3&sn=ed7db64a5b43191786167d83368cae8f)

    **Github**: <https://github.com/PuneetGrov3r/MediumPosts/tree/master/SpeedUpYourAlgorithms>


## 25.6 Pseudo Labeling

训练集训练好的模型应用于测试集后的结果(即伪标签)，与训练集混合在一起后重新训练模型。？

一般模型能忍受10%的噪音，所以不要把所有测试集与训练集混合，建议保持比例在10:1左右。？

#### Article

- [A Simple Pseudo-Labeling Function Implementation in Keras - 2017](https://shaoanlu.wordpress.com/2017/04/10/a-simple-pseudo-labeling-function-implementation-in-keras/)

- [标签传播算法（Label Propagation）及Python实现 - 2015](https://blog.csdn.net/zouxy09/article/details/49105265)


## 25.x Others

#### Article

- [标签平滑&深度学习：Google Brain解释了为什么标签平滑有用以及什么时候使用它(SOTA tips)​ - 2019](https://mp.weixin.qq.com/s?__biz=MzI4MDYzNzg4Mw==&mid=2247490838&idx=4&sn=a8900b30647184fe37fbdbde64fb3098)

