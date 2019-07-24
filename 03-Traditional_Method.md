# 3. Traditional Method

## 3.1 Overview


#### Practice

- 【Great!】<https://github.com/jcsyl/news-analyst> (Keras)

    对舆情事件进行词云展示，对评论进行情感分析和观点抽取。情感分析基于lstm 的三分类，观点抽取基于AP 算法的聚类和MMR的抽取

    **YAO**: 使用 TFIDF 和 TextRank 提取关键词，使用 Word2Vec 和 LSTM 进行情感三分类，通过 AP 聚类进行观点聚类和抽取！


## 3.2 TFIDF

本质上是一种 BoW 词袋模型

处理对象：整个 corpus 里某个 document 中的各个 term

返回结果：tfidf(document, term)，即某文档 document 中某字/词 term 对应的 TFIDF 值！基于此，可输出 corpus 中每个 document 对应的所有有效 term 及其 tfidf 值

#### Practice

- <https://github.com/speciallurain/CNKI_Patent_SVM>

    TFIDF 提取语义词典，SVM 进行文本分类

    **YAO**: TFIDF 提取语义词典的方式可以借鉴，用于提取关键词、构建标签体系、计算相似度等。

#### Library

- [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

    - 语料 corpus: Iterable of document， document是**分词后以空格分隔的字符串**
    - CountVectorizer: 将 corpus 中的词语转换为词频矩阵，即词袋模型 BoW 对应的矩阵表示
    - TfidfTransformer: 使用 CountVectorizer 的结果，生成每个词语的 tfidf 权值
    - TfidfVectorizer: 等同于 CountVectorizer + TfidfTransformer，直接基于 corpus 生成 tfidf 值

    Reference: <https://www.jianshu.com/p/c7e2771eccaa>

- [gensim](https://radimrehurek.com/gensim/models/tfidfmodel.html)

    - 初级语料 texts: Iterable of document， document是**分词列表**
    - 正式语料 corpus: 字典(初级语料-->字典) + 初级语料 --> 正式语料
    - corpora.Dictionary: 基于 texts 生成字典
    - models.TfidfModel: 基于 corpus 训练模型

    Reference: <https://blog.csdn.net/sinat_26917383/article/details/71436563>

- [jieba](https://github.com/fxsjy/jieba)

    - jieba.analyse.extract_tags：关键词提取，不需要提供 corpus，只需要一个 **document(一个字符串)** 即可！此时 jieba 会使用自带的 corpus 计算 idf 值
    - 参数 sentence: 待提取关键词的文本
    - 参数 topK: 返回TFIDF权重最大的前K个关键词，默认为20
    - 参数 withWeight: 是否一并返回关键词的权重值，默认为False
    - 参数 allowPOS: 仅返回指定词性的词，默认为空，即不筛选


## 3.3 TextRank

借鉴 PageRank 算法理念

处理对象：某个 document 中各个 term，没有 corpus 这一概念？

返回结果：

#### Library

- [jieba](https://github.com/fxsjy/jieba)

    - jieba.analyse.textrank: 关键词提取
    - 参数 sentence/topK/withWeight/allowPOS: 同上面的 TFIDF

#### Practice

- <https://github.com/liuhuanyong/KeyInfoExtraction>

    基于Textrank算法的文本摘要抽取与关键词抽取，基于TFIDF算法的关键词抽取


## 3.4 AP Clustering



## 3.10 Application

### Keyword Extraction

Methods: TFIDF, TextRank, Topic-Model(如LDA), RAKE