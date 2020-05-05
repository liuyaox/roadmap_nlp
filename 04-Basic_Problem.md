# 4. Basic Problem

## 4.1 Overview


## 4.2 Segmentation

#### Library

- 【Great】<https://github.com/lancopku/PKUSeg-python>

    中文分词工具包，准确度远超 Jieba

#### Paper

- [A Simple and Effective Neural Model for Joint Word Segmentation and POS Tagging - HLJU2018]()

    **Code**: <https://github.com/bamtercelboo/pytorch_Joint-Word-Segmentation-and-POS-Tagging> (PyTorch)

    **Code**: <https://github.com/zhangmeishan/NNTranJSTagger> (C++)

- [Effective Neural Solution for Multi-Criteria Word Segmentation - Canada2018](https://arxiv.org/abs/1712.02856)

    **Code**: <https://github.com/hankcs/multi-criteria-cws> (Dynet)

#### Practice

- <https://github.com/FanhuaandLuomu/BiLstm_CNN_CRF_CWS> (Tensorflow & Keras)

    BiLstm+CNN+CRF 法律文档（合同类案件）领域分词   [中文解读](https://www.jianshu.com/p/373ce87e6f32)

- <https://github.com/liweimin1996/CwsPosNerCNNRNNLSTM>

    基于字向量的CNN池化双向BiLSTM与CRF模型的网络，可能一体化的完成中文和英文分词，词性标注，实体识别。主要包括原始文本数据，数据转换,训练脚本,预训练模型,可用于序列标注研究.

#### Article

- [深度学习时代，分词真的有必要吗 - 2019](https://zhuanlan.zhihu.com/p/66155616)

- [从why到how的中文分词详解，从算法原理到开源工具 - 2019](https://mp.weixin.qq.com/s?__biz=MzI3ODgwODA2MA==&mid=2247488211&idx=2&sn=3adb96ff316dfc67663f72a9595eda5d)


## 4.3 Dependency Parsing

#### Practice

- <https://github.com/Samurais/text-dependency-parser>

    依存关系分析


## 4.4 Vocabulary & OOV

Character + Subword

#### Article

- [非主流自然语言处理：大规模语料词库自动生成 - 2017](http://www.sohu.com/a/157426068_609569)

- [word2vec缺少单词怎么办？](https://www.zhihu.com/question/329708785)


## 4.5 Subword

### 4.5.1 Overview

Subword算法有以下2大类：

- 基于统计学: BPE(具体实现有subword-nmt、SentencePiece等), WordPiece(Google未开源), ULM(具体实现有SentencePiece等)

- LMVR：考虑了词的形态特征（适合于德语、土耳其语、意大利语等）(Linguistically-motivated Vocabulary Reduction)

#### Article

- [子词技巧：The Tricks of Subword - 2019](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650411766&idx=3&sn=c5f92645737b469d386bf303bcbcf71f)

    **YAO**: OK

    增量法: BPE, WordPiece   减量法: ULM (Unigram Language Model)

    - BPE: 省略，详见4.5.2

    - WordPiece: 类似BPE，只是在算法步骤第2步时，BPE选择当前语料下**频次最大**的符号对，而WordPiece选择当前语料下**语言模型似然值提升最大**的符号对，即先将语料按当前词表分解，以训练语言模型，对整个语料计算一个似然值，之后在当前词表基础上，分别组合各个符号对，重新训练语言模型并计算似然值，取似然值提升最大的那个符号对，将其加入词表。听起来计算量较大，论文里有一些降低计算量的策略。

    - ULM: 与WordPiece都是用语言模型来挑选符号对，不同点是WordPiece是**初始化小词表+不断添加符号对**，ULM是**初始化大词表+不断剔除符号对(通过语言模型评估)**！每次迭代时，按**符号对的loss**排序并只保留TopK%

    - Subword Regularization: 省略，示例请参考<https://github.com/google/sentencepiece#subword-regularization>

    - BPE Dropout: 在BPE基础上，加入一定的随机性。在每次对训练数据处理分词时，**设定一定的概率(如10%)让一些合并不通过**，则相同的词每次会分出不同的子词。通过不同的分词方案，模型获得了对整个词更好更全面的理解！虽然简单，但在实际翻译任务中提升很大。

- [3 subword algorithms help to improve your NLP model performance - 2019](https://medium.com/@makcedward/how-subword-helps-on-your-nlp-model-83dd1b836f46)

    **YAO**: OK
    
    Algorithms: BPE, WordPiece, ULM   Implementation: SentencePiece

    For example, we can use **two vector (i.e. “sub” and “word”) to represent “subword”**

    Subword balances vocabulary size and footprint. **16k or 32k** subwords are recommended vocabulary size to have a good result.

    Many Asian language word can't be separated by space. Therefore, the initial vocabulary is larger than English. You may need to prepare **over 10k initial word** to kick start. In WordPiece paper, they propose to use **22k word and 11k** word for Japanese and Korean respectively.

    **TODO**: Chinese ?

- [深入理解NLP Subword算法：BPE、WordPiece、ULM - 2020](https://zhuanlan.zhihu.com/p/86965595)


### 4.5.2 Byte Pair Encoding (BPE)

Subword Units / BPE: [Neural Machine Translation of Rare Words with Subword Units - Edinburgh2016](https://arxiv.org/abs/1508.07909)

把 Rare Words 和 Unknown Words 用 Subword Units 序列来编码，更简单有效。BERT中使用了，有对应的tokenization.py，GPT-2也用了改进版的BPE

#### Article

- 【Great】[Byte Pair Encoding - 2019](https://leimao.github.io/blog/Byte-Pair-Encoding/)

    **YAO**: 有详细的 Token Learning Example 和 Encoding and Decoding Example 对应的代码

- 【Great】[subword-units - 2017](https://plmsmile.github.io/2017/10/19/subword-units/)

    **YAO**: OK

    翻译系统处理的是Open-Vocabulary场景，很多语言具有创造力，比如凝聚组合等，翻译系统需要一种**低于Word-Level**的机制！

    Subword神经网络模型可以从Subword表达中**学习到组合和直译**等语言能力，也可有效产生不在训练数据集中的词汇。

    BPE: 一种简单的数据压缩技术，它能**找出句子中经常出现的Byte Pair**，并用一个没有出现过的字符去替代，同时生成一个**紧凑的固定大小的subword词典**(能够更紧凑地表示较短序列，即下文的符号词表？？)
    
    其算法步骤如下(详见文中代码)：

    - 初始化符号词表：把语料**分解为最小单元(字符，如26个字母+其他字符，以空格分隔)**加入符号词表（后续加入的符号有字符对、符号对等），特殊处理单词末尾。示例：'dog' --> ' d o g -'
    
    - 不断迭代：统计当前所有**相邻符号对**的频次，找到频次最大的符号对(A B)，合并符号词表中所有(A B)为AB，并为其产生一个新的符号(这样常见的ngram会被合并为一个符号加入词表中)。示例：'Y A B C' -> 'Y AB C'
    
    - 迭代中止：重复以上迭代，直至词表大小达到指定大小。符号词表大小 = 初始词表大小 + 合并操作次数(算法超参数，可设置)

#### Code

- <https://github.com/bheinzerling/bpemb>

    A collection of pre-trained subword embeddings in 275 languages, based on BPE and trained on Wikipedia. It can be as input for neural models in NLP.

#### Library

- subword-nmt: <https://github.com/rsennrich/subword-nmt>


### 4.5.3 WordPiece

谷歌内部包，未开源

- [Japanese and Korean voice search - Google2012](https://ieeexplore.ieee.org/abstract/document/6289079)

- [Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation - Google2016](https://arxiv.org/abs/1609.08144)

#### Article

- [一文读懂BERT中的WordPiece - 2019](https://www.cnblogs.com/huangyc/p/10223075.html)


### 4.5.4 Unigram Language Model (ULM)

ULM: [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates - Google2018](https://arxiv.org/abs/1804.10959)


### 4.5.5 SentencePiece

SentencePiece: [A simple and language independent subword tokenizer and detokenizer for Neural Text Processing - Google2018](https://arxiv.org/abs/1808.06226)

面向神经网络文本生成系统的无监督文本Tokenization开源工具

YAO: BPE关注的是**character级别的组合**，而SentencePiece除此之外也关注**word/phrase级别的组合**

YAO: WordPiece VS SentencePiece，都统计频次，前者侧重**字符组合(subword) within word**，后者侧重**词语组合(subsentence) within sentence**，当然后者也有前者的功能

#### Article

- [Subword BPE 理解 - 2019](https://xinghanzzy.github.io/2019/03/08/Subword%20BPE%20%E7%90%86%E8%A7%A3&sentence%20piece/#Word-piece)

    **YAO**: OK

    SentencePiece能够从大量无监督语料中自动学习出经常出现的Phrase，这个Phrase形式不限、长度不限，可以是短词、短语、短句，甚至是长句！不像Jieba分词那样，切分出来的只是严格符合词法的词语，一般不会有短句、长句。

    - 本质作用：**从大量序列中自动学习出经常出现的子序列片断**

    - 模型训练和应用：支持命令行和Python调用，训练后生成一个Model和一个词典(明文，可编辑)。若分析某个领域相关问题，可用该领域的书籍和文档去训练模型，并不只限于被分析的内容本身，训练数据越多，模型效果越好。

#### Github

- 【Great】<https://github.com/google/sentencepiece>

    SentencePiece is an unsupervised text tokenizer and implements BPE and ULM with the extension of direct training from raw sentences.

    **YAO**: OK

    - 支持功能：**BPE和ULM + 字符级和词级别 + Language Independent + Subword Regularization + Text Normalization**

    - Unlike most unsupervised word segmentation algorithms, which assume an infinite vocabulary, SentencePiece's **final vocabulary size is fixed**, e.g., 8k, 16k, or 32k

    - **Trains from raw sentences**: Previous sub-word implementations assume that the input sentences are pre-tokenized. SentencePiece is fast enough to **train from raw sentences**. It's useful for Chinese and Japanese where no explicit spaces exist between words

    - **Whitespace is treated as a basic symbol**: SentencePiece把whitespace当作一般symbol对待(而非分隔符，处理时会先escape whitespace with '_')，This feature makes it possible to perform detokenization without relying on language-specific resources. 
    
    - **启发1**：处理一些语言时，若空格只是分隔符，为避免当成一般symbol来处理，可以先**手动去除所有空格**

    - 模型训练
        - input: 支持多文件输入，每个文件是one-sentence-per-line raw corpus file
        - input_format: txt支持raw corpus file，tsv支持<token, frequency>这种带有频次的训练数据(讲解BPE算法步骤时就是这种)
        - character_coverage: 模型cover的character数量，for language with rich character如中文和日文，建议取0.9995，for other with small character set，取1.0
        - model_type: 支持unigram(default), bpe, char, word，**当取word时，input sentence must be pretokenized**. **关键: 默认使用数字、空格来分隔**
        - redefine special meta tokens: UNK,BOS,EOS默认id是0,1,2，可设置，--bos_id=0 --eos_id=1 --unk_id=5 --pad_id=3，当取值-1时表示disabled
        - **split_by_whitespace:** 非常关键！！表示token是否按whitespace来分隔的！只有取值false时才会提取**crossing-tokens pieces**，即短语(token组合)
    
    - model_type: 三种粒度，char, word(其实是token), subword(其实是subtoken，算法有bpe和unigram)
        - char: vocab是语料中出现的所有字符(含空格)
        - word: 准确来说是token，vocab是语料中出现的所有token(语料需pretokenized，即用空格分隔好各个token，否则一长串无空格的字符串会被当成token，比如中文里一整句话！)
        - bpe:  vocab是从语料中提取出的subtoken
        - unigram: vocab同上，算法不同
        - YAO: 当语料没有pretokenized或者split_by_whitespace=false时，提取的subtoken其实是phrase/subsentence！尤其对于中文
        - YAO: 其实可这样通俗理解，char就是最小字符，word就是空格分隔的各个token(其实是word+sentence+paragraph，取决于是否用有空格分隔)，而subtoken就是在token内(其实是word内+sentence内+paragraph内)提取片段Piece，token=word时就是正儿八经的常见subword(字符组合)，token=sentence时就是常见phrase(单词组合)，token=paragraph时就是句常见句子组合！
        - YAO: 只有char不受是否有whitespace、是否pretokenzation影响，token受影响，因此subtoken也受影响。**关键：subtoken时，默认whitespace既是分隔token的分隔符，又是一般字符。若只当一般字符，令split_by_whitespace=false即可！**

    - 模型Encode(Decode类似)
        - output_format: 支持piece和id，支持nbest segmentation and segmentation sampling **TODO**: 干嘛的？
        - extra_options: 支持添加BOS/EOS以及reverse the input sentence **TODO**: 干嘛的？
        - generate_vocabulary: 场景为同时训练2种语言的模型后，使用该模型为其中一种语言的训练数据生成对应的vocabulary
        - vocabulary & vocabulary_threshold: Only produce symbols which also appear in the vocabulary (with at least some frequency)

    - 【Great】[Python Wraper详细使用示例 - Jupyter Notebook](https://gist.github.com/liuyaox/dd211cc9274e9a1d7201b459d11e3bb5)
        - SentencePieceProcessor既可以当编码器、解码器，也可以当字典使用
        - 可以自定义User Defined Symbols和Control Symbols
        - ULM对应的模型支持Sampling和NBest Segmentation，可用于数据增强
        - **处理英文等时建议使用Text Normalization**
        - 可使用限制Vocabulary，只编码解码Vocabulary中的Token
        - **通过split_by_whitespace=false，以提取Phrase，而非subtoken**

    - 示例
        - 从英文语料中提取常见Phrase
            ```
            spm_train --input corpus.txt \
                --model_prefix=m \
                --vocab_size=24000 \
                --model_type=unigram \
                --normalization_rule_name=nfkc_cf \ # 可取其他值？最好取这个值？
                --character_coverage=1.0 \
                --split_by_whitespace=false         # 关键所在  因为英文有空格，天然有Pretokenization
            ```

        - 从中文语料中提取常见Phrase，注意与示例1的区别，省略选项同示例1
            ```
            spm_train --input corpus.txt \
                --... \
                --character_coverage=0.9995 \       # 默认值，可省略
                --split_by_whitespace=True          # 默认值，可省略  因为中文无空格，无Pretokenization
            ```

- <https://github.com/yoheikikuta/bert-japanese>

    BERT with SentencePiece for Japanese text

- <https://github.com/wannaphong/thai-word-segmentation-sentencepiece>

    thai word segmentation using SentencePiece

#### Library

- [sentencepiece](https://pypi.org/project/sentencepiece/)

    编码(EncodeAsPieces/EncodeAsIds/NBestEncodeAsPieces)、解码(DecodePieces/DecodeIds)和转化(IdToPiece/PieceToId)等

#### FAQ

- [How to pickle a SentencePiece model #387](https://github.com/google/sentencepiece/issues/387)

    Wrap model in a class with methods \_\_getstate\_\_ and \_\_setstate\_\_   [Ref Code](https://github.com/asyml/texar-pytorch/blob/master/texar/torch/data/tokenizers/xlnet_tokenizer.py#L118)


## 4.6 Phrase Mining & Keyword Extraction

#### Article

- 【Great】[谈谈医疗健康领域的Phrase Mining - 2020](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650412407&idx=2&sn=2e96b4456afb1a208aab475ebf9fa1b8)

- 【Great】医疗健康领域的短文本解析探索 - 2020: [一](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650412504&idx=2&sn=09f7f5783a48c009744997714b52eec4) and [二](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650412504&idx=3&sn=37f03898a549e89e1da0a093b67e9925)

- 【Great】[知识图谱如何应用到文本标签化算法中 - 2020](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650412450&idx=2&sn=3c01ae8525dbab5202433b19fc0cc5db)

#### Practice

- <https://github.com/bigzhao/Keyword_Extraction>

    神策杯2018高校算法大师赛（中文关键词提取）第二名代码方案


## 4.7 New Word

#### Practice

- <https://github.com/xylander23/New-Word-Detection>

    新词发现算法(NewWordDetection)

- <https://github.com/zhanzecheng/Chinese_segment_augment>

    python3实现互信息和左右熵的新词发现
    
- <https://github.com/Rayarrow/New-Word-Discovery>

    新词发现 基于词频、凝聚系数和左右邻接信息熵


## 4.9 Disambiguation

词语纠错和语义消歧

**YAO**: 特别重要！几乎可以作为所有 NLP 任务最最开始时的处理！比分词还要靠前！

#### Practice

- 【Great】<https://github.com/taozhijiang/chinese_correct_wsd>

    简易的中文纠错和消歧

- <https://github.com/beyondacm/Autochecker4Chinese>

    中文文本错别字检测以及自动纠错 / Autochecker & autocorrecter for chinese

- <https://github.com/liuhuanyong/WordMultiSenseDisambiguation>

    基于百科知识库的中文词语多词义/义项获取与特定句子词语语义消歧

- <https://github.com/ccheng16/correction>

    Chinese "spelling" error correction

- <https://github.com/apanly/proofreadv1>

    中文文本自动纠错

#### Article

- [中文文本纠错算法--错别字纠正的二三事 - 2019](https://mp.weixin.qq.com/s?__biz=MjM5ODkzMzMwMQ==&mid=2650411874&idx=3&sn=f78fa6e6ba3493086503cbb7d11a7ff2)

    **YAO**: 对当前一些Library/Tool的概述和测试

    主要技术：错别字词典，编辑距离，语言模型(NGram, DNN)

    三个关键点：分词质量，领域相关词典质量，语言模型的种类和质量

    语言模型：计算序列的合理性；
    
    规则约束：优先选择拼音相同的候选字词，其次是形似的；纠正后分词个数更少；


## 4.10 Synonym

#### Article

- [如何扩充知识图谱中的同义词 - 2019](https://mp.weixin.qq.com/s?__biz=MzUyMDY0OTg3Nw==&mid=2247484011&idx=1&sn=8f6e6ae9e3d34b7a1dc2cd55812f55ca)

    同义词挖掘，以医学术语为例

