


## 10. Seq2Seq & Encoder-Decoder
### 10.1 Overview

Attention, Transformer, Pointer Network

### 10.2 Seq2Seq & Encoder2Decoder


### 10.3 Attention

#### Paper

Neural Machine Translation by Jointly Learning to Align and Translate-2014 (https://arxiv.org/abs/1409.0473v2)

#### Source

https://github.com/brightmart/text_classification (Tensorflow)

#### Article

a. Visualizing A Neural Machine Translation Model (Mechanics of Seq2seq Models With Attention) (https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

### 10.4 Hierarchical Attention Network

#### Paper

Hierarchical Attention Networks for Document Classification (https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf)

Structure: Word Encoder(BiGRU) -> Word Attention -> Sentence Encoder(BiGRU) -> Sentence Attention -> Softmax

共有Word和Sentence这2种level的Encoder + Attention

Encoder: To get rich representation of word/sentence

Attention: To get important word/sentence among words/sentences

![hierarchical_attention_network_structure](./image/hierarchical_attention_network01.png)

**巧妙之处**：受Attention启发，这种结构不仅可以获得每个Sentence中哪些words较为重要，而且可获得每个document(很多sentence)中哪些sentences较为重要！It enables the model to capture important information in different levels.

#### Source

https://github.com/brightmart/text_classification (Tensorflow)

### 10.5 Transformer - TOTODO

#### Paper

Attention Is All You Need - Google2017 (https://arxiv.org/abs/1706.03762)

#### Source

a. https://github.com/brightmart/text_classification (Tensorflow)

b. The Annotated Transformer (http://nlp.seas.harvard.edu/2018/04/03/attention.html) (PyTorch)

#### Article

a. The Illustrated Transformer (https://jalammar.github.io/illustrated-transformer/)

### 10.6 Summary