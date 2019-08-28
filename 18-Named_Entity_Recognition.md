

# 18. Named Entity Recognition (NER)

## 18.1 Overview

#### Article

- [浅析深度学习在实体识别和关系抽取中的应用 - 2017](https://blog.csdn.net/u013709270/article/details/78944538)

- 【Great】[Named entity recognition serie - 2018](https://www.depends-on-the-definition.com/introduction-named-entity-recognition-python/) (Keras)

    包括NER各种模型：CRF, Seq2Seq, LSTM+CRF, LSTM + Char Embedding, Residual LSTM + ELMo, NER with Bert, 


## 18.2 HMM & CRF

## 18.3 RNN

## 18.4 RNN + CRF

主要以 BiLSTM + CRF 为主

#### Paper

- [Neural Architectures for Named Entity Recognition - CMU2016](https://arxiv.org/abs/1603.01360)

    BiLSTM + CRF

- HSCRF:  [Hybrid semi-Markov CRF for Neural Sequence Labeling - USTC2018](https://arxiv.org/abs/1805.03838)

    **Github**: <https://github.com/ZhixiuYe/HSCRF-pytorch> (PyTorch)


#### Code

- [BiLTSM + CRF](http://www.voidcn.com/article/p-pykfinyn-bro.html) (Keras)

    ```python
    from keras.layers import Sequential, Embedding, LSTM, bidirectional, Dropout, TimeDistributed, Dense
    from keras_contrib.layers.crf import CRF
    from keras_contrib.utils import save_load_utils

    HIDDEN_UNITS = 200
    NUM_CLASS = 5
 
    model = Sequential()
    model.add(Embedding(2500, output_dim=128, input_length=100))
    model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(LSTM(HIDDEN_UNITS, return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(TimeDistributed(Dense(NUM_CLASS)))
    crf_layer = CRF(NUM_CLASS)
    model.add(crf_layer)

    model.compile('rmsprop', loss=crf_layer.loss_function, metrics=[crf_layer.accuracy])

    model_path = 'xxx'
    save_load_utils.save_all_weights(model, model_path)     # 保存模型
    save_load_utils.load_all_weights(model, model_path)     # 加载模型
    ```

- [BiLSTM + CNN + CRF](https://blog.csdn.net/xinfeng2005/article/details/78485748) (Keras)

    ```python
    from keras.callbacks import ModelCheckpoint, Callback
    from keras.layers import Input, Embedding, Bidirectional, LSTM, Dropout, ZeroPadding1D, Conv1D, TimeDistributed, Dense
    from keras.models import Model
    from keras_contrib.layers import CRF
    from visual_callbacks import AccLossPlotter

    class LossHistory(Callback):
        def on_train_begin(self, logs={}):
            self.losses = []

        def on_batch_end(self, batch, logs={}):
            self.losses.append(logs.get('loss'))

    # Input -> Embedding
    word_input = Input(shape=(max_len,), dtype='int32', name='word_input')
    word_emb = Embedding(len(char_value_dict)+2, output_dim=64, input_length=max_len, dropout=0.2, name='word_emb')(word_input)

    # Embedding -> BiLSTM -> Dropout => X1
    bilstm = Bidirectional(LSTM(32, dropout_W=0.1, dropout_U=0.1, return_sequences=True))(word_emb)
    bilstm_d = Dropout(0.1)(bilstm)

    # Embedding -> ZeroPadding1D -> Conv1D -> Dropout -> TimeDistributed(Dense) => X2
    half_window_size = 2
    paddinglayer = ZeroPadding1D(padding=half_window_size)(word_emb)
    conv = Conv1D(nb_filter=50, filter_length=(2 * half_window_size + 1), border_mode='valid')(paddinglayer)
    conv_d = Dropout(0.1)(conv)
    dense_conv = TimeDistributed(Dense(50))(conv_d)

    # X1 + X2 -> merge -> TimeDistributed(Dense) -> CRF
    rnn_cnn_merge = merge([bilstm_d, dense_conv], mode='concat', concat_axis=2)     # merge??? concatenate?
    dense = TimeDistributed(Dense(class_label_count))(rnn_cnn_merge)
    crf = CRF(class_label_count, sparse_target=False)
    crf_output = crf(dense)

    # Build & Compile
    model = Model(input=[word_input], output=[crf_output])
    model.compile(loss=crf.loss_function, optimizer='adam', metrics=[crf.accuracy])
    model.summary()

    # Train
    checkpointer = ModelCheckpoint(filepath="bilstm_1102_k205_tf130.w", verbose=0, save_best_only=True, save_weights_only=True)
    losshistory = LossHistory()
    plotter = AccLossPlotter(graphs=['acc', 'loss'], save_graph=True, save_graph_path=sys.path[0])
    history = model.fit(x_train, y_train, batch_size=32, epochs=500, callbacks=[checkpointer, losshistory, plotter], verbose=1, validation_split=0.1)
    ```


#### Practice

- <https://github.com/stephen-v/zh-NER-keras> (Keras)

    中文解读：[基于keras的BiLstm与CRF实现命名实体标注 - 2018](https://www.cnblogs.com/vipyoumay/p/ner-chinese-keras.html)

- <https://github.com/fangwater/Medical-named-entity-recognition-for-ccks2017> (PyTorch)

    A LSTM+CRF model for the seq2seq task for Medical named entity recognition in ccks2017

    **YAO**: PyTorch实现的CRF

- <https://github.com/yanwii/ChinsesNER-pytorch> (PyTorch)

    基于BI-LSTM+CRF的中文命名实体识别

- <https://github.com/phychaos/transformer_crf> (Tensorflow)

    Transformer + CRF

- <https://github.com/Determined22/zh-NER-TF> (Tensorflow)

    A very simple BiLSTM-CRF model for Chinese Named Entity Recognition 中文命名实体识别  [中文解读](https://blog.csdn.net/liangjiubujiu/article/details/79674606)
    
- <https://github.com/shiyybua/NER> (Tensorflow)

    BiRNN + CRF

    **Article**: [基于深度学习的命名实体识别详解](https://zhuanlan.zhihu.com/p/29412214)

- <https://github.com/pumpkinduo/KnowledgeGraph_NER> (Tensorflow)

    中文医学知识图谱命名实体识别，模型有：BiLSTM+CRF, Transformer+CRF

- 【Great】<https://github.com/baiyyang/medical-entity-recognition> (Tensorflow)

    包含传统的基于统计模型(CRF)和基于深度学习(Embedding-Bi-LSTM-CRF)下的医疗数据命名实体识别

- <https://github.com/Nrgeup/chinese_semantic_role_labeling> (Tensorflow)

    基于 Bi-LSTM 和 CRF 的中文语义角色标注

- <https://github.com/stephen-v/zh-NER-keras> (Keras)

    中文解读：[基于keras的BiLstm与CRF实现命名实体标注 - 2018](https://www.cnblogs.com/vipyoumay/p/ner-chinese-keras.html)

- <https://github.com/yanwii/ChinsesNER-pytorch> (PyTorch)

    基于BI-LSTM+CRF的中文命名实体识别 Pytorch

- <https://github.com/dkarunakaran/entity_recoginition_deep_learning> (Tensorflow)

    **Article**: [Entity extraction using Deep Learning based on Guillaume Genthial work on NER - 2018](https://medium.com/intro-to-artificial-intelligence/entity-extraction-using-deep-learning-8014acac6bb8)

- [Pytorch BiLSTM + CRF做NER - 2019](https://zhuanlan.zhihu.com/p/59845590)


#### Article

- [CRF Layer on the Top of BiLSTM 1-8 - 2017](https://github.com/createmomo/CRF-Layer-on-the-Top-of-BiLSTM) (Chainer)

- [bi-LSTM + CRF with character embeddings for NER and POS - 2017](https://guillaumegenthial.github.io/sequence-tagging-with-tensorflow.html)
    
    **Github**: <https://github.com/guillaumegenthial/tf_ner> (Tensorflow)

    **中文解读**: [命名实体识别（biLSTM+crf）](https://blog.csdn.net/xxzhix/article/details/81514040)


## 18.5 CNN

- <https://github.com/nlpdz/Medical-Named-Entity-Rec-Based-on-Dilated-CNN>

    基于膨胀卷积神经网络（Dilated Convolutions）训练好的医疗命名实体识别工具