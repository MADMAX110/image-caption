# !/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
from data_helper import get_word_index, get_img_feature, data_generator, get_InceptionV3_image_feature, plot_image, create_sequences
from tensorflow.keras.layers import Input, Dense, Reshape, Embedding, LSTM, Concatenate, TimeDistributed, GlobalAveragePooling1D, RepeatVector, Dropout, Add
from tensorflow.keras.utils import plot_model
import os
os.environ['PATH'] += os.pathsep + 'D:/Graphviz2.38/bin'
import warnings
warnings.filterwarnings('ignore')

def MyModel(max_len, vocab_size, learing_rate=1e-3):
    img_input = Input(shape=(2048,), dtype='float32', name='img_ipt')
    img_vector = Dropout(0.5)(img_input)
    img_vector = Dense(256, activation='relu')(img_vector)

    # sequence model
    caption_input = Input(shape=(max_len,), dtype='int32', name='caption_ipt')
    caption_embedding = Embedding(vocab_size, 256, mask_zero=True)(caption_input)
    caption_embedding = Dropout(0.5)(caption_embedding)
    caption_embedding = LSTM(256)(caption_embedding)

    # decoder model
    decoder = Add()([img_vector, caption_embedding])
    decoder = Dense(256, activation='relu')(decoder)
    outputs = Dense(vocab_size, activation='softmax', name='out')(decoder)

    model = tf.keras.Model(inputs=[img_input, caption_input], outputs=outputs)
    opt = tf.keras.optimizers.Adam(lr=learing_rate, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(optimizer=opt,
                  loss='categorical_crossentropy',#tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])
    model.summary()
    plot_model(model, show_shapes=True, to_file='model.png')
    return model

def predict(model, max_len, word_index, index_word, img_feature):
    start = word_index['<start>']
    cap_vector = [start]
    while True:
        vector = tf.keras.preprocessing.sequence.pad_sequences([cap_vector], maxlen=max_len, padding='post') # [1, seq_len]
        logits = model.predict([np.array(img_feature), np.array(vector)])
        pred = np.argmax(logits, axis=1)[0]
        if index_word[pred] == '<end>' or len(cap_vector) >= max_len:
            break
        cap_vector.append(pred)
    text = ' '.join([index_word[index] for index in cap_vector[1:]])
    return text

def beam_search(model, max_len, word_index, index_word, img_feature, beam_index=5):
    start = [word_index['<start>']]
    start_word = [[start, 0.0]]
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            vector = tf.keras.preprocessing.sequence.pad_sequences([s[0]], maxlen=max_len,
                                                                   padding='post')  # [1, seq_len]
            logits = model.predict([np.array(img_feature), np.array(vector)])
            word_preds = np.argsort(logits[0])[-beam_index:]
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += logits[0][w]
                temp.append([next_cap, prob])
        start_word = temp
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        start_word = start_word[-beam_index:]
    start_word = start_word[-1][0]
    intermediate_caption = [index_word[i] for i in start_word]
    final_caption = []
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break
    final_caption = ' '.join(final_caption[1:])
    return final_caption

if __name__ == '__main__':
    train = pd.read_csv('../data/train.csv')
    cap_vector, tokenizer, index_word, max_len= get_word_index(train['captions'])
    vocab_size = len(index_word) + 1

    img_feature = get_img_feature(train['img_path'])
    img, cap, label = create_sequences(img_feature, cap_vector, max_len, vocab_size)

    model = MyModel(max_len=max_len,
                    vocab_size=vocab_size)
    model.fit([img, cap], label, epochs=20, batch_size=64)

    model.save('models_checkpoints/model_sample_20000.h5')

    # model = tf.keras.models.load_model('models_checkpoints/model_sample_20000.h5')
    #
    # path = '../train2014/COCO_train2014_000000190271.jpg'
    # img_feature = get_InceptionV3_image_feature(path)
    # beam_search_3 = beam_search(model, max_len, tokenizer.word_index, index_word, img_feature, beam_index=3)
    # beam_search_5 = beam_search(model, max_len, tokenizer.word_index, index_word, img_feature, beam_index=5)
    # beam_search_7 = beam_search(model, max_len, tokenizer.word_index, index_word, img_feature, beam_index=7)
    # print('Beam Search, k=3:', beam_search_3)
    # print('Beam Search, k=5:', beam_search_5)
    # print('Beam Search, k=7:', beam_search_7)
    # plot_image(path, beam_search_5)
