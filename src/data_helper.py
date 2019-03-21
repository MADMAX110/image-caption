# !/usr/bin/env python
# -*- coding:utf-8 -*-

import pandas as pd
import json
import numpy as np
from tqdm import tqdm
from sklearn.utils import shuffle
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

def get_samples(num_examples=3000, seed=2019):
    annotation_file = '../annotations/captions_train2014.json'
    PATH = '../train2014/'
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    all_captions = []
    all_img_name_vector = []

    for annot in tqdm(annotations['annotations'], ncols=100):
        caption = '<start> ' + annot['caption'] + ' <end>'
        image_id = annot['image_id']
        full_coco_image_path = PATH + 'COCO_train2014_' + '%012d.jpg' % (image_id)

        all_img_name_vector.append(full_coco_image_path)
        all_captions.append(caption)

    train_captions, img_name_vector = shuffle(all_captions,
                                              all_img_name_vector,
                                              random_state=seed)
    train_captions = train_captions[: num_examples]
    img_name_vector = img_name_vector[: num_examples]

    data = pd.DataFrame({'captions': train_captions, 'img_path': img_name_vector})
    data.to_csv('../data/data.csv', index=False)

def get_train(train_size=3000):
    data = pd.read_csv('../data/data.csv')
    train = data[:train_size]
    train.to_csv('../data/train.csv', index=False)

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    x /= 255
    x = np.expand_dims(x, axis=0)
    return x

def get_InceptionV3_image_feature(image_path):
    img = preprocess(image_path)
    image_model = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
    feature = image_model.predict(img) # [1, 2048]
    return feature

def save_InceptionV3_feature(image_paths):
    for path in tqdm(image_paths, ncols=100):
        feature = get_InceptionV3_image_feature(path)[0]
        pos = path.rfind('/')
        path_of_feature = '../feature/' + path[pos + 1:]
        np.save(path_of_feature, feature)

def get_img_feature(img_paths):
    print('load images feature...')
    img_feature = []
    for img_path in tqdm(img_paths, ncols=100):
        pos = img_path.rfind('/')
        img_path = '../feature/'+ img_path[pos + 1:]
        img = np.load(img_path + '.npy')
        img_feature.append(img)
    print('success')
    return img_feature

def get_word_index(captions, top_k=5000):
    tokenizer = tf.keras.preprocessing.text.Tokenizer(#num_words=top_k,
                                                      oov_token='<unk>',
                                                      filters='!"#$%&()*+,-./:;=?@[\]^_`{|}~\t\n\r')
    tokenizer.fit_on_texts(captions)
    tokenizer.word_index = {key: value for key, value in tokenizer.word_index.items()}
    tokenizer.word_index['<pad>'] = 0
    index_word = {value: key for key, value in tokenizer.word_index.items()}
    cap_vector = tokenizer.texts_to_sequences(captions)
    max_len = max([len(cap) for cap in cap_vector])
    return cap_vector, tokenizer, index_word, max_len

def data_generator(img_feature, cap_vector, max_len, vocab_size, batch_size=64):
    partial_caps = []
    images = []
    label = []

    count = 0
    while True:
        for i, (img, text) in enumerate(zip(img_feature, cap_vector)):
            for idx in range(1, len(text)):
                count += 1
                in_seq, out_seq = text[:idx], text[idx]
                in_seq = tf.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_len, padding='post')[0]
                out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
                images.append(img)
                partial_caps.append(in_seq)
                label.append(out_seq)

                if count >= batch_size:
                    images = np.array(images)
                    partial_caps = np.array(partial_caps)
                    label = np.array(label)
                    yield [[images, partial_caps], label]
                    partial_caps = []
                    label = []
                    images = []

def create_sequences(img_feature, cap_vector, max_len, vocab_size):
    partial_caps = []
    images = []
    label = []

    for i, (img, text) in tqdm(enumerate(zip(img_feature, cap_vector)), ncols=100):
        for idx in range(1, len(text)):
            in_seq, out_seq = text[:idx], text[idx]
            in_seq = tf.keras.preprocessing.sequence.pad_sequences([in_seq], maxlen=max_len, padding='post')[0]
            out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
            images.append(img)
            partial_caps.append(in_seq)
            label.append(out_seq)
    return np.array(images), np.array(partial_caps), np.array(label)


def plot_image(path, sentence):
    img = Image.open(path)
    plt.imshow(img)
    plt.title(sentence)
    plt.show()

if __name__ == '__main__':
    get_samples(num_examples=50000)
    get_train(train_size=20000)