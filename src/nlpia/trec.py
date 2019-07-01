""" TREC Question Classification Dataset

~5000 sentences classified into ~5 types of Questions

Taxonomy:
Class   Definition
ABBREVIATION    abbreviation
  abb   abbreviation
  exp   expression abbreviated
ENTITY  entities
  animal    animals
  body  organs of body
  color     colors
  creative  inventions, books and other creative pieces
  currency  currency names
  dis.med.  diseases and medicine
  event     events
  food  food
  instrument    musical instrument
  lang  languages
  letter    letters like a-z
  other     other entities
  plant     plants
  product   products
  religion  religions
  sport     sports
  substance     elements and substances
  symbol    symbols and signs
  technique     techniques and methods
  term  equivalent terms
  vehicle   vehicles
  word  words with a special property
DESCRIPTION     description and abstract concepts
  definition    definition of sth.
  description   description of sth.
  manner    manner of an action
  reason    reasons
HUMAN   human beings
  group     a group or organization of persons
  ind   an individual
  title     title of a person
  description   description of a person
LOCATION    locations
  city  cities
  country   countries
  mountain  mountains
  other     other locations
  state     states
NUMERIC     numeric values
  code  postcodes or other codes
  count     number of sth.
  date  dates
  distance  linear measures
  money     prices
  order     ranks
  other     other numbers
  period    the lasting time of sth.
  percent   fractions
  speed     speed
  temp  temperature
  size  size, area and volume
  weight    weight
"""
import os
import regex
import requests

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

from keras import layers
from keras import Model
import keras.backend as K


# df = pd.concat([pd.read_html(f'http://cogcomp.org/Data/QA/QC/train_{i}.label')
#                 for i in (1000, 2000, 3000, 4000, 5500)], columns=['line'])

TREC_FILES = ['TREC_10.label'] + [f'train_{i}.label' for i in (1000, 2000, 3000, 4000, 5500)]
TREC_URLS = tuple([f'http://cogcomp.org/Data/QA/QC/{fn}' for fn in TREC_FILES])
TREC_CLASSES = [
    ('ABBR', 'ABBREVIATION', 'abbreviation'),
    ('ENTY', 'ENTITY', 'entities'),
    ('DESC', 'DESCRIPTION', 'description and abstract concepts'),
    ('HUM', 'HUMAN', 'human beings'),
    ('LOC', 'LOCATION', 'locations'),
    ('NUM', 'NUMERIC', 'numeric values'),
    ]
TREC_CLASS_ABBREVIATIONS = dict([(abb, name) for abb, name, desc in TREC_CLASSES])


def download_trec_urls(urls=TREC_URLS, output_filepath='trec_dataset.csv'):
    """ Download all 6 .label files for the TREC dataset and concatenate them into a DataFrame

    >>> df = download_trec_urls()
    >>> df.describe(include='all')
    >>> df.head()
      dataset        class subclass                                  sentence
    0    test      NUMERIC     dist      How far is it from Denver to Aspen ?
    1    test     LOCATION     city  What county is Modesto , California in ?
    2    test        HUMAN     desc                         Who was Galileo ?
    3    test  DESCRIPTION      def                         What is an atom ?
    4    test      NUMERIC     date          When did Hawaii become a state ?
           dataset   class subclass                 sentence
    count     5882    5882     5882                     5882
    unique       2       6       47                     5871
    top      train  ENTITY      ind  What is the Milky Way ?
    freq      5382    1339     1011                        2
    """
    responses = [requests.get(url) for url in TREC_URLS]
    lines = [('train' if 'train_' in r.url else 'test') + ':' + line for r in responses for line in r.text.splitlines()]

    lines = [tuple(regex.match(r'(train|test)[:]([A-Z]+)[:]([a-z]+)\s(.+)', line).groups()) for line in lines]
    df = pd.DataFrame(lines, columns='dataset class subclass sentence'.split())
    df = df.drop_duplicates()
    # df.describe(include='all')
    df['class'] = df['class'].apply(lambda x: TREC_CLASS_ABBREVIATIONS.get(x, x))
    return df


def read_trec_files(filepaths=TREC_FILES):
    """ process files like train_1000.label if they've already been downloaded (e.g. using curl or wget) """
    lines = []
    for fn in os.listdir():
        if fn.endswith('.label'):
            with open(fn, 'rb') as fin:
                lines.extend(('train', x) for x in fin.readlines())
    lines.extend([('test', x) for x in open('http://cogcomp.org/Data/QA/QC/TREC_10.label', 'rb').readlines()])
    lines = [(line[0], line[1].decode('latin')) for line in lines]
    lines = [[line[0]] + list(regex.match(r'([A-Z]+)[:]([a-z]+)\s(.+)', line[1])) for line in lines]
    # df = pd.concat([pd.read_html(f'http://cogcomp.org/Data/QA/QC/train_{i}.label')
    #                 for i in (1000, 2000, 3000, 4000, 5500)], columns=['line'])
    return pd.DataFrame(lines, columns='dataset class subclass sentence'.split())


SESSION = tf.Session()
SESSION.run([tf.global_variables_initializer(), tf.tables_initializer()])


class Embedder:
    def __init__(self, url="https://tfhub.dev/google/universal-sentence-encoder-large/3", session=SESSION):
        # Universal Sentence Encoder
        self.tf_model = hub.Module(url)
        self.session = session

    def embed_batch(self, sentences):
        return self.session.run(self.tf_model(sentences))

    def embed_sentence(self, s):
        return self.embed_batch([s])[0]

    def embed(self, *args, **kwargs):
        return self.tf_model(*args, **kwargs)

    def keras_embed(self, x):
        return self.tf_model(tf.squeeze(tf.cast(x, tf.string)),
                             signature="default", as_dict=True)["default"]


EMBEDDER = Embedder()


def build_classifier(embed_size=512, num_classes=6):
    input_text = layers.Input(shape=(1,), dtype=tf.string)
    embedding = layers.Lambda(EMBEDDER.keras_embed, output_shape=(embed_size,))(input_text)
    dense = layers.Dense(256, activation='relu')(embedding)
    pred = layers.Dense(num_classes, activation='softmax')(dense)
    model = Model(inputs=[input_text], outputs=pred)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def train_classifier(model, texts=None, labels=None, test_texts=None, test_labels=None, test_size=.1):
    global SESSION
    SESSION.close()
    df = None
    if isinstance(texts, pd.DataFrame):
        df, texts = texts, None
    if texts is None or labels is None:
        if df is None:
            df = download_trec_urls()
        if texts is None:
            texts = df['sentence']
        if labels is None:
            labels = df['class']
    if hasattr(texts, 'tolist'):
        texts = texts
    texts = np.array(texts, dtype=object)[:, np.newaxis]
    labels = np.asarray(pd.get_dummies(labels), dtype=np.int8)
    test_mask = np.random.binomial(1, test_size, size=(len(texts),)).astype(bool)
    train_texts, test_texts = texts[~test_mask, :], texts[test_mask, :]
    train_labels, test_labels = labels[~test_mask, :], labels[test_mask, :]
    with tf.Session() as SESSION:
        K.set_session(SESSION)
        SESSION.run(tf.global_variables_initializer())
        SESSION.run(tf.tables_initializer())
        history = model.fit(train_texts,
                            train_labels,
                            validation_data=(test_texts, test_labels),
                            epochs=10,
                            batch_size=32)
    model.save_weights('./trec_question_classifier_model.h5')
    return model, history
