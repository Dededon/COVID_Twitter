import os
import codecs
import numpy as np

os.environ['TF_KERAS'] = '1'

# Tensorflow Imports
import tensorflow as tf
from tensorflow.python import keras
import tensorflow.keras.backend as K
from keras import utils

# Keras-bert imports
from keras_bert import Tokenizer
from keras_bert import get_custom_objects
from keras_bert import load_trained_model_from_checkpoint


MAX_LEN = 128
pretrained_path = 'uncased_L-12_H-768_A-12'
config_path = os.path.join(pretrained_path, 'bert_config.json')
checkpoint_path = os.path.join(pretrained_path, 'bert_model.ckpt')
vocab_path = os.path.join(pretrained_path, 'vocab.txt')

token_dict = {}
with codecs.open(vocab_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)

TOKENIZER = Tokenizer(token_dict, cased=True)
print("length of BERT token dictionary:", len(token_dict))


model = load_trained_model_from_checkpoint(
    config_path,
    checkpoint_path,
    seq_len=MAX_LEN
)
model = keras.Model(model.input, get_custom_objects()['Extract'](0, name='Extract')(model.output))


def print_model_summary():
    model.summary()


def get_embedding(texts):
    text_id_input = list()
    segment_input = list()
    for text in texts:
        text_id = TOKENIZER.encode(text)[0]
        padding_len = MAX_LEN - len(text_id)
        text_id_input.append(text_id + [0] * padding_len)
        segment_input.append([0] * MAX_LEN)
    text_id_input = np.array(text_id_input)
    segment_input = np.array(segment_input)
    result = model.predict([text_id_input, segment_input])
    return result.mean(0)