from data_processor import get_dataset
from model import ChatModel
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

# USER PARAMS
T = 100 # sequence length
D = 20 # embedding dimensionality
M = 20 # hidden layer dimensionality
split_ratio = 0.75 # train/test split ratio
batch_size = 64

# convert xml to script-like dataset
dataset_str = get_dataset('text_messages.xml')

# TODO get rid of emojis or leave in?

# create the vocab by finding unique chars
vocab = sorted(set(dataset_str))
print('Vocab (V): {}'.format(len(vocab)))

# converts from textual character to an id that can be passed to embedding layer
chars_to_ids = keras.layers.StringLookup(vocabulary=list(vocab))

# split dataset from str into individual chars
dataset_chars = tf.strings.unicode_split(dataset_str, input_encoding='UTF-8')

# chars into ids
dataset_ids = chars_to_ids(dataset_chars)

# create Dataset object for more efficient handling
dataset_obj = tf.data.Dataset.from_tensor_slices(dataset_ids)

# create batches for training
# seq_len+1 because input and target share characters with a spill of 1
# 'hello' -> ('hell','ello')
sequences = dataset_obj.batch(T+1, drop_remainder=True)

# map each sequence to an input and target pair
def seq_splitter(seq):
    input = seq[:-1]
    target = seq[1:]
    return input, target
dataset_obj = sequences.map(seq_splitter)

# shuffle and batch
dataset_obj = (
    dataset_obj
    .shuffle(10000)
    .batch(batch_size, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# use custom subclassed Model
model = ChatModel(len(vocab), D, M)

for x, y in dataset_obj.take(1):
    test = model(x)

print('debug')


