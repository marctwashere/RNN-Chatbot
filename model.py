from data_processor import get_dataset
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

# USER PARAMS
T = 100 # sequence length
D = 20 # embedding dimensionality
M = 20 # hidden layer dimensionality
split_ratio = 0.75 # train/test split ratio

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

# create model with by subclassing keras.Model
class TextGen(keras.Model):

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(TextGen, self).__init__()
        self.embed = keras.layers.Embedding(vocab_size, embed_dim)
        self.lstm = keras.layers.LSTM(hidden_dim, activation='relu', return_sequences=True)
        self.dense = keras.layers.Dense(vocab_size, activation='softmax')
    
    def call(self, inputs, state=None, return_state=False):
        x = self.embed(inputs)

        # we need to be able to set state for text generation
        if state is None:
            x, state = self.lstm(x)
        else:
            x, state = self.lstm(x, initial_state=state)
        x = self.dense(x)

        # again, text gen needs state for next pass
        if return_state:
            return x, state
        else:
            return x
    
    


