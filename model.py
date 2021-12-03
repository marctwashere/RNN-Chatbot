from data_processor import get_dataset
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

# USER PARAMS
T = 20 # sequence length
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
dataset_ids = chars_to_ids(dataset_chars)

# create Dataset object for more efficient handling
dataset_obj = tf.data.Dataset.from_tensor_slices(dataset_ids)

# create batches for training
# seq_len+1 because input and target share characters with a spill of 1
# 'hello' -> ('hell','ello')
sequences = dataset_obj.batch(T+1, drop_remainder=True)

# create training and testing data
N_train = int(len(sequences)*split_ratio) # number of training samples
N_test = len(sequences) - N_train # number of test samples
X_train = np.empty((N_train, T))
Y_train = np.empty((N_train, T))
X_test = np.empty((N_test, T))
Y_test = np.empty((N_test, T))

for i in range(N_train):
    pass


# create the model
input = keras.layers.Embedding(V, D)
x = keras.layers.LSTM(M, activation='relu', return_sequences=True)(input)
output = keras.layers.Dense(V, activation='softmax')(x)