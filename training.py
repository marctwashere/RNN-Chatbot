"""
12/9/21
My loss kept exploding to nan. Even lowered the LR way down. Read online that this can be
due to ReLu instead of tanh, so I will switch and raise the LR some!

12/10/21
The loss decreases quite nicel with tanh but still explodes within the first epoch. Seems like a
particular input is messing things up. I'm going to do some code step through of the 
tensorflow library to try to find the input culprit.

With seed 42, the culprit batch is #144 in epoch #1. Tried training on dataset from batches 0 to 142
and this helped some but still exploded on the third epoch. At least I know now that the
problem is not nan values in the input (because it made it through 2 epochs)

When I forced GPU execution to be off, (CPU only) I received an InvalidArgumentError from the Embed
Layer complaining that the index 210 was looked up but vocab only spans from indexes 0-209. Not sure
why this was the only time an exception was raised instead of just continuing with NaN like
the GPU runs did.

Exception has occurred: InvalidArgumentError       (note: full exception trace is shown but execution is paused at: _run_module_as_main)
 indices[47,99] = 210 is not in [0, 210)
	 [[node chat_model/embedding/embedding_lookup (model.py:18) ]] [Op:__inference_train_function_2103]

12/11/21
So np.count_nonzero(dataset_ids>=210) shows that only ONE id in the whole dataset is >= 210.
Seems like model.fit() does some batch shuffling of its own becaue looping through dataset_obj
shows that the guilty pair is #158. It contains 210

FOUND THE FREAKING BUG. So when I create the embedding layer in model.py, I pass in len(vocab) which is
210. However, that layer's constructor needs the max index (210) + 1 = 211. This is because the UNK token
which is not considered in len(vocab) also has an embed at index 0. It was only breaking on one training
batch because the lower indexes are emojis and the particular emoji '🧧' only shows up one time in the
entire dataset (idx 717401), which is why the training went to Nan so randomly. Just adjusting the embed 
csonstructor by adding 1 to inpud_dim.

Upon testing, I confirm that the bug is fixed!!!!!!!

"""
from data_processor import get_dataset
from model import ChatModel
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np

# USER PARAMS
T = 100 # sequence length
D = 256 # embedding dimensionality
M = 1024 # hidden layer dimensionality
split_ratio = 0.75 # train/test split ratio
batch_size = 64
num_epochs = 30
lr = 1e-4

# seed the bois
def seed_everything(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_everything(42)

# convert xml to script-like dataset
print('Building the dataset from XML file...')
dataset_str = get_dataset('text_messages.xml')

# TODO get rid of emojis or leave in?

# create the vocab by finding unique chars
vocab = sorted(set(dataset_str))
print('Vocab size (V): {}'.format(len(vocab)))

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
model = ChatModel(len(vocab)+1, D, M) # +1 to account for UNK token

# add loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True) # i removed soft-max so we are using logits now
opt = keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=opt, loss=loss)

# save every epochs for empirical progress tests
chkpt = keras.callbacks.ModelCheckpoint('models/epoch{epoch}', save_freq='epoch', save_weights_only=True)

# train that bad boy
model.fit(dataset_obj, epochs=num_epochs, callbacks=[chkpt])

print('debug')


