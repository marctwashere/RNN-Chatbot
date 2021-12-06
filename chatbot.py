from data_processor import get_dataset
from model import ChatModel
import tensorflow.keras as keras
import tensorflow as tf
import math
import numpy as np

# convert xml to script-like dataset
dataset_str = get_dataset('text_messages.xml')

# TODO get rid of emojis or leave in?

# create the vocab by finding unique chars
vocab = sorted(set(dataset_str))
print('Vocab (V): {}'.format(len(vocab)))

# converts from textual character to an id that can be passed to embedding layer
chars_to_ids = keras.layers.StringLookup(vocabulary=list(vocab))

# converts from ids from model's output back into characters
ids_to_chars = keras.layers.StringLookup(vocabulary=list(vocab), invert=True)

# sends text to the model, returns hidden state
def send_text(model, text):
    chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
    ids = chars_to_ids(chars)
    ids = np.reshape(ids, (1, -1))
    (_, state) = model(inputs=ids, return_state=True)
    return state

def gen_text(model, state, stop_count=math.inf, stop_str='You:\n'):
    count = 0 # for terminating with stop_count
    text = '' # models outputted text goes here
    while stop_str not in text and count < stop_count:
        output, state = model(chars_to_ids(['\n']), state=state, return_state=True)
        assert len(output) == 1 # make sure model only returns one char
        single_char = ids_to_chars(output)[0]
        text += single_char
        count += 1

if __name__ == '__main__':

    # load up the model
    model = ChatModel(len(vocab), 20, 20)
    model.load_weights('current_model/ChatModel')

    # prime the model with 'You:\n'
    send_text(model, 'You:\n')

    # display instructions
    print("""
Welcome to the Brain Chatbot!
It is a texting interface that responds like my friend brian.
Whenever you see 'You:', it is your turn to type.
Press enter to send your response.
""")

    while True:
        user_resp = input('You:\n')
        

