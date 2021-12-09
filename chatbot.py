from data_processor import get_dataset
from model import ChatModel
import tensorflow.keras as keras
import tensorflow as tf
import math
import numpy as np

# seed the bois
def seed_everything(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
seed_everything(42)

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
def send_text(model, text, state=None):
    chars = tf.strings.unicode_split(text, input_encoding='UTF-8')
    ids = chars_to_ids(chars)
    ids = np.reshape(ids, (1, -1))
    if state is None:
        (_, state) = model(inputs=ids, return_state=True)
    else:
        (_, state) = model(inputs=ids, return_state=True, state=state)
    return state

def gen_text(model, state, stop_count=500, stop_str='You:\n'):
    count = 0 # for terminating with stop_count
    text = '' # models outputted text goes here

    # prime model with '\n'
    ids = chars_to_ids(['\n'])
    ids = np.reshape(ids, (1, -1))

    while stop_str not in text and count < stop_count:
        
        # pass one char in
        output, state = model(ids, state=state, return_state=True)
        assert output.shape[0] == 1 # make sure model only returns one char

        # sample from distrib to get output char id
        char_id = tf.random.categorical(output[:, 0, :], num_samples=1) # takes in 2dim (N, D)

        # convert back to char
        single_char = ids_to_chars(char_id)[0, 0]
        single_char = (
            single_char
            .numpy()
            .decode('utf-8')
        )

        # add to 'accumulation' string
        text += single_char

        # set up for next iteration
        ids = chars_to_ids([single_char])
        ids = np.reshape(ids, (1, -1))
        count += 1
    
    return text

if __name__ == '__main__':

    # load up the model
    # code broke with load_model(), would only accept RNN sequences of the length i trained on
    # using load_weights() instead
    model = ChatModel(len(vocab), 20, 20)
    model.load_weights('current_model/ChatModel')

    # display instructions
    print("""
Welcome to the Brain Chatbot!
It is a texting interface that responds like my friend brian.
Whenever you see 'You:', it is your turn to type.
Press enter to send your response.
""")

    # prime the model with 'You:\n'
    state = send_text(model, 'You:\n')
    user_resp = input('You:\n') + '\n' # '\n' added because the model expects it

    while True:
        # pass user text into the model
        state = send_text(model, user_resp, state=state)

        # generate model's response
        model_resp = gen_text(model, state)

        # display for the user
        print(model_resp)

        # get the next user input
        user_resp = input('') # no + '\n' becaues gen_text func uses it to prime the resp



