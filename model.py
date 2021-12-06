import tensorflow.keras as keras

# create model with by subclassing keras.Model
class ChatModel(keras.Model):

    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(ChatModel, self).__init__()
        self.embed = keras.layers.Embedding(vocab_size, embed_dim)
        self.gru = keras.layers.GRU(hidden_dim, activation='relu', return_sequences=True, return_state=True)
        self.dense = keras.layers.Dense(vocab_size)
    
    def call(self, inputs, state=None, return_state=False):
        x = self.embed(inputs)

        # we need to be able to set state for text generation
        if state is None:
            x, state = self.gru(x)
        else:
            x, state = self.gru(x, initial_state=state)
        
        x = self.dense(x)

        # again, text gen needs state for next pass
        if return_state:
            return x, state
        else:
            return x


