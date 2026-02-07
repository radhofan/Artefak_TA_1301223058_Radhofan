import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_movie_input
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-100k')
args = parser.parse_args()

ARCHITECTURES = [
    {'name': 'tiny',   'layers': [4, 2],           'reg_layers': [0, 0]},
    {'name': 'small',  'layers': [8, 4],           'reg_layers': [0, 0]},
    {'name': 'medium', 'layers': [16, 8, 4],       'reg_layers': [0, 0, 0]},
    {'name': 'large',  'layers': [32, 16, 8, 4],   'reg_layers': [0, 0, 0, 0]},
    {'name': 'xlarge', 'layers': [64, 32, 16, 8],  'reg_layers': [0, 0, 0, 0]}
]

def get_model(num_users, num_items, layers, reg_layers):
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0]//2, name='user_embedding',
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        embeddings_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0]//2, name='item_embedding',
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        embeddings_regularizer=l2(reg_layers[0]), input_length=1)
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    vector = Concatenate()([user_latent, item_latent])
    for idx in range(1, len(layers)):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)
        vector = layer(vector)
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name='prediction')(vector)
    model = Model(inputs=[user_input, item_input], outputs=prediction)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model

ncf_data = load_movie_input(csv_path=f'data/{args.dataset}/{args.dataset}.csv', binary_labels=True, threshold=4)
num_users = ncf_data['num_users']
num_items = ncf_data['num_items']
os.makedirs(f'models/baseline/{args.dataset}', exist_ok=True)
for arch in ARCHITECTURES:
    model = get_model(num_users, num_items, arch['layers'], arch['reg_layers'])
    model.save_weights(f'models/baseline/{args.dataset}/{arch["name"]}.weights.h5')