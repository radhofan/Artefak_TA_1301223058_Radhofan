'''
Simple MLP Model Builder - Just creates and saves models
Expects GAN-augmented CSV to already exist
'''
import numpy as np
import keras
from keras import initializations
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Embedding, Input, Dense, merge, Flatten
from keras.optimizers import Adam
from load_movielens import load_movie_input
from time import time

#################### EDITABLE PARAMETERS ####################
AUGMENTED_CSV = 'movielens_100k_augmented.csv'  # GAN output (must exist!)
LEARNING_RATE = 0.001

# Model architectures
ARCHITECTURES = [
    {'name': 'tiny',   'layers': [4, 2],           'reg_layers': [0, 0]},
    {'name': 'small',  'layers': [8, 4],           'reg_layers': [0, 0]},
    {'name': 'medium', 'layers': [16, 8, 4],       'reg_layers': [0, 0, 0]},
    {'name': 'large',  'layers': [32, 16, 8, 4],   'reg_layers': [0, 0, 0, 0]},
    {'name': 'xlarge', 'layers': [64, 32, 16, 8],  'reg_layers': [0, 0, 0, 0]}
]

#################### BUILD MODEL ####################

def init_normal(shape, name=None):
    return initializations.normal(shape, scale=0.01, name=name)

def get_model(num_users, num_items, layers, reg_layers):
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    
    MLP_Embedding_User = Embedding(input_dim=num_users, output_dim=layers[0]/2, name='user_embedding',
                                  init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim=num_items, output_dim=layers[0]/2, name='item_embedding',
                                  init=init_normal, W_regularizer=l2(reg_layers[0]), input_length=1)
    
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    vector = merge([user_latent, item_latent], mode='concat')
    
    for idx in xrange(1, len(layers)):
        layer = Dense(layers[idx], W_regularizer=l2(reg_layers[idx]), activation='relu', name='layer%d' % idx)
        vector = layer(vector)
    
    prediction = Dense(1, activation='sigmoid', init='lecun_uniform', name='prediction')(vector)
    
    model = Model(input=[user_input, item_input], output=prediction)
    model.compile(optimizer=Adam(lr=LEARNING_RATE), loss='binary_crossentropy')
    
    return model

#################### MAIN ####################

if __name__ == '__main__':
    # Load GAN-augmented data
    print("Loading GAN-augmented data from: %s" % AUGMENTED_CSV)
    ncf_data = load_movie_input(csv_path=AUGMENTED_CSV, binary_labels=True, threshold=4)
    
    num_users = ncf_data['num_users']
    num_items = ncf_data['num_items']
    
    print("Loaded data: #user=%d, #item=%d" % (num_users, num_items))
    
    # Create models
    print("\nCreating models...")
    for arch in ARCHITECTURES:
        model = get_model(num_users, num_items, arch['layers'], arch['reg_layers'])
        filename = 'Pretrain/ml-100k_MLP_%s_%d.h5' % (arch['name'], time())
        model.save_weights(filename, overwrite=True)
        print("Saved: %s (layers: %s)" % (filename, arch['layers']))
    
    print("\nDone. Created %d models." % len(ARCHITECTURES))