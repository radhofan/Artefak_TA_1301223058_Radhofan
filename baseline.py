import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_movie_input, split_train_test
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from time import time
import argparse
import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-100k')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=256)
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
    
    MLP_Embedding_User = Embedding(
        input_dim=num_users, 
        output_dim=layers[0]//2, 
        name='user_embedding',
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        embeddings_regularizer=l2(reg_layers[0]), 
        input_length=1
    )
    
    MLP_Embedding_Item = Embedding(
        input_dim=num_items, 
        output_dim=layers[0]//2, 
        name='item_embedding',
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        embeddings_regularizer=l2(reg_layers[0]), 
        input_length=1
    )
    
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    vector = Concatenate()([user_latent, item_latent])
    
    for idx in range(1, len(layers)):
        layer = Dense(
            layers[idx], 
            kernel_regularizer=l2(reg_layers[idx]), 
            activation='relu', 
            name='layer%d' % idx
        )
        vector = layer(vector)
    
    prediction = Dense(
        1, 
        activation='sigmoid', 
        kernel_initializer='lecun_uniform', 
        name='prediction'
    )(vector)
    
    model = Model(inputs=[user_input, item_input], outputs=prediction)
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')
    return model

# Load and split data
print("Loading data...")
ncf_data = load_movie_input(csv_path=f'data/{args.dataset}/{args.dataset}.csv', binary_labels=True, threshold=4)
train_data, _ = split_train_test(ncf_data, test_ratio=0.2, random_state=42)

num_users = ncf_data['num_users']
num_items = ncf_data['num_items']

train_users = np.array(train_data['user_input'])
train_items = np.array(train_data['item_input'])
train_labels = np.array(train_data['labels'])

# Filter out invalid IDs (ensure 0-indexing)
valid_mask = (train_users < num_users) & (train_items < num_items)
train_users = train_users[valid_mask]
train_items = train_items[valid_mask]
train_labels = train_labels[valid_mask]

print(f"Model dimensions: #user={num_users}, #item={num_items}")
print(f"Training samples: {len(train_users)}")
print(f"User ID range: {train_users.min()} to {train_users.max()} (expected: 0 to {num_users-1})")
print(f"Item ID range: {train_items.min()} to {train_items.max()} (expected: 0 to {num_items-1})")

os.makedirs(f'models/baseline/{args.dataset}', exist_ok=True)

print("\nTraining baseline models...")
for arch in ARCHITECTURES:
    print("\n" + "="*50)
    print(f"Training {arch['name']}...")
    print("="*50)
    
    model = get_model(num_users, num_items, arch['layers'], arch['reg_layers'])
    
    # ACTUALLY TRAIN THE MODEL!
    t1 = time()
    history = model.fit(
        [train_users, train_items],
        train_labels,
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        shuffle=True
    )
    t2 = time()
    
    model.save_weights(f'models/baseline/{args.dataset}/{arch["name"]}.weights.h5')
    print(f"Saved trained baseline: {arch['name']}")
    print(f"Training time: {t2-t1:.1f} seconds")
    print(f"Final loss: {history.history['loss'][-1]:.4f}")

print("\n" + "="*50)
print(f"Done training {len(ARCHITECTURES)} baseline models!")
print("="*50)