'''
Simple MLP Model Builder - Loads existing models and retrains with APPENDED augmented data
Appends GAN-augmented data to the original dataset
'''
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_movie_input, split_train_val_test
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from time import time
import argparse
import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

#################### EDITABLE PARAMETERS ####################
parser = argparse.ArgumentParser()
parser.add_argument('--gan', type=str, default='ctgan')
parser.add_argument('--dataset', type=str, default='ml-100k')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=256)
args = parser.parse_args()

AUGMENTED_CSV = f'generated/{args.gan}/{args.dataset}-augmented.csv'
LEARNING_RATE = 0.001

ARCHITECTURES = [
    {'name': 'tiny',   'layers': [4, 2],           'reg_layers': [0, 0]},
    {'name': 'small',  'layers': [8, 4],           'reg_layers': [0, 0]},
    {'name': 'medium', 'layers': [16, 8, 4],       'reg_layers': [0, 0, 0]},
    {'name': 'large',  'layers': [32, 16, 8, 4],   'reg_layers': [0, 0, 0, 0]},
    {'name': 'xlarge', 'layers': [64, 32, 16, 8],  'reg_layers': [0, 0, 0, 0]}
]

#################### BUILD MODEL ####################
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
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='binary_crossentropy')

    return model

#################### MAIN ####################
if __name__ == '__main__':
    print("Loading original data...")
    original_data = load_movie_input(
        csv_path=f'data/{args.dataset}/{args.dataset}.csv',
        binary_labels=True,
        threshold=4
    )
    num_users = original_data['num_users']
    num_items = original_data['num_items']

    train_data, val_data, _ = split_train_val_test(original_data, test_ratio=0.2, val_ratio=0.1, random_state=42)

    print("Loading GAN-augmented data from: %s" % AUGMENTED_CSV)
    augmented_data = load_movie_input(csv_path=AUGMENTED_CSV, binary_labels=True, threshold=4)

    print("\nCombining original and augmented data...")
    train_users = np.concatenate([
        np.array(train_data['user_input']),
        np.array(augmented_data['user_input'])
    ])
    train_items = np.concatenate([
        np.array(train_data['item_input']),
        np.array(augmented_data['item_input'])
    ])
    train_labels = np.concatenate([
        np.array(train_data['labels']),
        np.array(augmented_data['labels'])
    ])

    valid_mask = (train_users < num_users) & (train_items < num_items)
    train_users = train_users[valid_mask]
    train_items = train_items[valid_mask]
    train_labels = train_labels[valid_mask]

    val_users = np.array(val_data['user_input'])
    val_items = np.array(val_data['item_input'])
    val_labels = np.array(val_data['labels'])

    valid_mask_val = (val_users < num_users) & (val_items < num_items)
    val_users = val_users[valid_mask_val]
    val_items = val_items[valid_mask_val]
    val_labels = val_labels[valid_mask_val]

    print("Model dimensions: #user=%d, #item=%d" % (num_users, num_items))
    print("Original train samples: %d" % len(train_data['user_input']))
    print("Augmented samples: %d" % len(augmented_data['user_input']))
    print("Total training samples: %d" % len(train_users))
    print("Validation samples: %d" % len(val_users))

    os.makedirs(f'models/repaired/{args.gan}/{args.dataset}', exist_ok=True)

    print("\nLoading and retraining models...")
    for arch in ARCHITECTURES:
        print("\n" + "="*50)
        print("Processing: %s" % arch['name'])
        print("="*50)

        original_path = f'models/baseline/{args.dataset}/{arch["name"]}.weights.h5'
        new_path = f'models/repaired/{args.gan}/{args.dataset}/{arch["name"]}.weights.h5'

        model = get_model(num_users, num_items, arch['layers'], arch['reg_layers'])

        if os.path.exists(original_path):
            model.load_weights(original_path)
            print("Loaded weights from: %s" % original_path)
        else:
            print("WARNING: Original model not found at %s - training from scratch" % original_path)

        early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        print("Retraining on combined data (original + augmented)...")
        t1 = time()
        history = model.fit(
            [train_users, train_items],
            train_labels,
            batch_size=args.batch_size,
            epochs=args.epochs,
            verbose=1,
            shuffle=True,
            validation_data=([val_users, val_items], val_labels),
            callbacks=[early_stop]
        )
        t2 = time()

        model.save_weights(new_path)
        print("Saved retrained model: %s" % new_path)
        print("Training time: %.1f seconds" % (t2 - t1))
        print("Final loss: %.4f" % history.history['loss'][-1])

    print("\n" + "="*50)
    print("Done. Retrained %d models." % len(ARCHITECTURES))
    print("="*50)