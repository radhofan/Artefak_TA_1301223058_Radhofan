import numpy as np
import pandas as pd
import tensorflow as tf

# SET SEED
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
import random
random.seed(SEED)

from tensorflow.keras.models import Model # type: ignore
from tensorflow.keras.layers import Embedding, Input, Dense, Flatten, Concatenate # type: ignore
from tensorflow.keras.regularizers import l2 # type: ignore
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils import load_movie_input
from sklearn.metrics import f1_score, precision_score, mean_absolute_error
from sklearn.neighbors import NearestNeighbors
import glob
import argparse

#################### METRICS ####################

def calculate_consistency(y_true, y_pred, k=5):
    nbrs = NearestNeighbors(n_neighbors=min(k+1, len(y_pred)), algorithm='auto').fit(y_pred.reshape(-1, 1))
    distances, indices = nbrs.kneighbors(y_pred.reshape(-1, 1))
    
    consistency_scores = []
    for i in range(len(y_pred)):
        if len(indices[i]) > 1:
            nn_idx = indices[i][1]
            consistency_scores.append(abs(y_pred[i] - y_pred[nn_idx]) / 2)
    
    return 1 - np.mean(consistency_scores) if consistency_scores else 0

def calculate_theil_index(values):
    values = values[values > 0]
    if len(values) == 0:
        return 0
    mu = np.mean(values)
    if mu == 0:
        return 0
    return np.mean((values / mu) * np.log(values / mu + 1e-10))

def calculate_maed(y_pred, groups):
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return 0
    exposures = [np.mean(y_pred[groups == g]) for g in unique_groups]
    return np.mean([abs(e1 - e2) for i, e1 in enumerate(exposures) for e2 in exposures[i+1:]])

def calculate_absolute_unfairness(y_true, y_pred, groups):
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        return 0
    
    errors = []
    for g in unique_groups:
        mask = groups == g
        if np.sum(mask) > 0:
            group_error = np.abs(np.mean(y_pred[mask]) - np.mean(y_true[mask]))
            errors.append(group_error)
    
    if len(errors) < 2:
        return 0
    return np.mean([abs(e1 - e2) for i, e1 in enumerate(errors) for e2 in errors[i+1:]])

def calculate_hit_ratio_per_user(user_input, item_input, labels, predictions, k=10, threshold=0.5):
    unique_users = np.unique(user_input)
    hit_rates = []
    
    for user in unique_users:
        user_mask = user_input == user
        user_labels = labels[user_mask]
        user_preds = predictions[user_mask]
        
        if len(user_preds) == 0:
            continue
            
        top_k = min(k, len(user_preds))
        top_k_indices = np.argsort(user_preds)[-top_k:]
        hits = np.sum(user_labels[top_k_indices] >= threshold)
        hit_rates.append(1 if hits > 0 else 0)
    
    return np.mean(hit_rates) if hit_rates else 0

def calculate_ndcg_per_user(user_input, item_input, labels, predictions, k=10):
    unique_users = np.unique(user_input)
    ndcg_scores = []
    
    for user in unique_users:
        user_mask = user_input == user
        user_labels = labels[user_mask]
        user_preds = predictions[user_mask]
        
        if len(user_preds) == 0:
            continue
            
        top_k = min(k, len(user_preds))
        top_k_indices = np.argsort(user_preds)[-top_k:][::-1]
        dcg = np.sum([(2**user_labels[i] - 1) / np.log2(idx + 2) for idx, i in enumerate(top_k_indices)])
        
        ideal_indices = np.argsort(user_labels)[-top_k:][::-1]
        idcg = np.sum([(2**user_labels[i] - 1) / np.log2(idx + 2) for idx, i in enumerate(ideal_indices)])
        
        ndcg_scores.append(dcg / idcg if idcg > 0 else 0)
    
    return np.mean(ndcg_scores) if ndcg_scores else 0

#################### MODEL BUILDER ####################

def get_model(num_users, num_items, layers, reg_layers):
    user_input = Input(shape=(1,), dtype='int32', name='user_input')
    item_input = Input(shape=(1,), dtype='int32', name='item_input')
    
    MLP_Embedding_User = Embedding(
        input_dim=num_users, output_dim=layers[0]//2, name='user_embedding',
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        embeddings_regularizer=l2(reg_layers[0])
    )
    
    MLP_Embedding_Item = Embedding(
        input_dim=num_items, output_dim=layers[0]//2, name='item_embedding',
        embeddings_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        embeddings_regularizer=l2(reg_layers[0])
    )
    
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    vector = Concatenate()([user_latent, item_latent])
    
    for idx in range(1, len(layers)):
        layer = Dense(layers[idx], kernel_regularizer=l2(reg_layers[idx]), 
                     activation='relu', name='layer%d' % idx)
        vector = layer(vector)
    
    prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', 
                      name='prediction')(vector)
    
    model = Model(inputs=[user_input, item_input], outputs=prediction)
    return model

#################### MAIN ####################

ARCHITECTURES = [
    {'name': 'tiny',   'layers': [4, 2],           'reg_layers': [0, 0]},
    {'name': 'small',  'layers': [8, 4],           'reg_layers': [0, 0]},
    {'name': 'medium', 'layers': [16, 8, 4],       'reg_layers': [0, 0, 0]},
    {'name': 'large',  'layers': [32, 16, 8, 4],   'reg_layers': [0, 0, 0, 0]},
    {'name': 'xlarge', 'layers': [64, 32, 16, 8],  'reg_layers': [0, 0, 0, 0]}
]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gan', type=str, default='ctgan')
    parser.add_argument('--dataset', type=str, default='ml-100k')
    args = parser.parse_args()
    
    from utils import split_train_test
    
    if args.gan == 'baseline':
        original_data = load_movie_input(csv_path=f'data/{args.dataset}/{args.dataset}.csv', binary_labels=True, threshold=4)
        _, test_data = split_train_test(original_data, test_ratio=0.2, random_state=42)
        num_users = original_data['num_users']
        num_items = original_data['num_items']
        model_files = glob.glob(f'models/baseline/{args.dataset}/*.weights.h5')
    else:
        original_data = load_movie_input(csv_path=f'data/{args.dataset}/{args.dataset}.csv', binary_labels=True, threshold=4)
        _, test_data = split_train_test(original_data, test_ratio=0.2, random_state=42)
        
        gan_data = load_movie_input(csv_path=f'generated/{args.gan}/{args.dataset}-augmented.csv', binary_labels=True, threshold=4)
        num_users = gan_data['num_users']
        num_items = gan_data['num_items']
        model_files = glob.glob(f'models/repaired/{args.gan}/{args.dataset}/*.weights.h5')
    
    user_input = test_data['user_input']
    item_input = test_data['item_input']
    labels = test_data['labels']
    groups = (test_data['gender'] == 'M').astype(int)
    
    valid_mask = (user_input < num_users) & (item_input < num_items)
    user_input = user_input[valid_mask]
    item_input = item_input[valid_mask]
    labels = labels[valid_mask]
    groups = groups[valid_mask]
    
    results = []
    
    for model_file in sorted(model_files):
        model_name = os.path.basename(model_file).replace('.weights.h5', '')
        arch = next((a for a in ARCHITECTURES if a['name'] == model_name), None)
        if not arch:
            continue
        
        model = get_model(num_users, num_items, arch['layers'], arch['reg_layers'])
        model.load_weights(model_file)
        predictions = model.predict([user_input, item_input], verbose=0).flatten()
        
        cnt = calculate_consistency(labels, predictions)
        ti = calculate_theil_index(predictions)
        maed = calculate_maed(predictions, groups)
        u_abs = calculate_absolute_unfairness(labels, predictions, groups)
        hr = calculate_hit_ratio_per_user(user_input, item_input, labels, predictions, k=10)
        ndcg = calculate_ndcg_per_user(user_input, item_input, labels, predictions, k=10)
        hr = np.mean(hr)
        ndcg = np.mean(ndcg)
        mae = mean_absolute_error(labels, predictions)
        pred_binary = (predictions >= 0.5).astype(int)
        macro_f1 = f1_score(labels, pred_binary, average='macro', zero_division=0)
        precision = precision_score(labels, pred_binary, average='macro', zero_division=0)
        
        results.append({
            'Model': model_name,
            'Layers': str(arch['layers']),
            'CNT': cnt,
            'TI': ti,
            'MAED': maed,
            'U_abs': u_abs,
            'HR@10': hr,
            'NDCG@10': ndcg,
            'MAE': mae,
            'Macro_F1': macro_f1,
            'Precision': precision
        })
    
    os.makedirs(f'results/{args.gan}/{args.dataset}', exist_ok=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'results/{args.gan}/{args.dataset}/results.csv', index=False)
    
    print("\n" + "="*80)
    print(f"EVALUATION RESULTS: {args.gan.upper()} on {args.dataset}")
    print("="*80)
    print(results_df.to_string(index=False))
    print("="*80)
    print(f"\nResults saved to: results/{args.gan}/{args.dataset}/results.csv\n")