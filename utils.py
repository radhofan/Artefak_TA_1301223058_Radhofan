import pandas as pd
import numpy as np
import os
import argparse
import random

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

def load_movie_aug(data_path='data/ml-100k', output_csv='movielens_100k_for_gan.csv'):
    ratings = pd.read_csv(
        os.path.join(data_path, 'u.data'),
        sep='\t',
        names=['user_id', 'item_id', 'rating', 'timestamp'],
        engine='python'
    )
    
    users = pd.read_csv(
        os.path.join(data_path, 'u.user'),
        sep='|',
        names=['user_id', 'age', 'gender', 'occupation', 'zip_code'],
        engine='python'
    )
    
    data = pd.merge(ratings, users, on='user_id', how='left')
    
    data['gender_binary'] = (data['gender'] == 'M').astype(int)
    
    gan_data = data[['user_id', 'item_id', 'rating', 'timestamp', 'gender', 'gender_binary', 'age', 'occupation']]
    
    gan_data.to_csv(output_csv, index=False)
    
    print("=" * 80)
    print("MovieLens 100k Dataset Loaded for GAN Training")
    print("=" * 80)
    print(f"Total interactions: {len(gan_data)}")
    print(f"Unique users: {gan_data['user_id'].nunique()}")
    print(f"Unique items: {gan_data['item_id'].nunique()}")
    print(f"\nGender distribution:")
    print(gan_data['gender'].value_counts())
    print(f"\nRating distribution:")
    print(gan_data['rating'].value_counts().sort_index())
    print(f"\nData saved to: {output_csv}")
    print("=" * 80)
    print(f"\nFirst 5 rows:")
    print(gan_data.head())
    
    return gan_data


def load_movie_input(csv_path='data/ml-100k/movielens_100k_for_gan.csv', binary_labels=True, threshold=4):
    data = pd.read_csv(csv_path)
    user_input = data['user_id'].values
    item_input = data['item_id'].values
    
    if binary_labels:
        labels = (data['rating'] >= threshold).astype(np.float32).values
    else:
        labels = data['rating'].values
    
    gender = data['gender'].values
    gender_binary = data['gender_binary'].values
    
    num_users = data['user_id'].nunique()
    num_items = data['item_id'].nunique()
    
    print("=" * 80)
    print("NCF Input Data Prepared")
    print("=" * 80)
    print(f"Total samples: {len(user_input)}")
    print(f"Unique users: {num_users}")
    print(f"Unique items: {num_items}")
    
    if binary_labels:
        print(f"\nBinary label distribution (threshold={threshold}):")
        print(f"  Positive (1): {labels.sum():.0f} ({100*labels.mean():.2f}%)")
        print(f"  Negative (0): {(1-labels).sum():.0f} ({100*(1-labels.mean()):.2f}%)")
    
    print(f"\nGender distribution (metadata):")
    unique, counts = np.unique(gender, return_counts=True)
    for g, c in zip(unique, counts):
        print(f"  {g}: {c} ({100*c/len(gender):.2f}%)")
    
    print("=" * 80)
    
    return {
        'user_input': user_input,
        'item_input': item_input,
        'labels': labels,
        'gender': gender,
        'gender_binary': gender_binary,
        'num_users': num_users,
        'num_items': num_items,
        'raw_data': data 
    }


def split_train_test(ncf_data, test_ratio=0.2, random_state=42):
    np.random.seed(random_state)
    
    n_samples = len(ncf_data['user_input'])
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    
    test_size = int(n_samples * test_ratio)
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    train_data = {
        'user_input': ncf_data['user_input'][train_indices],
        'item_input': ncf_data['item_input'][train_indices],
        'labels': ncf_data['labels'][train_indices],
        'gender': ncf_data['gender'][train_indices],
        'gender_binary': ncf_data['gender_binary'][train_indices],
        'num_users': ncf_data['num_users'],
        'num_items': ncf_data['num_items']
    }
    
    test_data = {
        'user_input': ncf_data['user_input'][test_indices],
        'item_input': ncf_data['item_input'][test_indices],
        'labels': ncf_data['labels'][test_indices],
        'gender': ncf_data['gender'][test_indices],
        'gender_binary': ncf_data['gender_binary'][test_indices],
        'num_users': ncf_data['num_users'],
        'num_items': ncf_data['num_items']
    }
    
    print(f"Split complete: {len(train_indices)} train, {len(test_indices)} test")
    
    return train_data, test_data


def split_train_val_test(ncf_data, test_ratio=0.2, val_ratio=0.1, random_state=42):
    np.random.seed(random_state)

    n_samples = len(ncf_data['user_input'])
    indices = np.arange(n_samples)
    np.random.shuffle(indices)

    test_size = int(n_samples * test_ratio)
    val_size = int(n_samples * val_ratio)

    test_indices = indices[:test_size]
    val_indices = indices[test_size:test_size + val_size]
    train_indices = indices[test_size + val_size:]

    def slice_data(idx):
        return {
            'user_input': ncf_data['user_input'][idx],
            'item_input': ncf_data['item_input'][idx],
            'labels': ncf_data['labels'][idx],
            'gender': ncf_data['gender'][idx],
            'gender_binary': ncf_data['gender_binary'][idx],
            'num_users': ncf_data['num_users'],
            'num_items': ncf_data['num_items']
        }

    train_data = slice_data(train_indices)
    val_data = slice_data(val_indices)
    test_data = slice_data(test_indices)

    print(f"Split complete: {len(train_indices)} train, {len(val_indices)} val, {len(test_indices)} test")

    return train_data, val_data, test_data


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ml-100k')
    args = parser.parse_args()
    
    if args.dataset == 'ml-100k':
        gan_data = load_movie_aug(
            data_path=f'data/{args.dataset}',
            output_csv=f'data/{args.dataset}/{args.dataset}.csv'
        )
    elif args.dataset == 'ml-200k':
        pass
    