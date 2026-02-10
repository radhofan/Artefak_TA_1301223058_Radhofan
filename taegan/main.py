import pandas as pd
import numpy as np
import torch
import json
import os
import argparse
import random
from sklearn.preprocessing import LabelEncoder

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from taegan import TAEGAN

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-100k')
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--warmup_epochs', type=int, default=9)
parser.add_argument('--batch_size', type=int, default=500)
args = parser.parse_args()

# Load data
data = pd.read_csv(f'data/{args.dataset}/{args.dataset}-fair.csv')
data_sample = data.sample(n=10000, random_state=42)

# Prepare cache directory
cache_dir = f'taegan/taegan_cache/{args.dataset}'
os.makedirs(cache_dir, exist_ok=True)

print("Preprocessing data for TAEGAN...")

# Identify discrete columns
discrete_columns = ['user_id', 'item_id', 'rating', 'gender', 'gender_binary', 'occupation']
continuous_columns = [col for col in data_sample.columns if col not in discrete_columns]

# Encode discrete columns and create one-hot
encoded_data = []
span_info = []
encoders = {}

for col in data_sample.columns:
    if col in discrete_columns:
        # One-hot encode discrete columns
        le = LabelEncoder()
        encoded = le.fit_transform(data_sample[col].astype(str))
        n_classes = len(le.classes_)
        one_hot = np.eye(n_classes)[encoded]
        encoded_data.append(one_hot)
        span_info.append([n_classes, "onehot"])
        encoders[col] = le
    else:
        # Normalize continuous columns to [-1, 1] range
        values = data_sample[col].values.reshape(-1, 1)
        min_val = values.min()
        max_val = values.max()
        normalized = 2 * (values - min_val) / (max_val - min_val + 1e-8) - 1
        encoded_data.append(normalized)
        span_info.append([1, "normal"])
        encoders[col] = {'min': min_val, 'max': max_val}

# Concatenate all features
train_data = np.concatenate(encoded_data, axis=1)
train_data_tensor = torch.FloatTensor(train_data)

# Save preprocessed data
torch.save(train_data_tensor, os.path.join(cache_dir, "train-data.pt"))
with open(os.path.join(cache_dir, "span-info.json"), "w") as f:
    json.dump(span_info, f)

# Save encoders for recovery
with open(os.path.join(cache_dir, "encoders.json"), "w") as f:
    # Convert encoders to serializable format
    serializable_encoders = {}
    for col, enc in encoders.items():
        if col in discrete_columns:
            serializable_encoders[col] = {
                'type': 'discrete',
                'classes': enc.classes_.tolist()
            }
        else:
            serializable_encoders[col] = {
                'type': 'continuous',
                'min': float(enc['min']),
                'max': float(enc['max'])
            }
    json.dump(serializable_encoders, f)

# Save column order
with open(os.path.join(cache_dir, "columns.json"), "w") as f:
    json.dump(list(data_sample.columns), f)

print("Starting TAEGAN training...")
model = TAEGAN(cache_dir)
model.train(
    batch_size=args.batch_size,
    epochs=args.epochs,
    warmup_epochs=args.warmup_epochs
)
print("Training complete!")

print("Generating synthetic data...")
synthetic_tensor = model.generate(n=10000, batch_size=args.batch_size)

# Recover synthetic data
print("Recovering synthetic data to original format...")
with open(os.path.join(cache_dir, "encoders.json"), "r") as f:
    serializable_encoders = json.load(f)
with open(os.path.join(cache_dir, "columns.json"), "r") as f:
    columns = json.load(f)

synthetic_data_dict = {}
idx = 0

for col in columns:
    enc_info = serializable_encoders[col]
    if enc_info['type'] == 'discrete':
        # Decode one-hot
        n_classes = len(enc_info['classes'])
        one_hot = synthetic_tensor[:, idx:idx+n_classes].numpy()
        decoded_indices = np.argmax(one_hot, axis=1)
        decoded_values = [enc_info['classes'][i] for i in decoded_indices]
        synthetic_data_dict[col] = decoded_values
        idx += n_classes
    else:
        # Denormalize continuous
        normalized = synthetic_tensor[:, idx:idx+1].numpy().flatten()
        denormalized = (normalized + 1) / 2 * (enc_info['max'] - enc_info['min'] + 1e-8) + enc_info['min']
        synthetic_data_dict[col] = denormalized
        idx += 1

synthetic_data = pd.DataFrame(synthetic_data_dict)

# Convert back to original dtypes
for col in data_sample.columns:
    if col in discrete_columns:
        synthetic_data[col] = synthetic_data[col].astype(data_sample[col].dtype)
    else:
        synthetic_data[col] = synthetic_data[col].astype(float)

os.makedirs('generated/taegan', exist_ok=True)
synthetic_data.to_csv(f'generated/taegan/{args.dataset}-augmented.csv', index=False)
print(f"Generated {len(synthetic_data)} samples")