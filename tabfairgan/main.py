import pandas as pd
import os
import argparse
import numpy as np
import random
import torch

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from tabfairgan import TFG

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-100k')
parser.add_argument('--epochs', type=int, default=50)  # was 200
parser.add_argument('--fair_epochs', type=int, default=10)  # was 50
parser.add_argument('--lamda', type=float, default= 0.01)  # was 0.5
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--device', type=str, default='cuda:0')
args = parser.parse_args()

data = pd.read_csv(f'data/{args.dataset}/{args.dataset}-fair.csv')
data_sample = data.sample(n=10000, random_state=42)

original_dtypes = data_sample.dtypes.to_dict()

data_sample['gender_binary'] = data_sample['gender_binary'].astype(str)
data_sample['rating'] = data_sample['rating'].astype(str)
data_sample['occupation'] = data_sample['occupation'].astype(str)
data_sample['user_id'] = data_sample['user_id'].astype(str)
data_sample['item_id'] = data_sample['item_id'].astype(str)

fairness_config = {
    'fair_epochs': args.fair_epochs,
    'lamda': args.lamda,
    'S': 'gender_binary',
    'Y': 'rating',
    'S_under': '0',
    'Y_desire': '5'
}

print("Starting TabFairGAN training...")
tfg = TFG(
    data_sample, 
    epochs=args.epochs, 
    batch_size=args.batch_size, 
    device=args.device,
    fairness_config=fairness_config
)

tfg.train()
print("Training complete!")

synthetic_data = tfg.generate_fake_df(num_rows=10000)

for col, dtype in original_dtypes.items():
    if col in synthetic_data.columns:
        synthetic_data[col] = synthetic_data[col].astype(dtype)

os.makedirs(f'generated/tabfairgan', exist_ok=True)
synthetic_data.to_csv(f'generated/tabfairgan/{args.dataset}-augmented.csv', index=False)
print(f"Generated {len(synthetic_data)} samples")