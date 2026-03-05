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

from be_great import GReaT

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-100k')
parser.add_argument('--epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)  
args = parser.parse_args()

data = pd.read_csv(f'data/{args.dataset}/{args.dataset}-fair.csv')
data_sample = data.sample(n=20000, random_state=42)

print("Starting GReaT training...")
great = GReaT(
    llm='distilgpt2',
    epochs=args.epochs,
    batch_size=args.batch_size,
    save_steps=100000
)
great.fit(data_sample)
print("Training complete!")

synthetic_data = great.sample(n_samples=20000)

os.makedirs('generated/great', exist_ok=True)
synthetic_data.to_csv(f'generated/great/{args.dataset}-augmented.csv', index=False)
print(f"Generated {len(synthetic_data)} samples")