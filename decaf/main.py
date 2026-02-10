# Installation: pip install synthcity
# If that fails, try: pip install synthcity --break-system-packages

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

from synthcity.plugins import Plugins
from synthcity.plugins.core.dataloader import GenericDataLoader

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-100k')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=500)
parser.add_argument('--datasize', type=int, default=1000)
args = parser.parse_args()

data = pd.read_csv(f'data/{args.dataset}/{args.dataset}-fair.csv')
data_sample = data.sample(n=args.datasize, random_state=42)

# Create data loader with sensitive column marked
loader = GenericDataLoader(
    data_sample,
    sensitive_columns=['gender', 'gender_binary']
)

print("Starting DECAF training...")

# Use DECAF as a simple plugin
# DECAF will infer the causal structure automatically
decaf = Plugins().get(
    "decaf",
    n_iter=args.epochs,
    batch_size=args.batch_size
)

# Train
decaf.fit(loader)
print("Training complete!")

# Generate synthetic data
synthetic_data = decaf.generate(count=args.datasize).dataframe()

os.makedirs('generated/decaf', exist_ok=True)
synthetic_data.to_csv(f'generated/decaf/{args.dataset}-augmented.csv', index=False)
print(f"Generated {len(synthetic_data)} samples")