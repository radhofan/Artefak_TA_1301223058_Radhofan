# import os
# import argparse
# import numpy as np
# import random
# import torch
# import pandas as pd
# SEED = 42
# np.random.seed(SEED)
# random.seed(SEED)
# torch.manual_seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(SEED)
#     torch.cuda.manual_seed_all(SEED)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
# from ctgan import CTGAN
# parser = argparse.ArgumentParser()
# parser.add_argument('--dataset', type=str, default='ml-100k')
# args = parser.parse_args()

# data = pd.read_csv(f'data/{args.dataset}/{args.dataset}-fair.csv')
# data_sample = data.sample(n=10000, random_state=42)
# discrete_columns = ['user_id', 'item_id', 'rating', 'gender', 'gender_binary', 'occupation']

# ctgan = CTGAN(epochs=50, batch_size=500, verbose=True)
# print("Starting CTGAN training on full data...")
# ctgan.fit(data_sample, discrete_columns=discrete_columns)
# print("Training complete!")
# synthetic_data = ctgan.sample(10000)

# female_data = data[data['gender'] == 'F']
# print(f"Female rows in fair data: {len(female_data)}")
# ctgan_female = CTGAN(epochs=50, batch_size=500, verbose=True)
# print("Starting CTGAN training on female-only data...")
# ctgan_female.fit(female_data, discrete_columns=discrete_columns)
# print("Female CTGAN training complete!")
# synthetic_female = ctgan_female.sample(50000)
# print(f"Generated {len(synthetic_female)} synthetic female samples")

# combined = pd.concat([synthetic_data, synthetic_female], ignore_index=True)
# os.makedirs('generated/ctgan', exist_ok=True)
# combined.to_csv(f'generated/ctgan/{args.dataset}-augmented.csv', index=False)
# print(f"Total generated samples: {len(combined)}")

import os
import argparse
import numpy as np
import random
import torch
import pandas as pd

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

from ctgan import CTGAN

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-100k')
args = parser.parse_args()

data = pd.read_csv(f'data/{args.dataset}/{args.dataset}-fair.csv')
data_sample = data.sample(n=20000, random_state=42)

discrete_columns = ['user_id', 'item_id', 'rating', 'gender', 'gender_binary', 'occupation']
ctgan = CTGAN(epochs=50, batch_size=500, verbose=True)

print("Starting CTGAN training...")
ctgan.fit(data_sample, discrete_columns=discrete_columns)

print("Training complete!")
synthetic_data = ctgan.sample(20000)

os.makedirs('generated', exist_ok=True)
synthetic_data.to_csv(f'generated/ctgan/{args.dataset}-augmented.csv', index=False)
print(f"Generated {len(synthetic_data)} samples")