from ctgan import CTGAN
import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-100k')
args = parser.parse_args()

data = pd.read_csv(f'data/{args.dataset}/{args.dataset}-fair.csv')
data_sample = data.sample(n=10000, random_state=42)

discrete_columns = ['user_id', 'item_id', 'rating', 'gender', 'gender_binary', 'occupation']
ctgan = CTGAN(epochs=50, batch_size=500, verbose=True)

print("Starting CTGAN training...")
ctgan.fit(data_sample, discrete_columns=discrete_columns)

print("Training complete!")
synthetic_data = ctgan.sample(10000)

os.makedirs('generated', exist_ok=True)
synthetic_data.to_csv(f'generated/ctgan/{args.dataset}-augmented.csv', index=False)
print(f"Generated {len(synthetic_data)} samples")