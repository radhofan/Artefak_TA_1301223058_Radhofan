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

from cfgan import CFGAN

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='ml-100k')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--datasize', type=int, default=20000)
parser.add_argument('--lambda_fair', type=float, default=0.5, help='Weight for fairness loss')
parser.add_argument('--lambda_causal', type=float, default=0.3, help='Weight for causal consistency')
parser.add_argument('--noise_dim', type=int, default=256, help='Dimension of noise vector')
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--use_interventional', action='store_true', help='Use interventional generator for sampling')
args = parser.parse_args()

# Load data
data = pd.read_csv(f'data/{args.dataset}/{args.dataset}.csv')
data_sample = data.sample(n=args.datasize, random_state=42)

print("Starting CFGAN training...")
print(f"Dataset: {args.dataset}")
print(f"Samples: {len(data_sample)}")
print(f"Device: {args.device}")
print(f"Epochs: {args.epochs}")
print(f"Lambda Fair: {args.lambda_fair}")
print(f"Lambda Causal: {args.lambda_causal}")
print("-" * 60)

# Initialize CFGAN
cfgan = CFGAN(
    data=data_sample,
    sensitive_attr='gender_binary',
    target_attr='rating',
    noise_dim=args.noise_dim,
    hidden_dims=[256, 256],
    device=args.device,
    lr_g=0.0002,
    lr_d=0.0002,
    lambda_fair=args.lambda_fair,
    lambda_causal=args.lambda_causal
)

# Train the model
cfgan.train(
    epochs=args.epochs,
    batch_size=args.batch_size,
    n_critic=5,
    verbose=True
)

print("\nTraining complete!")
print("-" * 60)

# Generate synthetic data
print("\nGenerating synthetic data...")
synthetic_data = cfgan.generate(
    n_samples=args.datasize,
    use_interventional=args.use_interventional
)

# Compute fairness metrics
print("\nComputing fairness metrics on generated data...")
fairness_metrics = cfgan.compute_fairness_metrics(synthetic_data)
print("Fairness Metrics:")
for metric, value in fairness_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Also compute on original data for comparison
print("\nFairness Metrics on Original Data:")
original_metrics = cfgan.compute_fairness_metrics(data_sample)
for metric, value in original_metrics.items():
    print(f"  {metric}: {value:.4f}")

# Save generated data
os.makedirs('generated/cfgan', exist_ok=True)
output_file = f'generated/cfgan/{args.dataset}-augmented.csv'
synthetic_data.to_csv(output_file, index=False)

print(f"\nGenerated {len(synthetic_data)} samples")
print(f"Saved to: {output_file}")
print("-" * 60)

# Display sample of generated data
print("\nSample of generated data (first 5 rows):")
print(synthetic_data.head())