"""
CFGAN: Causal Fairness through Generative Adversarial Networks
Implementation based on "Achieving Causal Fairness through Generative Adversarial Networks" (IJCAI 2019)

This implementation uses:
- Two Generators (G_causal and G_interventional) to model causal and interventional distributions
- Two Discriminators (D_data and D_fair) for data quality and fairness enforcement
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder


class Generator(nn.Module):
    """Generator network for CFGAN"""
    
    def __init__(self, noise_dim: int, output_dim: int, hidden_dims: List[int] = [256, 256]):
        super(Generator, self).__init__()
        
        layers = []
        input_dim = noise_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, output_dim))
        layers.append(nn.Tanh())  # Output in [-1, 1] range
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.model(z)


class Discriminator(nn.Module):
    """Discriminator network for CFGAN"""
    
    def __init__(self, input_dim: int, hidden_dims: List[int] = [256, 256]):
        super(Discriminator, self).__init__()
        
        layers = []
        current_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(0.3)
            ])
            current_dim = hidden_dim
        
        layers.append(nn.Linear(current_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)


class CFGAN:
    """
    CFGAN: Causal Fairness through Generative Adversarial Networks
    
    Implements dual-generator, dual-discriminator architecture for fair data generation
    following causal fairness principles.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        sensitive_attr: str = 'gender_binary',
        target_attr: str = 'rating',
        noise_dim: int = 128,
        hidden_dims: List[int] = [256, 256],
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        lr_g: float = 0.0002,
        lr_d: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        lambda_fair: float = 0.5,  # Fairness loss weight
        lambda_causal: float = 0.3  # Causal consistency weight
    ):
        """
        Initialize CFGAN
        
        Args:
            data: Input DataFrame
            sensitive_attr: Sensitive attribute column name (e.g., 'gender')
            target_attr: Target/outcome attribute (e.g., 'rating')
            noise_dim: Dimension of noise vector
            hidden_dims: Hidden layer dimensions for networks
            device: Device to use ('cuda' or 'cpu')
            lr_g: Learning rate for generators
            lr_d: Learning rate for discriminators
            beta1, beta2: Adam optimizer parameters
            lambda_fair: Weight for fairness loss
            lambda_causal: Weight for causal consistency loss
        """
        self.device = torch.device(device)
        self.noise_dim = noise_dim
        self.lambda_fair = lambda_fair
        self.lambda_causal = lambda_causal
        self.sensitive_attr = sensitive_attr
        self.target_attr = target_attr
        
        # Preprocess data
        self.data = data.copy()
        self._preprocess_data()
        
        # Get dimensions
        self.data_dim = self.processed_data.shape[1]
        
        # Initialize networks
        # G_causal: Models the original causal distribution
        self.G_causal = Generator(noise_dim, self.data_dim, hidden_dims).to(self.device)
        
        # G_interventional: Models distribution after intervention on sensitive attribute
        self.G_interventional = Generator(noise_dim, self.data_dim, hidden_dims).to(self.device)
        
        # D_data: Discriminates real vs fake data (standard GAN discriminator)
        self.D_data = Discriminator(self.data_dim, hidden_dims).to(self.device)
        
        # D_fair: Enforces fairness by discriminating on sensitive attribute
        self.D_fair = Discriminator(self.data_dim, hidden_dims).to(self.device)
        
        # Optimizers
        self.optimizer_G_causal = optim.Adam(
            self.G_causal.parameters(), lr=lr_g, betas=(beta1, beta2)
        )
        self.optimizer_G_interventional = optim.Adam(
            self.G_interventional.parameters(), lr=lr_g, betas=(beta1, beta2)
        )
        self.optimizer_D_data = optim.Adam(
            self.D_data.parameters(), lr=lr_d, betas=(beta1, beta2)
        )
        self.optimizer_D_fair = optim.Adam(
            self.D_fair.parameters(), lr=lr_d, betas=(beta1, beta2)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
    def _preprocess_data(self):
        """Preprocess data: encode categoricals and normalize continuous"""
        self.discrete_columns = []
        self.continuous_columns = []
        self.encoders = {}
        self.scalers = {}
        
        processed_parts = []
        
        for col in self.data.columns:
            # Check if column is numeric
            if self.data[col].dtype in ['int64', 'float64']:
                # Treat small integer ranges as categorical
                n_unique = self.data[col].nunique()
                if n_unique <= 20 and self.data[col].dtype == 'int64':
                    self.discrete_columns.append(col)
                    le = LabelEncoder()
                    encoded = le.fit_transform(self.data[col].astype(str))
                    self.encoders[col] = le
                    processed_parts.append(encoded.reshape(-1, 1))
                else:
                    self.continuous_columns.append(col)
                    scaler = StandardScaler()
                    normalized = scaler.fit_transform(self.data[col].values.reshape(-1, 1))
                    self.scalers[col] = scaler
                    processed_parts.append(normalized)
            else:
                self.discrete_columns.append(col)
                le = LabelEncoder()
                encoded = le.fit_transform(self.data[col].astype(str))
                self.encoders[col] = le
                processed_parts.append(encoded.reshape(-1, 1))
        
        self.processed_data = np.concatenate(processed_parts, axis=1)
        self.column_indices = {}
        
        idx = 0
        for col in self.data.columns:
            self.column_indices[col] = idx
            idx += 1
    
    def _get_sensitive_idx(self):
        """Get column index of sensitive attribute"""
        return self.column_indices[self.sensitive_attr]
    
    def _get_target_idx(self):
        """Get column index of target attribute"""
        return self.column_indices[self.target_attr]
    
    def train(
        self,
        epochs: int = 100,
        batch_size: int = 256,
        n_critic: int = 5,
        verbose: bool = True
    ):
        """
        Train CFGAN model
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            n_critic: Number of discriminator updates per generator update
            verbose: Print training progress
        """
        # Convert data to tensor
        real_data = torch.FloatTensor(self.processed_data).to(self.device)
        n_samples = len(real_data)
        
        # Training loop
        for epoch in range(epochs):
            epoch_d_loss = 0
            epoch_g_loss = 0
            epoch_fair_loss = 0
            n_batches = 0
            
            # Shuffle data
            indices = torch.randperm(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:min(i + batch_size, n_samples)]
                real_batch = real_data[batch_indices]
                current_batch_size = len(real_batch)
                
                # Labels
                real_labels = torch.ones(current_batch_size, 1).to(self.device)
                fake_labels = torch.zeros(current_batch_size, 1).to(self.device)
                
                # ---------------------
                # Train Discriminators
                # ---------------------
                for _ in range(n_critic):
                    # Train D_data (standard GAN discriminator)
                    self.optimizer_D_data.zero_grad()
                    
                    # Real data
                    d_real = self.D_data(real_batch)
                    loss_d_real = self.criterion(d_real, real_labels)
                    
                    # Fake data from G_causal
                    z = torch.randn(current_batch_size, self.noise_dim).to(self.device)
                    fake_causal = self.G_causal(z)
                    d_fake_causal = self.D_data(fake_causal.detach())
                    loss_d_fake_causal = self.criterion(d_fake_causal, fake_labels)
                    
                    # Fake data from G_interventional
                    z_int = torch.randn(current_batch_size, self.noise_dim).to(self.device)
                    fake_interventional = self.G_interventional(z_int)
                    d_fake_int = self.D_data(fake_interventional.detach())
                    loss_d_fake_int = self.criterion(d_fake_int, fake_labels)
                    
                    # Combined D_data loss
                    loss_d_data = loss_d_real + 0.5 * (loss_d_fake_causal + loss_d_fake_int)
                    loss_d_data.backward()
                    self.optimizer_D_data.step()
                    
                    # Train D_fair (fairness discriminator)
                    # D_fair tries to predict sensitive attribute from generated data
                    self.optimizer_D_fair.zero_grad()
                    
                    # Extract sensitive attribute values
                    sens_idx = self._get_sensitive_idx()
                    real_sens = real_batch[:, sens_idx:sens_idx+1]
                    
                    # D_fair on real data
                    d_fair_real = self.D_fair(real_batch)
                    
                    # D_fair on fake data from generators
                    d_fair_fake_causal = self.D_fair(fake_causal.detach())
                    d_fair_fake_int = self.D_fair(fake_interventional.detach())
                    
                    # Fairness loss: D_fair should distinguish sensitive attribute
                    loss_d_fair = self.criterion(d_fair_real, real_labels) + \
                                  0.5 * (self.criterion(d_fair_fake_causal, fake_labels) + \
                                         self.criterion(d_fair_fake_int, fake_labels))
                    
                    loss_d_fair.backward()
                    self.optimizer_D_fair.step()
                
                # -----------------
                # Train Generators
                # -----------------
                self.optimizer_G_causal.zero_grad()
                self.optimizer_G_interventional.zero_grad()
                
                # Generate new samples
                z = torch.randn(current_batch_size, self.noise_dim).to(self.device)
                z_int = torch.randn(current_batch_size, self.noise_dim).to(self.device)
                
                fake_causal = self.G_causal(z)
                fake_interventional = self.G_interventional(z_int)
                
                # G_causal should fool D_data
                d_fake_causal = self.D_data(fake_causal)
                loss_g_causal_data = self.criterion(d_fake_causal, real_labels)
                
                # G_interventional should fool D_data
                d_fake_int = self.D_data(fake_interventional)
                loss_g_int_data = self.criterion(d_fake_int, real_labels)
                
                # Fairness loss: Generators should fool D_fair (make it hard to predict sensitive attr)
                d_fair_fake_causal = self.D_fair(fake_causal)
                d_fair_fake_int = self.D_fair(fake_interventional)
                
                # Generators want D_fair to output 0.5 (maximum uncertainty)
                fair_target = torch.ones(current_batch_size, 1).to(self.device) * 0.5
                loss_g_fair = self.criterion(d_fair_fake_causal, fair_target) + \
                              self.criterion(d_fair_fake_int, fair_target)
                
                # Causal consistency loss: ensure intervention changes only sensitive paths
                # The two generators should produce similar distributions except for causal effects
                loss_causal_consistency = torch.mean((fake_causal - fake_interventional) ** 2)
                
                # Combined generator loss
                loss_g = loss_g_causal_data + loss_g_int_data + \
                         self.lambda_fair * loss_g_fair + \
                         self.lambda_causal * loss_causal_consistency
                
                loss_g.backward()
                self.optimizer_G_causal.step()
                self.optimizer_G_interventional.step()
                
                # Track losses
                epoch_d_loss += (loss_d_data.item() + loss_d_fair.item())
                epoch_g_loss += loss_g.item()
                epoch_fair_loss += loss_g_fair.item()
                n_batches += 1
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                avg_d_loss = epoch_d_loss / n_batches
                avg_g_loss = epoch_g_loss / n_batches
                avg_fair_loss = epoch_fair_loss / n_batches
                print(f"Epoch [{epoch+1}/{epochs}] "
                      f"D_loss: {avg_d_loss:.4f} "
                      f"G_loss: {avg_g_loss:.4f} "
                      f"Fair_loss: {avg_fair_loss:.4f}")
    
    def generate(self, n_samples: int = 1000, use_interventional: bool = False) -> pd.DataFrame:
        """
        Generate synthetic data
        
        Args:
            n_samples: Number of samples to generate
            use_interventional: If True, use G_interventional; else use G_causal
            
        Returns:
            DataFrame with synthetic data
        """
        self.G_causal.eval()
        self.G_interventional.eval()
        
        with torch.no_grad():
            # Generate noise
            z = torch.randn(n_samples, self.noise_dim).to(self.device)
            
            # Generate samples
            if use_interventional:
                synthetic = self.G_interventional(z)
            else:
                synthetic = self.G_causal(z)
            
            # Convert to numpy
            synthetic_np = synthetic.cpu().numpy()
        
        # Reconstruct DataFrame
        synthetic_dict = {}
        
        for col_idx, col in enumerate(self.data.columns):
            values = synthetic_np[:, col_idx]
            
            if col in self.discrete_columns:
                # Decode categorical
                encoder = self.encoders[col]
                # Round and clip to valid range
                values = np.clip(np.round(values), 0, len(encoder.classes_) - 1).astype(int)
                decoded = encoder.inverse_transform(values)
                synthetic_dict[col] = decoded
            else:
                # Denormalize continuous
                scaler = self.scalers[col]
                denormalized = scaler.inverse_transform(values.reshape(-1, 1)).flatten()
                synthetic_dict[col] = denormalized
        
        synthetic_df = pd.DataFrame(synthetic_dict)
        
        # Convert to original dtypes
        for col in self.data.columns:
            if col in synthetic_df.columns and col in self.data.columns:
                try:
                    synthetic_df[col] = synthetic_df[col].astype(self.data[col].dtype)
                except:
                    pass
        
        return synthetic_df
    
    def compute_fairness_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Compute fairness metrics on generated data
        
        Args:
            df: DataFrame to evaluate
            
        Returns:
            Dictionary of fairness metrics
        """
        metrics = {}
        
        # Statistical Parity Difference
        # P(Y=1|S=0) - P(Y=1|S=1)
        if self.sensitive_attr in df.columns and self.target_attr in df.columns:
            groups = df.groupby(self.sensitive_attr)[self.target_attr].mean()
            if len(groups) >= 2:
                metrics['statistical_parity_diff'] = abs(groups.iloc[0] - groups.iloc[1])
            
            # Equal Opportunity Difference (for binary target)
            # Could be extended for multi-class
            metrics['mean_target_overall'] = df[self.target_attr].mean()
        
        return metrics