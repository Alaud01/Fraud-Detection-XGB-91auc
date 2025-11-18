"""
VAE-based Synthetic Fraud Data Generator
=========================================
This script uses Variational Autoencoders (VAE) to generate synthetic fraud cases
to address class imbalance in the fraud detection dataset.

The VAE architecture is specifically designed for tabular data and focuses on:
1. Learning the distribution of fraud cases
2. Generating diverse and realistic synthetic fraud samples
3. Preserving important statistical properties
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class TabularVAE(nn.Module):
    """
    Variational Autoencoder for Tabular Data
    
    Architecture designed specifically for fraud transaction data with:
    - Multiple hidden layers with batch normalization and dropout
    - Separate encoder and decoder paths
    - Reparameterization trick for sampling
    - Beta-VAE variant for improved disentanglement
    """
    
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], latent_dim=32, dropout_rate=0.3):
        """
        Initialize the VAE architecture
        
        Parameters:
        -----------
        input_dim : int
            Number of input features
        hidden_dims : list
            List of hidden layer dimensions for encoder
        latent_dim : int
            Dimension of the latent space
        dropout_rate : float
            Dropout rate for regularization
        """
        super(TabularVAE, self).__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Encoder architecture
        encoder_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Latent space parameters
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)
        
        # Decoder architecture (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        
        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        decoder_layers.append(nn.Linear(hidden_dims[0], input_dim))
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        """Encode input into latent space parameters"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        where epsilon ~ N(0,1)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent vector back to input space"""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass through the VAE"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


def vae_loss_function(x_recon, x, mu, logvar, beta=1.0):
    """
    VAE loss function combining reconstruction loss and KL divergence
    
    Parameters:
    -----------
    x_recon : torch.Tensor
        Reconstructed input
    x : torch.Tensor
        Original input
    mu : torch.Tensor
        Mean of latent distribution
    logvar : torch.Tensor
        Log variance of latent distribution
    beta : float
        Weight for KL divergence term (beta-VAE)
    """
    # Reconstruction loss (MSE for continuous features)
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')
    
    # KL divergence loss
    # KL(N(mu, sigma) || N(0, 1)) = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss with beta weighting
    total_loss = recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, kl_loss


class FraudDataLoader:
    """
    Load and preprocess fraud detection data specifically for VAE training
    """
    
    def __init__(self, data_path, target_col='isFraud'):
        """
        Initialize data loader
        
        Parameters:
        -----------
        data_path : str
            Path to the fraud data CSV file
        target_col : str
            Name of the target column
        """
        self.data_path = data_path
        self.target_col = target_col
        self.scaler = None
        self.feature_names = None
        
    def load_and_preprocess(self, fraud_only=True):
        """
        Load and preprocess the fraud data
        
        Parameters:
        -----------
        fraud_only : bool
            If True, only return fraud cases for training
        
        Returns:
        --------
        X : np.ndarray
            Preprocessed feature data
        """
        print("Loading fraud data...")
        
        # Try to load from different possible locations
        try:
            df = pd.read_csv(self.data_path)
        except FileNotFoundError:
            print(f"File not found at {self.data_path}")
            print("Attempting to load from IEEE fraud detection data...")
            return self._load_ieee_data(fraud_only)
        
        # Extract fraud cases only
        if fraud_only and self.target_col in df.columns:
            fraud_df = df[df[self.target_col] == 1].copy()
            print(f"Loaded {len(fraud_df)} fraud cases from {len(df)} total transactions")
        else:
            fraud_df = df.copy()
        
        # Remove target column
        if self.target_col in fraud_df.columns:
            fraud_df = fraud_df.drop(columns=[self.target_col])
        
        # Store feature names
        self.feature_names = fraud_df.columns.tolist()
        
        # Handle missing values
        fraud_df = fraud_df.fillna(fraud_df.median())
        
        # Remove any remaining inf or nan values
        fraud_df = fraud_df.replace([np.inf, -np.inf], np.nan)
        fraud_df = fraud_df.fillna(0)
        
        # Convert to numpy array
        X = fraud_df.values.astype(np.float32)
        
        print(f"Preprocessed data shape: {X.shape}")
        return X
    
    def _load_ieee_data(self, fraud_only=True):
        """
        Load and preprocess IEEE fraud detection data with minimal preprocessing
        
        This method loads the raw IEEE data and applies basic preprocessing
        suitable for VAE training.
        """
        import os
        
        # Define paths
        train_trans_path = 'data/ieee-fraud-detection/train_transaction.csv'
        train_id_path = 'data/ieee-fraud-detection/train_identity.csv'
        
        if not os.path.exists(train_trans_path):
            raise FileNotFoundError("IEEE fraud detection data not found!")
        
        print("Loading IEEE fraud detection data...")
        
        # Load data
        train_trans = pd.read_csv(train_trans_path)
        train_id = pd.read_csv(train_id_path)
        
        # Merge datasets
        df = pd.merge(train_trans, train_id, on='TransactionID', how='left')
        
        # Extract fraud cases
        if fraud_only:
            fraud_df = df[df['isFraud'] == 1].copy()
            print(f"Extracted {len(fraud_df)} fraud cases from {len(df)} total transactions")
        else:
            fraud_df = df.copy()
        
        # Remove unnecessary columns
        cols_to_drop = ['TransactionID', 'isFraud', 'TransactionDT']
        fraud_df = fraud_df.drop(columns=[col for col in cols_to_drop if col in fraud_df.columns])
        
        # Handle categorical columns - simple label encoding
        for col in fraud_df.columns:
            if fraud_df[col].dtype == 'object':
                fraud_df[col] = pd.factorize(fraud_df[col])[0]
        
        # Store feature names
        self.feature_names = fraud_df.columns.tolist()
        
        # Handle missing values with median for numerical stability
        fraud_df = fraud_df.fillna(fraud_df.median())
        fraud_df = fraud_df.replace([np.inf, -np.inf], np.nan)
        fraud_df = fraud_df.fillna(0)
        
        # Convert to numpy
        X = fraud_df.values.astype(np.float32)
        
        print(f"Preprocessed IEEE data shape: {X.shape}")
        return X
    
    def scale_data(self, X):
        """
        Scale the data using RobustScaler (better for outliers)
        
        Parameters:
        -----------
        X : np.ndarray
            Input data to scale
        
        Returns:
        --------
        X_scaled : np.ndarray
            Scaled data
        """
        if self.scaler is None:
            self.scaler = RobustScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled
    
    def inverse_scale(self, X_scaled):
        """
        Inverse transform scaled data back to original scale
        
        Parameters:
        -----------
        X_scaled : np.ndarray
            Scaled data
        
        Returns:
        --------
        X : np.ndarray
            Data in original scale
        """
        if self.scaler is None:
            raise ValueError("Scaler not fitted yet!")
        
        return self.scaler.inverse_transform(X_scaled)


def train_vae(model, dataloader, optimizer, epochs, beta_schedule='constant', 
              beta_start=1.0, beta_end=4.0, device='cpu'):
    """
    Train the VAE model
    
    Parameters:
    -----------
    model : TabularVAE
        VAE model to train
    dataloader : DataLoader
        DataLoader for training data
    optimizer : torch.optim.Optimizer
        Optimizer for training
    epochs : int
        Number of training epochs
    beta_schedule : str
        Type of beta scheduling ('constant', 'linear', 'cyclical')
    beta_start : float
        Starting beta value
    beta_end : float
        Ending beta value
    device : str
        Device to train on ('cpu' or 'cuda')
    
    Returns:
    --------
    loss_history : dict
        Dictionary containing loss history
    """
    model.train()
    loss_history = {
        'total_loss': [],
        'recon_loss': [],
        'kl_loss': [],
        'beta': []
    }
    
    print(f"\nTraining VAE for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        
        # Calculate beta for this epoch (beta-VAE annealing)
        if beta_schedule == 'linear':
            beta = beta_start + (beta_end - beta_start) * (epoch / epochs)
        elif beta_schedule == 'cyclical':
            cycle_length = epochs // 4
            beta = beta_start + (beta_end - beta_start) * ((epoch % cycle_length) / cycle_length)
        else:  # constant
            beta = beta_start
        
        for batch_idx, (data,) in enumerate(dataloader):
            data = data.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            x_recon, mu, logvar = model(data)
            
            # Calculate loss
            loss, recon_loss, kl_loss = vae_loss_function(x_recon, data, mu, logvar, beta)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track losses
            epoch_total_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()
        
        # Average losses
        avg_total_loss = epoch_total_loss / len(dataloader.dataset)
        avg_recon_loss = epoch_recon_loss / len(dataloader.dataset)
        avg_kl_loss = epoch_kl_loss / len(dataloader.dataset)
        
        # Store history
        loss_history['total_loss'].append(avg_total_loss)
        loss_history['recon_loss'].append(avg_recon_loss)
        loss_history['kl_loss'].append(avg_kl_loss)
        loss_history['beta'].append(beta)
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - "
                  f"Total Loss: {avg_total_loss:.4f}, "
                  f"Recon Loss: {avg_recon_loss:.4f}, "
                  f"KL Loss: {avg_kl_loss:.4f}, "
                  f"Beta: {beta:.4f}")
    
    print("Training complete!")
    return loss_history


def generate_synthetic_fraud(model, n_samples, input_dim, temperature=1.0, 
                           diversity_boost=True, device='cpu', data_loader=None):
    """
    Generate synthetic fraud samples using the trained VAE
    
    Parameters:
    -----------
    model : TabularVAE
        Trained VAE model
    n_samples : int
        Number of synthetic samples to generate
    input_dim : int
        Input dimension
    temperature : float
        Temperature parameter for sampling (higher = more diverse)
    diversity_boost : bool
        If True, sample from expanded latent space for more diversity
    device : str
        Device to generate on
    data_loader : DataLoader, optional
        DataLoader for real data to estimate latent space distribution for data-aware sampling
    
    Returns:
    --------
    synthetic_data : np.ndarray
        Generated synthetic samples
    """
    model.eval()
    
    with torch.no_grad():
        if diversity_boost and data_loader is not None:
            print("Using data-aware sampling for diversity boost.")
            # Encode all real data to get latent space stats
            all_mu = []
            for (data,) in data_loader:
                data = data.to(device)
                mu, _ = model.encode(data)
                all_mu.append(mu)
            
            all_mu = torch.cat(all_mu, dim=0)
            
            # Calculate mean and std of the latent space distribution
            mu_mean = torch.mean(all_mu, dim=0)
            mu_std = torch.std(all_mu, dim=0)
            
            # Sample from a normal distribution with learned mean and scaled std
            z = torch.randn(n_samples, model.latent_dim).to(device)
            z = z * mu_std * temperature + mu_mean

        elif diversity_boost:
            # Fallback to simple temperature-based sampling if data_loader is not provided
            print("Using simple temperature-based sampling for diversity boost.")
            # Sample from N(0, temperature^2 * I) instead of N(0, I)
            z = torch.randn(n_samples, model.latent_dim).to(device) * temperature
        else:
            z = torch.randn(n_samples, model.latent_dim).to(device)
        
        # Decode latent samples
        synthetic_data = model.decode(z)
        synthetic_data = synthetic_data.cpu().numpy()
    
    return synthetic_data


def plot_training_history(loss_history, save_path='vae_training_history.png'):
    """
    Plot training history
    
    Parameters:
    -----------
    loss_history : dict
        Dictionary containing loss history
    save_path : str
        Path to save the plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Total loss
    axes[0, 0].plot(loss_history['total_loss'], label='Total Loss', color='blue')
    axes[0, 0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Reconstruction loss
    axes[0, 1].plot(loss_history['recon_loss'], label='Reconstruction Loss', color='green')
    axes[0, 1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # KL divergence loss
    axes[1, 0].plot(loss_history['kl_loss'], label='KL Divergence', color='red')
    axes[1, 0].set_title('KL Divergence Loss', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Beta schedule
    axes[1, 1].plot(loss_history['beta'], label='Beta Value', color='purple')
    axes[1, 1].set_title('Beta Schedule', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Beta')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}")
    plt.show()


def compare_distributions(real_data, synthetic_data, feature_names=None, 
                         n_features_to_plot=6, save_path='distribution_comparison.png'):
    """
    Compare distributions of real and synthetic data
    
    Parameters:
    -----------
    real_data : np.ndarray
        Real fraud data
    synthetic_data : np.ndarray
        Synthetic fraud data
    feature_names : list
        List of feature names
    n_features_to_plot : int
        Number of features to plot
    save_path : str
        Path to save the plot
    """
    n_features = min(n_features_to_plot, real_data.shape[1])
    
    # Select features with highest variance for visualization
    feature_vars = np.var(real_data, axis=0)
    top_features = np.argsort(feature_vars)[-n_features:]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, feature_idx in enumerate(top_features[:n_features]):
        ax = axes[i]
        
        # Plot histograms
        ax.hist(real_data[:, feature_idx], bins=50, alpha=0.5, label='Real', 
                color='blue', density=True)
        ax.hist(synthetic_data[:, feature_idx], bins=50, alpha=0.5, label='Synthetic', 
                color='red', density=True)
        
        # Set labels
        if feature_names and feature_idx < len(feature_names):
            title = f'{feature_names[feature_idx]}'
        else:
            title = f'Feature {feature_idx}'
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Distribution comparison plot saved to {save_path}")
    plt.show()


def evaluate_synthetic_quality(real_data, synthetic_data):
    """
    Evaluate the quality of synthetic data using statistical tests
    
    Parameters:
    -----------
    real_data : np.ndarray
        Real fraud data
    synthetic_data : np.ndarray
        Synthetic fraud data
    
    Returns:
    --------
    metrics : dict
        Dictionary containing quality metrics
    """
    metrics = {}
    
    # 1. Mean and standard deviation comparison
    real_mean = np.mean(real_data, axis=0)
    synth_mean = np.mean(synthetic_data, axis=0)
    mean_diff = np.mean(np.abs(real_mean - synth_mean))
    
    real_std = np.std(real_data, axis=0)
    synth_std = np.std(synthetic_data, axis=0)
    std_diff = np.mean(np.abs(real_std - synth_std))
    
    metrics['mean_difference'] = mean_diff
    metrics['std_difference'] = std_diff
    
    # 2. Correlation matrix comparison
    real_corr = np.corrcoef(real_data.T)
    synth_corr = np.corrcoef(synthetic_data.T)
    corr_diff = np.mean(np.abs(real_corr - synth_corr))
    
    metrics['correlation_difference'] = corr_diff
    
    # 3. Kolmogorov-Smirnov test for distribution similarity
    ks_statistics = []
    for i in range(min(20, real_data.shape[1])):  # Test first 20 features
        ks_stat, p_value = stats.ks_2samp(real_data[:, i], synthetic_data[:, i])
        ks_statistics.append(ks_stat)
    
    metrics['mean_ks_statistic'] = np.mean(ks_statistics)
    
    # 4. Coverage - percentage of synthetic samples within real data range
    real_min = np.min(real_data, axis=0)
    real_max = np.max(real_data, axis=0)
    
    within_range = np.logical_and(
        synthetic_data >= real_min,
        synthetic_data <= real_max
    )
    coverage = np.mean(within_range)
    
    metrics['coverage'] = coverage
    
    print("\n" + "="*60)
    print("Synthetic Data Quality Metrics")
    print("="*60)
    print(f"Mean Difference: {metrics['mean_difference']:.6f}")
    print(f"Std Difference: {metrics['std_difference']:.6f}")
    print(f"Correlation Difference: {metrics['correlation_difference']:.6f}")
    print(f"Mean KS Statistic: {metrics['mean_ks_statistic']:.6f}")
    print(f"Coverage (within range): {metrics['coverage']:.2%}")
    print("="*60)
    
    return metrics


def main():
    """
    Main function to train VAE and generate synthetic fraud data
    """
    
    print("="*60)
    print("VAE-Based Synthetic Fraud Data Generator")
    print("="*60)
    
    # Configuration
    CONFIG = {
        'data_path': 'data/synthetic/fraud_data.csv',  # Adjust if needed
        'batch_size': 64,
        'epochs': 150,
        'learning_rate': 0.001,
        'latent_dim': 64,
        'hidden_dims': [512, 256, 128],
        'dropout_rate': 0.3,
        'beta_schedule': 'cyclical',  # 'constant', 'linear', or 'cyclical'
        'beta_start': 0.5,
        'beta_end': 2.0,
        'n_synthetic_samples': 10000,  # Number of synthetic samples to generate
        'temperature': 2.0,  # Increased temperature for more diversity
        'diversity_boost': True,
        'save_model': True,
        'model_path': 'vae_fraud_model.pt'
    }
    
    # 1. Load and preprocess data
    print("\n" + "="*60)
    print("Step 1: Loading and Preprocessing Data")
    print("="*60)
    
    data_loader = FraudDataLoader(CONFIG['data_path'])
    X_fraud = data_loader.load_and_preprocess(fraud_only=True)
    
    # Scale data
    X_scaled = data_loader.scale_data(X_fraud)
    
    print(f"\nFraud cases loaded: {X_scaled.shape[0]}")
    print(f"Number of features: {X_scaled.shape[1]}")
    
    # 2. Prepare DataLoader
    X_tensor = torch.FloatTensor(X_scaled)
    dataset = TensorDataset(X_tensor)
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    # 3. Initialize model
    print("\n" + "="*60)
    print("Step 2: Initializing VAE Model")
    print("="*60)
    
    input_dim = X_scaled.shape[1]
    model = TabularVAE(
        input_dim=input_dim,
        hidden_dims=CONFIG['hidden_dims'],
        latent_dim=CONFIG['latent_dim'],
        dropout_rate=CONFIG['dropout_rate']
    ).to(device)
    
    print(f"\nModel architecture:")
    print(f"Input dimension: {input_dim}")
    print(f"Hidden dimensions: {CONFIG['hidden_dims']}")
    print(f"Latent dimension: {CONFIG['latent_dim']}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 4. Train model
    print("\n" + "="*60)
    print("Step 3: Training VAE")
    print("="*60)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    
    loss_history = train_vae(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        epochs=CONFIG['epochs'],
        beta_schedule=CONFIG['beta_schedule'],
        beta_start=CONFIG['beta_start'],
        beta_end=CONFIG['beta_end'],
        device=device
    )
    
    # Plot training history
    plot_training_history(loss_history)
    
    # 5. Generate synthetic data
    print("\n" + "="*60)
    print("Step 4: Generating Synthetic Fraud Data")
    print("="*60)
    
    print(f"\nGenerating {CONFIG['n_synthetic_samples']} synthetic fraud samples...")
    
    synthetic_scaled = generate_synthetic_fraud(
        model=model,
        n_samples=CONFIG['n_synthetic_samples'],
        input_dim=input_dim,
        temperature=CONFIG['temperature'],
        diversity_boost=CONFIG['diversity_boost'],
        device=device,
        data_loader=dataloader
    )
    
    # Inverse transform to original scale
    synthetic_data = data_loader.inverse_scale(synthetic_scaled)
    
    print(f"Generated {synthetic_data.shape[0]} synthetic samples")
    print(f"Shape: {synthetic_data.shape}")
    
    # 6. Evaluate quality
    print("\n" + "="*60)
    print("Step 5: Evaluating Synthetic Data Quality")
    print("="*60)
    
    metrics = evaluate_synthetic_quality(X_fraud, synthetic_data)
    
    # 7. Compare distributions
    compare_distributions(
        real_data=X_fraud,
        synthetic_data=synthetic_data,
        feature_names=data_loader.feature_names,
        n_features_to_plot=6
    )
    
    # 8. Save synthetic data
    print("\n" + "="*60)
    print("Step 6: Saving Synthetic Data")
    print("="*60)
    
    # Create DataFrame with synthetic data
    if data_loader.feature_names:
        synthetic_df = pd.DataFrame(synthetic_data, columns=data_loader.feature_names)
    else:
        synthetic_df = pd.DataFrame(synthetic_data)
    
    # Add isFraud column (all synthetic samples are fraud)
    synthetic_df['isFraud'] = 1
    
    # Save to CSV
    output_path = 'data/synthetic/vae_synthetic_fraud_data.csv'
    synthetic_df.to_csv(output_path, index=False)
    print(f"\nSynthetic fraud data saved to: {output_path}")
    
    # Also save as numpy array for easy loading
    np.save('data/synthetic/vae_synthetic_fraud_data.npy', synthetic_data)
    print(f"Synthetic fraud data (numpy) saved to: data/synthetic/vae_synthetic_fraud_data.npy")
    
    # 9. Save model if requested
    if CONFIG['save_model']:
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': CONFIG,
            'input_dim': input_dim,
            'feature_names': data_loader.feature_names,
            'scaler': data_loader.scaler
        }, CONFIG['model_path'])
        print(f"\nVAE model saved to: {CONFIG['model_path']}")
    
    # 10. Summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"\nReal Fraud Data:")
    print(f"  Number of samples: {X_fraud.shape[0]}")
    print(f"  Mean: {np.mean(X_fraud):.4f}")
    print(f"  Std: {np.std(X_fraud):.4f}")
    print(f"  Min: {np.min(X_fraud):.4f}")
    print(f"  Max: {np.max(X_fraud):.4f}")
    
    print(f"\nSynthetic Fraud Data:")
    print(f"  Number of samples: {synthetic_data.shape[0]}")
    print(f"  Mean: {np.mean(synthetic_data):.4f}")
    print(f"  Std: {np.std(synthetic_data):.4f}")
    print(f"  Min: {np.min(synthetic_data):.4f}")
    print(f"  Max: {np.max(synthetic_data):.4f}")
    
    print("\n" + "="*60)
    print("VAE-Based Synthetic Fraud Generation Complete!")
    print("="*60)
    
    return model, synthetic_data, synthetic_df, metrics


if __name__ == "__main__":
    model, synthetic_data, synthetic_df, metrics = main()

