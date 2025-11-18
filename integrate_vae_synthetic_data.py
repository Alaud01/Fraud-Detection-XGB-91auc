"""
Integration Script: VAE Synthetic Data with Fraud Detection Pipeline
=====================================================================
This script demonstrates how to integrate VAE-generated synthetic fraud data
with the existing fraud detection pipeline to address class imbalance.

Usage:
------
1. First run vae_fraud_generator.py to generate synthetic fraud data
2. Then run this script to combine real and synthetic data
3. Use the augmented dataset for training your fraud detection model
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


class SyntheticDataIntegrator:
    """
    Integrate VAE-generated synthetic fraud data with real data
    """
    
    def __init__(self, real_data_path=None, synthetic_data_path='data/synthetic/vae_synthetic_fraud_data.csv'):
        """
        Initialize the integrator
        
        Parameters:
        -----------
        real_data_path : str
            Path to real fraud detection data
        synthetic_data_path : str
            Path to VAE-generated synthetic data
        """
        self.real_data_path = real_data_path
        self.synthetic_data_path = synthetic_data_path
        self.real_data = None
        self.synthetic_data = None
        self.augmented_data = None
        
    def load_data(self):
        """Load real and synthetic data"""
        
        print("Loading data...")
        
        # Load synthetic data
        try:
            self.synthetic_data = pd.read_csv(self.synthetic_data_path)
            print(f"Loaded {len(self.synthetic_data)} synthetic fraud samples")
        except FileNotFoundError:
            print(f"Synthetic data not found at {self.synthetic_data_path}")
            print("Please run vae_fraud_generator.py first!")
            return False
        
        # If real data path is provided, load it
        if self.real_data_path:
            try:
                self.real_data = pd.read_csv(self.real_data_path)
                print(f"Loaded {len(self.real_data)} real samples")
            except FileNotFoundError:
                print(f"Real data not found at {self.real_data_path}")
                return False
        
        return True
    
    def analyze_class_distribution(self, data, title="Class Distribution"):
        """
        Analyze and visualize class distribution
        
        Parameters:
        -----------
        data : pd.DataFrame
            Data to analyze
        title : str
            Title for the plot
        """
        if 'isFraud' not in data.columns:
            print("No 'isFraud' column found in data")
            return
        
        fraud_count = data['isFraud'].sum()
        non_fraud_count = len(data) - fraud_count
        
        print(f"\n{title}")
        print("-" * 50)
        print(f"Total samples: {len(data)}")
        print(f"Fraud cases: {fraud_count} ({fraud_count/len(data)*100:.2f}%)")
        print(f"Non-fraud cases: {non_fraud_count} ({non_fraud_count/len(data)*100:.2f}%)")
        print(f"Fraud ratio: 1:{non_fraud_count/fraud_count:.2f}")
        
        # Visualize
        plt.figure(figsize=(10, 6))
        
        # Count plot
        ax = sns.countplot(x='isFraud', data=data, palette=['green', 'red'])
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('Class (0=Non-Fraud, 1=Fraud)', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Add count labels on bars
        for p in ax.patches:
            height = p.get_height()
            ax.text(p.get_x() + p.get_width()/2., height,
                   f'{int(height)}\n({height/len(data)*100:.1f}%)',
                   ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def create_augmented_dataset(self, synthetic_ratio=1.0, balance_classes=False):
        """
        Create augmented dataset by combining real and synthetic data
        
        Parameters:
        -----------
        synthetic_ratio : float
            Ratio of synthetic samples to use (0 to 1)
            e.g., 0.5 = use 50% of available synthetic samples
        balance_classes : bool
            If True, add enough synthetic samples to balance classes
        
        Returns:
        --------
        augmented_data : pd.DataFrame
            Combined dataset with real and synthetic data
        """
        
        if self.real_data is None:
            print("Real data not loaded. Using synthetic data only.")
            return self.synthetic_data
        
        print("\nCreating augmented dataset...")
        
        # Get fraud and non-fraud counts from real data
        fraud_count = self.real_data['isFraud'].sum()
        non_fraud_count = len(self.real_data) - fraud_count
        
        # Determine how many synthetic samples to add
        if balance_classes:
            # Add enough synthetic samples to balance classes
            n_synthetic_to_add = max(0, non_fraud_count - fraud_count)
            print(f"Balancing classes: adding {n_synthetic_to_add} synthetic fraud samples")
        else:
            # Use the specified ratio
            n_synthetic_to_add = int(len(self.synthetic_data) * synthetic_ratio)
            print(f"Using {synthetic_ratio*100:.0f}% of synthetic data: {n_synthetic_to_add} samples")
        
        # Sample synthetic data
        if n_synthetic_to_add > 0:
            if n_synthetic_to_add <= len(self.synthetic_data):
                synthetic_subset = self.synthetic_data.sample(n=n_synthetic_to_add, random_state=42)
            else:
                # If we need more than available, sample with replacement
                print(f"Warning: Requested {n_synthetic_to_add} samples but only {len(self.synthetic_data)} available.")
                print(f"Sampling with replacement...")
                synthetic_subset = self.synthetic_data.sample(n=n_synthetic_to_add, replace=True, random_state=42)
            
            # Add marker column to identify synthetic samples
            synthetic_subset = synthetic_subset.copy()
            synthetic_subset['is_synthetic'] = 1
            
            # Add marker to real data
            real_data_marked = self.real_data.copy()
            real_data_marked['is_synthetic'] = 0
            
            # Combine datasets
            self.augmented_data = pd.concat([real_data_marked, synthetic_subset], ignore_index=True)
            
            print(f"\nAugmented dataset created:")
            print(f"  Real samples: {len(self.real_data)}")
            print(f"  Synthetic samples added: {len(synthetic_subset)}")
            print(f"  Total samples: {len(self.augmented_data)}")
        else:
            self.augmented_data = self.real_data.copy()
            self.augmented_data['is_synthetic'] = 0
            print("No synthetic samples added")
        
        return self.augmented_data
    
    def compare_datasets(self):
        """
        Compare real data, synthetic data, and augmented data distributions
        """
        
        if self.real_data is None:
            print("Real data not available for comparison")
            return
        
        print("\n" + "="*70)
        print("Dataset Comparison")
        print("="*70)
        
        # Analyze each dataset
        if self.real_data is not None:
            self.analyze_class_distribution(self.real_data, "Real Data Distribution")
        
        if self.augmented_data is not None:
            self.analyze_class_distribution(self.augmented_data, "Augmented Data Distribution")
    
    def visualize_feature_distributions(self, n_features=6):
        """
        Visualize feature distributions comparing real and synthetic data
        
        Parameters:
        -----------
        n_features : int
            Number of features to visualize
        """
        
        if self.augmented_data is None:
            print("Augmented data not created yet. Call create_augmented_dataset() first.")
            return
        
        # Exclude non-feature columns
        exclude_cols = ['isFraud', 'is_synthetic', 'TransactionID']
        feature_cols = [col for col in self.augmented_data.columns if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            print("No feature columns found")
            return
        
        # Select features with highest variance
        feature_data = self.augmented_data[feature_cols]
        feature_vars = feature_data.var()
        top_features = feature_vars.nlargest(n_features).index.tolist()
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features[:n_features]):
            ax = axes[i]
            
            # Get data
            real_feature = self.augmented_data[self.augmented_data['is_synthetic'] == 0][feature]
            synthetic_feature = self.augmented_data[self.augmented_data['is_synthetic'] == 1][feature]
            
            # Plot
            ax.hist(real_feature, bins=50, alpha=0.5, label='Real', color='blue', density=True)
            ax.hist(synthetic_feature, bins=50, alpha=0.5, label='Synthetic', color='red', density=True)
            
            ax.set_title(feature, fontsize=12, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Density')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('feature_comparison_real_vs_synthetic.png', dpi=300, bbox_inches='tight')
        print("\nFeature comparison plot saved to: feature_comparison_real_vs_synthetic.png")
        plt.show()
    
    def save_augmented_data(self, output_path='data/augmented_fraud_data.csv'):
        """
        Save augmented dataset to CSV
        
        Parameters:
        -----------
        output_path : str
            Path to save the augmented data
        """
        
        if self.augmented_data is None:
            print("No augmented data to save. Call create_augmented_dataset() first.")
            return
        
        self.augmented_data.to_csv(output_path, index=False)
        print(f"\nAugmented dataset saved to: {output_path}")
        print(f"Shape: {self.augmented_data.shape}")
        
        return output_path
    
    def get_train_test_split(self, test_size=0.2, stratify=True, remove_synthetic_marker=False):
        """
        Get train-test split from augmented data
        
        Parameters:
        -----------
        test_size : float
            Proportion of data for testing
        stratify : bool
            Whether to stratify by fraud label
        remove_synthetic_marker : bool
            Whether to remove the 'is_synthetic' column
        
        Returns:
        --------
        X_train, X_test, y_train, y_test : tuple
            Train-test split
        """
        
        if self.augmented_data is None:
            print("No augmented data available. Call create_augmented_dataset() first.")
            return None
        
        # Prepare features and target
        y = self.augmented_data['isFraud']
        X = self.augmented_data.drop(columns=['isFraud'])
        
        if remove_synthetic_marker and 'is_synthetic' in X.columns:
            X = X.drop(columns=['is_synthetic'])
        
        # Split data
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, stratify=y, random_state=42
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42
            )
        
        print(f"\nTrain-Test Split:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Testing samples: {len(X_test)}")
        print(f"  Training fraud ratio: {y_train.sum()}/{len(y_train)} ({y_train.sum()/len(y_train)*100:.2f}%)")
        print(f"  Testing fraud ratio: {y_test.sum()}/{len(y_test)} ({y_test.sum()/len(y_test)*100:.2f}%)")
        
        return X_train, X_test, y_train, y_test


def load_vae_model_and_generate(model_path='vae_fraud_model.pt', n_samples=5000):
    """
    Load a pre-trained VAE model and generate additional synthetic samples
    
    Parameters:
    -----------
    model_path : str
        Path to the saved VAE model
    n_samples : int
        Number of samples to generate
    
    Returns:
    --------
    synthetic_data : pd.DataFrame
        Generated synthetic data
    """
    import torch
    from vae_fraud_generator import TabularVAE, generate_synthetic_fraud
    
    print(f"Loading VAE model from {model_path}...")
    
    # Load model
    checkpoint = torch.load(model_path)
    
    # Reconstruct model
    model = TabularVAE(
        input_dim=checkpoint['input_dim'],
        hidden_dims=checkpoint['config']['hidden_dims'],
        latent_dim=checkpoint['config']['latent_dim'],
        dropout_rate=checkpoint['config']['dropout_rate']
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    print(f"Generating {n_samples} new synthetic samples...")
    
    # Generate synthetic data
    synthetic_scaled = generate_synthetic_fraud(
        model=model,
        n_samples=n_samples,
        input_dim=checkpoint['input_dim'],
        temperature=1.2,
        diversity_boost=True,
        device=device
    )
    
    # Inverse transform
    scaler = checkpoint['scaler']
    synthetic_data = scaler.inverse_transform(synthetic_scaled)
    
    # Create DataFrame
    if checkpoint['feature_names']:
        synthetic_df = pd.DataFrame(synthetic_data, columns=checkpoint['feature_names'])
    else:
        synthetic_df = pd.DataFrame(synthetic_data)
    
    synthetic_df['isFraud'] = 1
    
    print(f"Generated {len(synthetic_df)} synthetic fraud samples")
    
    return synthetic_df


def example_usage():
    """
    Example of how to use the integrator
    """
    
    print("="*70)
    print("VAE Synthetic Data Integration Example")
    print("="*70)
    
    # Option 1: Use with preprocessed data from your pipeline
    # After running your preprocessing in detect_fraud.ipynb, save the data:
    # train_df.to_csv('data/preprocessed_train_data.csv', index=False)
    
    # Then integrate synthetic data
    integrator = SyntheticDataIntegrator(
        real_data_path=None,  # Set this if you have preprocessed real data
        synthetic_data_path='data/synthetic/vae_synthetic_fraud_data.csv'
    )
    
    # Load data
    if not integrator.load_data():
        print("\nFailed to load data. Please ensure:")
        print("1. You have run vae_fraud_generator.py to create synthetic data")
        print("2. The file paths are correct")
        return
    
    # If you only have synthetic data (for demonstration)
    if integrator.real_data is None:
        print("\n" + "="*70)
        print("Synthetic Data Summary")
        print("="*70)
        print(f"Synthetic samples available: {len(integrator.synthetic_data)}")
        print(f"Features: {len(integrator.synthetic_data.columns) - 1}")  # -1 for isFraud column
        
        # Show some statistics
        print("\nSynthetic Data Statistics:")
        print(integrator.synthetic_data.describe())
        
        return integrator.synthetic_data
    
    # Create augmented dataset
    # Option A: Balance classes
    augmented_data = integrator.create_augmented_dataset(balance_classes=True)
    
    # Option B: Use specific ratio of synthetic data
    # augmented_data = integrator.create_augmented_dataset(synthetic_ratio=0.5)
    
    # Compare distributions
    integrator.compare_datasets()
    
    # Visualize feature distributions
    integrator.visualize_feature_distributions(n_features=6)
    
    # Save augmented data
    integrator.save_augmented_data('data/augmented_fraud_data.csv')
    
    # Get train-test split ready for modeling
    X_train, X_test, y_train, y_test = integrator.get_train_test_split(
        test_size=0.2,
        stratify=True,
        remove_synthetic_marker=True
    )
    
    print("\n" + "="*70)
    print("Ready for Model Training!")
    print("="*70)
    print("\nYou can now use X_train, X_test, y_train, y_test for training")
    print("Or load the saved augmented data: data/augmented_fraud_data.csv")
    
    return integrator, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # Run example
    result = example_usage()
    
    print("\n" + "="*70)
    print("Integration Complete!")
    print("="*70)
    print("\nNext steps:")
    print("1. Use the augmented dataset to train your XGBoost model")
    print("2. Compare model performance with and without synthetic data")
    print("3. Adjust synthetic_ratio or use balance_classes as needed")
    print("\nTo generate more synthetic samples, you can:")
    print("- Adjust n_synthetic_samples in vae_fraud_generator.py")
    print("- Or use load_vae_model_and_generate() function")

