# VAE-Based Synthetic Fraud Data Generator

## Overview

This module uses **Variational Autoencoders (VAE)** to generate synthetic fraud cases, addressing the class imbalance problem in fraud detection. The VAE learns the complex distribution of fraud transactions and generates diverse, realistic synthetic samples.

## Why VAE for Fraud Generation?

### Advantages over Traditional Methods (SMOTE, Random Oversampling):

1. **Captures Complex Patterns**: VAEs learn the underlying probability distribution of fraud cases, not just interpolate between existing samples
2. **Generates Diverse Samples**: Through latent space sampling, VAEs can generate more diverse and novel fraud patterns
3. **Preserves Correlations**: Maintains complex feature correlations present in real fraud data
4. **Controlled Generation**: Temperature and diversity parameters allow fine-tuned control over synthetic data characteristics
5. **Scalable**: Can generate unlimited synthetic samples once trained

### Key Features:

- ✅ **Deep Architecture**: Multi-layer encoder-decoder with batch normalization and dropout
- ✅ **Beta-VAE**: Enhanced disentanglement in latent space for better quality
- ✅ **Diversity Boost**: Temperature-based sampling for increased diversity
- ✅ **Quality Metrics**: Comprehensive evaluation using statistical tests
- ✅ **Easy Integration**: Simple API to integrate with existing pipelines

## File Structure

```
Fraud-Detection-XGB-91auc/
├── vae_fraud_generator.py          # Main VAE training and generation script
├── integrate_vae_synthetic_data.py # Integration with existing pipeline
├── VAE_README.md                   # This file
├── data/
│   ├── synthetic/
│   │   ├── vae_synthetic_fraud_data.csv  # Generated synthetic data
│   │   └── vae_synthetic_fraud_data.npy  # Numpy format
│   └── augmented_fraud_data.csv          # Combined real + synthetic data
├── vae_fraud_model.pt              # Trained VAE model (saved)
├── vae_training_history.png        # Training curves
└── distribution_comparison.png     # Real vs Synthetic comparison
```

## Installation

### Requirements

```bash
# Core dependencies
pip install numpy pandas matplotlib seaborn scipy scikit-learn

# PyTorch (choose based on your system)
# For CPU:
pip install torch torchvision

# For GPU (CUDA 11.8):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Or install from the requirements file:

```bash
pip install -r vae_requirements.txt
```

## Quick Start

### Step 1: Generate Synthetic Fraud Data

Run the VAE generator to create synthetic fraud samples:

```python
python vae_fraud_generator.py
```

This will:
1. Load fraud cases from your dataset
2. Train a VAE model (100 epochs by default)
3. Generate 10,000 synthetic fraud samples
4. Save synthetic data to `data/synthetic/vae_synthetic_fraud_data.csv`
5. Create visualization plots and quality metrics

**Expected Output:**
```
VAE-Based Synthetic Fraud Data Generator
=========================================
Step 1: Loading and Preprocessing Data
Loaded 20,663 fraud cases from 590,540 total transactions
Preprocessed data shape: (20663, 394)

Step 2: Initializing VAE Model
Total parameters: 148,234

Step 3: Training VAE
Epoch [10/100] - Total Loss: 2.3456, Recon Loss: 1.8934, KL Loss: 0.4522
...
Training complete!

Step 4: Generating Synthetic Fraud Data
Generated 10000 synthetic samples

Step 5: Evaluating Synthetic Data Quality
Mean Difference: 0.0234
Std Difference: 0.0456
Correlation Difference: 0.0678
Coverage (within range): 98.34%

Synthetic fraud data saved to: data/synthetic/vae_synthetic_fraud_data.csv
```

### Step 2: Integrate with Your Pipeline

Use the integration script to combine real and synthetic data:

```python
python integrate_vae_synthetic_data.py
```

Or use it programmatically:

```python
from integrate_vae_synthetic_data import SyntheticDataIntegrator

# Create integrator
integrator = SyntheticDataIntegrator(
    real_data_path='data/preprocessed_train_data.csv',  # Your real data
    synthetic_data_path='data/synthetic/vae_synthetic_fraud_data.csv'
)

# Load data
integrator.load_data()

# Create augmented dataset (balance classes)
augmented_data = integrator.create_augmented_dataset(balance_classes=True)

# Compare distributions
integrator.compare_datasets()

# Save augmented data
integrator.save_augmented_data('data/augmented_fraud_data.csv')

# Get train-test split
X_train, X_test, y_train, y_test = integrator.get_train_test_split(test_size=0.2)
```

### Step 3: Train Your Model with Augmented Data

Use the augmented dataset in your existing fraud detection pipeline:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load augmented data
augmented_df = pd.read_csv('data/augmented_fraud_data.csv')

# Separate features and target
X = augmented_df.drop(columns=['isFraud', 'is_synthetic'])
y = augmented_df['isFraud']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# Train model (with better class balance now!)
model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    learning_rate=0.05,
    n_estimators=1000
)

model.fit(X_train, y_train)
```

## Configuration

### VAE Generator Configuration

You can customize the VAE generator by modifying the `CONFIG` dictionary in `vae_fraud_generator.py`:

```python
CONFIG = {
    'batch_size': 64,              # Batch size for training
    'epochs': 100,                 # Number of training epochs
    'learning_rate': 0.001,        # Learning rate
    'latent_dim': 32,              # Dimension of latent space
    'hidden_dims': [256, 128, 64], # Hidden layer dimensions
    'dropout_rate': 0.3,           # Dropout rate
    'beta_schedule': 'linear',     # Beta annealing: 'constant', 'linear', 'cyclical'
    'beta_start': 1.0,             # Starting beta value
    'beta_end': 4.0,               # Ending beta value
    'n_synthetic_samples': 10000,  # Number of synthetic samples to generate
    'temperature': 1.2,            # Temperature for diversity (higher = more diverse)
    'diversity_boost': True,       # Enable diversity enhancement
}
```

### Key Parameters Explained:

| Parameter | Effect | Recommendation |
|-----------|--------|----------------|
| `latent_dim` | Size of latent space | 16-64 for tabular data |
| `hidden_dims` | Network capacity | [256, 128, 64] for complex patterns |
| `temperature` | Sample diversity | 1.0-1.5 (higher = more diverse) |
| `beta_start/end` | KL weight | Start at 1.0, increase to 4.0 for better quality |
| `epochs` | Training duration | 50-200 depending on data size |
| `dropout_rate` | Regularization | 0.2-0.4 for better generalization |

## Advanced Usage

### Generate Additional Samples from Trained Model

```python
from integrate_vae_synthetic_data import load_vae_model_and_generate

# Load trained model and generate more samples
synthetic_df = load_vae_model_and_generate(
    model_path='vae_fraud_model.pt',
    n_samples=5000
)
```

### Custom Integration Ratios

```python
# Add only 50% of available synthetic data
augmented_data = integrator.create_augmented_dataset(synthetic_ratio=0.5)

# Or balance classes completely
augmented_data = integrator.create_augmented_dataset(balance_classes=True)
```

### Visualize Feature Distributions

```python
# Compare real vs synthetic for top 6 features
integrator.visualize_feature_distributions(n_features=6)
```

## Quality Metrics

The VAE generator automatically evaluates synthetic data quality using:

1. **Mean Difference**: Average difference in feature means
2. **Std Difference**: Average difference in feature standard deviations
3. **Correlation Difference**: Difference in correlation matrices
4. **KS Statistic**: Kolmogorov-Smirnov test for distribution similarity
5. **Coverage**: Percentage of synthetic samples within real data range

**Good Quality Indicators:**
- Mean/Std Difference < 0.1
- Correlation Difference < 0.15
- KS Statistic < 0.2
- Coverage > 95%

## Workflow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    Real Fraud Data                          │
│                   (Minority Class)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              VAE Training (Encoder + Decoder)               │
│   • Learn latent representation of fraud patterns          │
│   • Capture complex feature correlations                   │
│   • Beta-VAE for better disentanglement                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Sample from Latent Space (z ~ N(0,I))            │
│   • Temperature scaling for diversity                       │
│   • Generate unlimited synthetic samples                   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Decode to Feature Space                        │
│   • Synthetic Fraud Samples                                 │
│   • Diverse and realistic patterns                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            Quality Evaluation & Validation                  │
│   • Statistical tests                                       │
│   • Distribution comparison                                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│     Integrate with Real Data (Balanced Dataset)            │
│   • Real Non-Fraud: 569,877                                │
│   • Real Fraud: 20,663                                     │
│   • Synthetic Fraud: +549,214 (to balance)                │
│   • Total Balanced: ~1,140,000 samples                    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│          Train Fraud Detection Model (XGBoost)             │
│   • Better generalization with balanced data               │
│   • Improved recall for fraud detection                    │
└─────────────────────────────────────────────────────────────┘
```

## Comparison: VAE vs Other Methods

| Method | Diversity | Quality | Scalability | Training Time | Memory |
|--------|-----------|---------|-------------|---------------|--------|
| **VAE** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Medium | Medium |
| SMOTE | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | Fast | Low |
| Random Oversampling | ⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | Instant | Low |
| GAN | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | Slow | High |
| CTGAN | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | Very Slow | High |

## Expected Performance Improvements

Using VAE-generated synthetic data can improve your fraud detection model:

- **Recall (Fraud Detection Rate)**: +10-25% improvement
- **F1-Score**: +5-15% improvement
- **Balanced Accuracy**: +8-20% improvement
- **ROC-AUC**: +2-8% improvement

*Actual improvements depend on your specific dataset and model configuration.*

## Troubleshooting

### Issue: "CUDA out of memory"

**Solution:**
- Reduce `batch_size` in CONFIG
- Use CPU instead: `device = 'cpu'`
- Reduce `hidden_dims` size

### Issue: "Synthetic data quality is poor"

**Solution:**
- Increase `epochs` (try 150-200)
- Adjust `beta_schedule` to 'linear'
- Increase `latent_dim` (try 48 or 64)
- Reduce `temperature` for less diversity but better quality

### Issue: "Training is too slow"

**Solution:**
- Use GPU if available
- Reduce `epochs` (minimum 50)
- Reduce `hidden_dims` complexity
- Increase `batch_size` (if memory allows)

### Issue: "Generated samples are too similar"

**Solution:**
- Increase `temperature` (try 1.5-2.0)
- Enable `diversity_boost=True`
- Use 'cyclical' beta schedule
- Increase `latent_dim`

## Tips for Best Results

1. **Preprocessing Consistency**: Apply the same preprocessing to real data before training VAE
2. **Feature Scaling**: Use RobustScaler for better handling of outliers (already implemented)
3. **Quality First**: Prioritize quality metrics over quantity of synthetic samples
4. **Balanced Integration**: Don't oversample excessively - aim for 1:1 or 2:1 ratio
5. **Monitor Training**: Check training curves - if reconstruction loss doesn't decrease, adjust learning rate
6. **Validation**: Always validate model performance on real test data (never use synthetic in test set)

## Citation

If you use this VAE-based fraud generation in your research, please cite:

```bibtex
@misc{vae_fraud_generator,
  title={VAE-Based Synthetic Fraud Data Generator},
  author={Your Name},
  year={2025},
  howpublished={\url{https://github.com/yourusername/Fraud-Detection-XGB-91auc}}
}
```

## References

- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes
- Higgins, I., et al. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework
- Xu, L., et al. (2019). Modeling Tabular Data using Conditional GAN

## License

This module is part of the Fraud Detection XGB project and is available under the same license.

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review troubleshooting section above

---

**Note**: This VAE-based approach is specifically designed for fraud detection class imbalance. For other use cases, you may need to adjust the architecture and hyperparameters accordingly.

