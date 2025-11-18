# VAE Fraud Generation - Quick Start Guide

## 🚀 Get Started in 3 Steps

### Step 1: Install Dependencies

```bash
pip install -r vae_requirements.txt
```

### Step 2: Generate Synthetic Fraud Data

```bash
python vae_fraud_generator.py
```

**Output:**
- `data/synthetic/vae_synthetic_fraud_data.csv` - 10,000 synthetic fraud samples
- `vae_fraud_model.pt` - Trained VAE model (reusable)
- `vae_training_history.png` - Training curves
- `distribution_comparison.png` - Quality visualization

**Time:** 5-15 minutes (depends on CPU/GPU)

### Step 3: Integrate with Your Pipeline

```bash
python integrate_vae_synthetic_data.py
```

Or use in your code:

```python
from integrate_vae_synthetic_data import SyntheticDataIntegrator

# Load data
integrator = SyntheticDataIntegrator(
    real_data_path='your_data.csv',
    synthetic_data_path='data/synthetic/vae_synthetic_fraud_data.csv'
)
integrator.load_data()

# Create balanced dataset
augmented = integrator.create_augmented_dataset(balance_classes=True)

# Save
integrator.save_augmented_data('data/augmented_fraud_data.csv')
```

## 📊 Use in Your Existing Model

Replace your current training data:

```python
# Old way (imbalanced)
train_df = pd.read_csv('data/preprocessed_train_data.csv')

# New way (balanced with synthetic data)
train_df = pd.read_csv('data/augmented_fraud_data.csv')
train_df = train_df.drop(columns=['is_synthetic'])  # Remove marker

# Continue as normal
X_train = train_df.drop(columns=['isFraud'])
y_train = train_df['isFraud']
```

## 🎛️ Quick Configuration

Edit `CONFIG` in `vae_fraud_generator.py`:

```python
CONFIG = {
    'n_synthetic_samples': 10000,  # ← Change this to generate more/less
    'temperature': 1.2,            # ← Higher = more diverse (1.0-2.0)
    'epochs': 100,                 # ← More epochs = better quality
}
```

## 🎯 Expected Results

**Before VAE (Imbalanced):**
- Fraud: 20,663 (3.5%)
- Non-fraud: 569,877 (96.5%)
- Model recall: ~60-70%

**After VAE (Balanced):**
- Fraud: 570,000 (50%)
- Non-fraud: 569,877 (50%)
- Model recall: ~75-85% ✨

## 💡 Pro Tips

1. **Start Small**: Generate 5K samples first, test, then scale up
2. **Quality Check**: Look at `distribution_comparison.png` - synthetic should match real
3. **Don't Overdo**: 1:1 ratio (fraud:non-fraud) is usually optimal
4. **Validate on Real**: Always test on real held-out data
5. **Reuse Model**: Once trained, generate more samples instantly

## 🔍 Verify Quality

Good synthetic data should have:
- ✅ Mean Difference < 0.1
- ✅ Correlation Difference < 0.15
- ✅ Coverage > 95%

Check these in the output logs!

## 🆘 Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `batch_size` to 32 |
| Poor quality | Increase `epochs` to 150-200 |
| Too slow | Reduce `epochs` to 50, use GPU |
| Too similar samples | Increase `temperature` to 1.5 |

## 📚 Learn More

- Full documentation: `VAE_README.md`
- Example notebook: `vae_fraud_example.ipynb`
- Integration: `integrate_vae_synthetic_data.py`

## ✅ Checklist

- [ ] Install dependencies
- [ ] Run `vae_fraud_generator.py`
- [ ] Check quality metrics (console output)
- [ ] Review `distribution_comparison.png`
- [ ] Integrate with your data
- [ ] Train model with augmented data
- [ ] Compare performance (before/after)

---

**Questions?** Check `VAE_README.md` for detailed documentation.

**Need help?** Review the example notebook: `vae_fraud_example.ipynb`

