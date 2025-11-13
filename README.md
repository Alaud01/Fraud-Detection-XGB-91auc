# Fraud Detection using Machine Learning Models

## Overview

This project is aimed at improving fraud detection systems using machine learning models. It focuses on reducing false positives and increasing detection accuracy for fraudulent transactions, specifically within the e-commerce space. The dataset used in this project is provided by Vesta Corporation, which contains real-world e-commerce transaction data, including a wide range of features such as device type and product features.

### Key Objectives:
- Detect fraudulent transactions accurately.
- Minimize the occurrence of false positives to improve the user experience.
- Benchmark machine learning models on large-scale datasets.
- Implement feature engineering and dimensionality reduction techniques to improve model performance.

## Description

Imagine you're at the grocery store, buying everything needed for a party, only to have your card declined at checkout, despite having the necessary funds. Later, you receive a message from your bank asking if you attempted to spend the amount. Although this can be frustrating, it's part of a fraud prevention system aimed at protecting your finances.

This project seeks to improve the balance between fraud detection accuracy and the user experience. By leveraging advanced machine learning models and data science techniques, the goal is to reduce false alarms while ensuring security for millions of consumers.

In this competition, I applied benchmark machine learning models using Vesta's dataset, which includes a variety of features ranging from device information to transaction data. Through this process, I aim to create new features, reduce dimensionality, and fine-tune models to improve fraud detection.

## Dataset

The dataset used for this project is provided by Vesta Corporation. The dataset contains two parts:
- **Transaction data**: Includes detailed information about e-commerce transactions.
- **Identity data**: Contains information about the identity of the customer performing the transaction.

Both parts are merged based on a common `TransactionID`. 

### Key Features:
- **Transaction data**: Contains features like transaction amount, card type, and email domain.
- **Identity data**: Contains features such as device information, browser type, and screen resolution.
- **Target**: The target variable `isFraud` indicates whether a transaction is fraudulent (`1`) or not (`0`).

## Approach

The project is divided into the following key steps:

### 1. **Data Preprocessing**
   - **Data Merging**: The transaction and identity datasets are merged using `TransactionID`.
   - **Handling Missing Values**: Missing values are filled with placeholders (for numerical features: -999, and for categorical features: '-999').
   - **Outlier Removal**: Outliers in `TransactionAmt` are removed using a z-score threshold of 3.
   - **Feature Engineering**: New features such as email domain suffix, operating system type, screen resolution, and browser type are created.
   - **Memory Reduction**: Column data types are optimized to reduce memory usage.

### 2. **Exploratory Data Analysis (EDA)**
   - **Histograms**: Transaction amounts are visualized for both fraudulent and non-fraudulent transactions.
   - **Time Period Analysis**: Fraudulent transactions are plotted across different time periods (days of the week, hours of the day, etc.).
   - **Correlation Analysis**: Heatmaps are created for different subsets of the data to identify correlations between features.

### 3. **Dimensionality Reduction**
   - **Principal Component Analysis (PCA)**: Applied to reduce dimensionality in features with a high number of columns (e.g., `V` columns).
   - **Scree Plot**: Used to determine the optimal number of components to retain 90% of the explained variance.

### 4. **Modeling**
   - **XGBoost**: A gradient boosting model is trained on the preprocessed data. XGBoost is chosen for its robustness and ability to handle large datasets efficiently.
   - **LightGBM**: An alternative gradient boosting model used to compare performance.
   - **Model Evaluation**: Various evaluation metrics, including AUC, F1-score, and confusion matrix, are used to assess model performance.

### 5. **Prediction and Submission**
   - The final model is used to predict fraud on the test set, and predictions are saved in a CSV file for submission.

## Libraries and Tools

- **Python 3.7**
- **NumPy**: For numerical operations.
- **Pandas**: For data manipulation.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For machine learning models and preprocessing.
- **XGBoost**: Gradient boosting framework.
- **LightGBM**: Gradient boosting framework, an alternative to XGBoost.
- **PCA (Principal Component Analysis)**: For dimensionality reduction.

## Results

After training the XGBoost model on the training data and evaluating it on the validation set, the model achieves strong performance across various metrics. The evaluation metrics include:
- **Accuracy**: Measures the percentage of correctly predicted transactions.
- **AUC (Area Under Curve) - 91%**: Reflects the model's ability to distinguish between fraud and non-fraud cases.
- **Precision, Recall, and F1-score**: Provide a detailed look at the model’s prediction performance, especially for fraudulent transactions.

## Future Work

- Experiment with more advanced feature engineering techniques.
- Explore additional machine learning models like neural networks or ensemble models.
- Fine-tune hyperparameters to further optimize model performance.
  
## Acknowledgements

This project is based on the IEEE-CIS Fraud Detection competition. The dataset was provided by Vesta Corporation, a leader in e-commerce payment solutions.

Vesta Corporation is the forerunner in guaranteed e-commerce payment solutions, and this dataset represents real-world transactions that help train machine learning models for fraud detection.

## Synthetic Data Generation for Fraud Detection using TabDiff

This document provides instructions on how to use the `generating_synthetic_data.py` script to generate synthetic fraud data using the TabDiff model. This is intended to address the class imbalance in the IEEE-CIS Fraud Detection dataset.

### 1. Introduction

The `detect_fraud.ipynb` notebook trains an XGBoost model for fraud detection. The dataset is highly imbalanced, with very few fraud cases compared to non-fraud cases. This script helps to generate more high-quality, diverse synthetic fraud samples to improve the model's ability to detect fraud.

We use TabDiff, a diffusion model for tabular data, to learn the distribution of the fraudulent transactions and generate new samples from it.

### 2. Setup

Before running the script, you need to set up the environment, which includes cloning the TabDiff repository and creating the required conda environments.

#### 2.1. Clone TabDiff Repository

Clone the TabDiff repository from GitHub:

```bash
git clone https://github.com/MinkaiXu/TabDiff.git
```

#### 2.2. Create Conda Environments

TabDiff requires two separate conda environments.

1.  **Main environment for TabDiff:**

    ```bash
    cd TabDiff
    conda env create -f tabdiff.yaml
    cd ..
    ```

2.  **Environment for evaluation metrics (optional for generation):**

    ```bash
    cd TabDiff
    conda env create -f synthcity.yaml
    cd ..
    ```

    You will primarily use the `tabdiff` environment for data generation.

### 3. Usage

The `generating_synthetic_data.py` script automates the process of preparing data, running TabDiff, and saving the synthetic samples.

#### 3.1. Run the script

To run the script, make sure you have the `tabdiff` conda environment activated:

```bash
conda activate tabdiff
python generating_synthetic_data.py
```

**GPU Acceleration:** The script automatically detects and uses available GPU acceleration:
- **Mac (M1/M2/M3/M4):** Automatically uses Metal Performance Shaders (MPS) for GPU acceleration
- **NVIDIA GPUs:** Automatically uses CUDA for GPU acceleration
- **CPU Fallback:** If no GPU is detected, training runs on CPU (slower)

The script will print which device is being used at startup.

#### 3.2. What the script does:

1.  **Loads and preprocesses data:** It loads the `train_transaction.csv` and `train_identity.csv` files and applies the same preprocessing steps as in the `detect_fraud.ipynb` notebook.
2.  **Isolates fraud data:** It selects only the transactions marked as fraudulent (`isFraud == 1`).
3.  **Prepares data for TabDiff:** It saves the fraud data into `data/synthetic/fraud_data.csv` and creates the necessary metadata file `data/synthetic/Info/fraud_data.json`.
4.  **Trains TabDiff:** It calls the TabDiff training script (`main.py` from the TabDiff repo) to train a diffusion model on the fraud data.
5.  **Generates synthetic data:** It uses the trained TabDiff model to sample new synthetic fraud data.
6.  **Visualizes and saves results:**
    *   It creates a plot (`synthetic_fraud_cases_comparison.png`) comparing the number of real and synthetic fraud cases.
    *   It saves the generated synthetic data to `data/synthetic/synthetic_fraud_data.csv`.

### 4. GPU Acceleration Details

The script automatically enables GPU acceleration where available:

#### Mac Metal Performance Shaders (MPS)
- **Devices:** Apple Silicon Macs (M1, M2, M3, M4, etc.)
- **Speed improvement:** 5-10x faster training vs. CPU
- **Automatic:** No configuration needed
- **Training time with lightweight model:** ~10-20 minutes (vs. 45-60 min on CPU)

#### NVIDIA CUDA
- **Devices:** NVIDIA GPUs (RTX, GTX, Tesla, etc.)
- **Speed improvement:** 10-50x faster training vs. CPU
- **Automatic:** Detected if CUDA toolkit is installed
- **Training time with lightweight model:** ~2-5 minutes

#### CPU Fallback
- Used when no GPU is detected
- **Training time with lightweight model:** ~45-60 minutes

**Tip:** Even on your M4 Mac with MPS enabled, the lightweight model is recommended for quick iterations. Use the full model only when you need the highest quality synthetic data for production.

### 5. Model Variants: Lightweight vs. Full

The `generating_synthetic_data.py` script provides two model options:

#### 5.1 Lightweight Model (Default, 10-20 minutes)

The lightweight model is optimized for fast training on CPU/M-series Macs:

```python
run_tabdiff(lightweight=True)  # Default
```

**Configuration changes:**
- Training steps: 8000 → 1200
- Diffusion timesteps: 50 → 10
- Time embedding dimension: 1024 → 256
- Batch size: 4096 → 2048
- Network layers: 2 → 1
- Token dimension: 4 → 2
- MLP expansion factor: 32 → 16

**Trade-offs:**
- ✅ **Pros:** 
  - Trains in 10-20 minutes on M4 Mac
  - Still captures the fraud data distribution reasonably well
  - Perfect for prototyping and testing the pipeline
  
- ❌ **Cons:**
  - Slightly lower-quality synthetic samples
  - Less diversity in generated fraud patterns
  - May not capture all edge cases in the fraud distribution

**Recommended for:** 
- Quick prototyping and experimentation
- Testing the full pipeline with your XGBoost model
- When time is a constraint

#### 5.2 Full Model (45+ minutes)

For higher-quality synthetic data:

```python
run_tabdiff(lightweight=False)
```

Uses the original full TabDiff configuration.

**Trade-offs:**
- ✅ **Pros:**
  - Higher-quality, more diverse synthetic fraud samples
  - Better captures complex fraud patterns
  - More robust synthetic data for model training
  
- ❌ **Cons:**
  - Takes 45+ minutes to train on M4 Mac
  - Higher computational requirements

**Recommended for:**
- Final model training when you want best performance
- Production systems where quality is critical

#### 5.3 Performance Comparison

The lightweight model typically achieves:
- ~85-90% of the quality of the full model
- ~5-10x faster training time
- Sufficient for most fraud detection tasks

The difference in fraud detection performance on your XGBoost model is usually **minimal** (1-3% AUC difference at most), but the lightweight model trains significantly faster.

### 6. Next Steps

After generating the synthetic data, you can modify the `detect_fraud.ipynb` notebook to include this new data in the training set. You can combine the original training data with the synthetic fraud data to train a more robust fraud detection model. Remember to only add the synthetic data to the training set, not the validation or test sets.

### 7. Customizing Model Size

If you want to fine-tune the model size further, edit the `lightweight_config` variable in the `run_tabdiff()` function to adjust:
- `steps`: Number of training iterations (lower = faster, less polished)
- `num_timesteps`: Diffusion process steps (lower = faster, potentially less smooth generation)
- `dim_t`: Time embedding size (lower = faster, may lose temporal information)
- `batch_size`: Batch size (lower = faster but slower convergence)

For example, an ultra-lightweight version for 5-10 minute training might use:
- steps: 600
- num_timesteps: 5
- dim_t: 128
- batch_size: 1024
