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

In this competition, participants will benchmark machine learning models using Vesta's dataset, which includes a variety of features ranging from device information to transaction data. Through this process, participants will aim to create new features, reduce dimensionality, and fine-tune models to improve fraud detection.

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
- **AUC (Area Under Curve)**: Reflects the model's ability to distinguish between fraud and non-fraud cases.
- **Precision, Recall, and F1-score**: Provide a detailed look at the model’s prediction performance, especially for fraudulent transactions.

## Future Work

- Experiment with more advanced feature engineering techniques.
- Explore additional machine learning models like neural networks or ensemble models.
- Fine-tune hyperparameters to further optimize model performance.
  
## Acknowledgements

This project is based on the IEEE-CIS Fraud Detection competition. The dataset was provided by Vesta Corporation, a leader in e-commerce payment solutions.

Vesta Corporation is the forerunner in guaranteed e-commerce payment solutions, and this dataset represents real-world transactions that help train machine learning models for fraud detection.
