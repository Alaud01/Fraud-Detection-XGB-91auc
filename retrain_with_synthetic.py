#!/usr/bin/env python3
"""
Script to retrain fraud detection model with synthetic data from TabDiff.

This script:
1. Loads synthetic data from TabDiff results
2. Loads and processes original training data through full pipeline
3. Properly merges synthetic and original datasets at the correct stage
4. Processes test data through same pipeline
5. Applies PCA dimensionality reduction
6. Retrains XGBoost model with combined dataset
7. Generates predictions and saves as new_submission.csv

The script uses classes and functions from detect_fraud.ipynb but includes
a fixed final_preprocessing method that handles missing columns gracefully.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score
import xgboost as xgb
from scipy import stats
import re
from datetime import datetime, timedelta
import seaborn as sns
from sklearn.decomposition import PCA
import warnings
from pathlib import Path
import json
from sklearn.metrics import (
    classification_report, roc_auc_score, roc_curve, auc, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score
)

warnings.filterwarnings('ignore')


# ============================================================================
# Classes from detect_fraud.ipynb (with fixed final_preprocessing)
# ============================================================================

class preprocessDatasets:
    """Preprocessing class with fixed final_preprocessing method."""
    
    def join_ID(self, transaction_df: pd.DataFrame, ID_df: pd.DataFrame):
        """Join datasets based on transaction ID."""
        merged_df = pd.merge(transaction_df, ID_df, on='TransactionID', how='outer')
        col_list = []
        for col in merged_df.columns:
            if '-' in col:
                col = col.replace('-', '_')
            col_list.append(col)
        merged_df.columns = col_list
        return merged_df

    def replace_blanks(self, df: pd.DataFrame):
        """Replace blanks with -999."""
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(-999)
            else:
                df[col] = df[col].fillna('-999')
        return df

    def encode_df(self, df: pd.DataFrame):
        """Encode/scale dataframe columns."""
        label_encoder = LabelEncoder()
        scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()

        for col in df.columns:
            if col == 'TransactionID' or col == 'isFraud':
                continue
            elif re.match(r'^D\d+$', col):
                df[col] = minmax_scaler.fit_transform(df[[col]])
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = scaler.fit_transform(df[[col]])
            else:
                df[col] = label_encoder.fit_transform(df[col].astype(str))
        return df

    def remove_outliers(self, data: pd.DataFrame, column: str):
        """Remove outliers >3sigma based on a column."""
        z_scores = stats.zscore(data[column])
        non_outliers = np.abs(z_scores) < 3
        original_len = len(data)
        data = data[non_outliers]
        final_len = len(data)
        print(f'{original_len - final_len} values were removed, since they contained outliers (z-score over 3)')
        return data

    def remove_empty_cols(self, data: pd.DataFrame):
        """Drop columns if more than 90% is missing."""
        threshold = 0.90
        data.dropna(thresh=int((1 - threshold) * len(data)), axis=1, inplace=True)
        return data

    def feature_engineer(self, data: pd.DataFrame):
        """Feature engineer dataset to extract more details."""
        # Handle P_emaildomain split
        if 'P_emaildomain' in data.columns:
            email_p_str = data['P_emaildomain'].astype(str)
            split_p = email_p_str.str.split('.', n=1, expand=True)
            if split_p.shape[1] == 1:
                split_p[1] = ''
            data['P_emailserver'] = split_p[0].fillna('')
            data['P_suffix'] = split_p[1].fillna('')

        # Handle R_emaildomain split
        if 'R_emaildomain' in data.columns:
            email_r_str = data['R_emaildomain'].astype(str)
            split_r = email_r_str.str.split('.', n=1, expand=True)
            if split_r.shape[1] == 1:
                split_r[1] = ''
            data['R_emailserver'] = split_r[0].fillna('')
            data['R_suffix'] = split_r[1].fillna('')

        # Handle id_30 (OS) split
        if 'id_30' in data.columns:
            id_30_str = data['id_30'].astype(str)
            data['os'] = id_30_str.str.split(' ', expand=True)[0].fillna('')

        # Handle id_33 (screen size) split
        if 'id_33' in data.columns:
            id_33_str = data['id_33'].astype(str)
            split_screen = id_33_str.str.split('x', expand=True)
            if split_screen.shape[1] >= 1:
                data['screen_width'] = pd.to_numeric(split_screen[0], errors='coerce')
            else:
                data['screen_width'] = None
            if split_screen.shape[1] >= 2:
                data['screen_height'] = pd.to_numeric(split_screen[1], errors='coerce')
            else:
                data['screen_height'] = None

        # Modify browser identification
        if 'id_31' in data.columns:
            id_31_str = data['id_31'].astype(str)
            data['browser'] = id_31_str.str.split(' ', expand=True)[0].str.lower().fillna('')
        if 'DeviceInfo' in data.columns:
            device_str = data['DeviceInfo'].astype(str)
            data['device_name'] = device_str.str.split(' ', expand=True)[0].str.lower().fillna('')

        def matchPatterns(df: pd.DataFrame, patterns, col_name: str):
            for pattern, value in patterns.items():
                df[col_name] = df[col_name].str.replace(pattern, value, regex=True)
            return df

        browser_patterns = {
            r'samsung/sm-g532m|samsung/sch|samsung/sm-g531h': 'samsung',
            r'generic/android': 'android',
            r'mozilla/firefox': 'firefox',
            r'nokia/lumia': 'nokia',
            r'zte/blade': 'zte',
            r'lg/k-200': 'lg',
            r'lanix/ilium': 'lanix',
            r'blu/dash': 'blu',
            r'm4tel/m4': 'm4'
        }

        device_patterns = {
            r'samsung|sgh|sm|gt-': 'samsung',
            r'mot': 'motorola',
            r'ale-|.*-l|hi': 'huawei',
            r'lg': 'lg',
            r'rv:': 'rv',
            r'blade': 'zte',
            r'xt': 'sony',
            r'iphone': 'ios',
            r'lenovo': 'lenovo',
            r'mi|redmi': 'xiaomi',
            r'ilium': 'ilium',
            r'alcatel': 'alcatel',
            r'asus': 'asus'
        }

        if 'browser' in data.columns:
            data = matchPatterns(data, browser_patterns, 'browser')
        if 'device_name' in data.columns:
            data = matchPatterns(data, device_patterns, 'device_name')

        # Modify date format to split day, month, etc.
        if 'TransactionDT' in data.columns:
            start_date = datetime(2017, 11, 30)
            data['TransactionFullDate'] = data['TransactionDT'].apply(lambda x: start_date + timedelta(seconds=x))
            data['TransactionDate'] = data['TransactionFullDate'].dt.date
            data['DayOfWeek'] = data['TransactionFullDate'].dt.dayofweek.apply(lambda x: (x + 1) % 7)
            data['HourOfDay'] = data['TransactionFullDate'].dt.hour
            data['Month'] = data['TransactionFullDate'].dt.month
        return data

    def reduce_memory(self, df: pd.DataFrame):
        """Reduce memory by minimizing column data types."""
        start = df.memory_usage().sum() / 1024**2
        print('Starting memory usage of the dataframe: {:.2f} MB'.format(start))

        for col in df.columns:
            col_type = df[col].dtype

            if str(col_type).startswith('float'):
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

            elif str(col_type).startswith('int'):
                col_min = df[col].min()
                col_max = df[col].max()
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype('category')

        end = df.memory_usage().sum() / 1024**2
        print('Memory usage after downsizing: {:.2f} MB'.format(end))
        print('Memory usage decreased by {:.1f}%'.format(100 * (start - end) / start))
        return df

    def final_preprocessing(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """
        Apply final preprocessing steps with FIXED column dropping.
        Only drops columns that actually exist in the dataframe.
        """
        # Sort values to keep predictions in submission format
        train_df = train_df.sort_values(by='TransactionID', ascending=True).reset_index(drop=True)
        test_df = test_df.sort_values(by='TransactionID', ascending=True).reset_index(drop=True)

        # Define columns to drop
        cols_to_drop = ['P_emaildomain', 'R_emaildomain', 'id_30', 'id_31', 'id_33', 
                        'DeviceInfo', 'TransactionDT', 'TransactionFullDate', 
                        'TransactionDate', 'TransactionID']
        
        # Only drop columns that exist
        train_cols_to_drop = [col for col in cols_to_drop if col in train_df.columns]
        test_cols_to_drop = [col for col in cols_to_drop if col in test_df.columns]
        
        train_df.drop(columns=train_cols_to_drop, inplace=True)
        test_df.drop(columns=test_cols_to_drop, inplace=True)

        # Due to removal of cols with >90% missing values, the column removals need to be carried over to test dataset as well
        col_drop = []
        for col in test_df.columns:
            if col not in train_df.columns:
                col_drop.append(col)

        dropped_df = test_df.drop(columns=col_drop)
        return train_df, dropped_df


class ReduceDeminesion:
    """Dimensionality reduction using PCA."""
    
    def __init__(self, traindata: pd.DataFrame, testdata: pd.DataFrame):
        self.traindata = traindata
        self.testdata = testdata
        self.v_cols = []

    def plot_and_reduceD(self, plots=True):
        """Plot scree plot and apply PCA."""
        plt.style.use('seaborn-v0_8-whitegrid')

        v_cols = [col for col in self.traindata.columns if re.match(r'^V\d+$', col)]

        # PCA on training data
        v_data = self.traindata[v_cols]
        pca = PCA().fit(v_data)
        v_data_pca = pca.transform(v_data)
        explained_variance = pca.explained_variance_ratio_.cumsum()

        # Apply PCA to test data using the same PCA model
        v_test_data = self.testdata[v_cols]
        v_test_data_pca = pca.transform(v_test_data)

        # Number of components to retain based on Kaiser criterion
        num_components_kaiser = sum(eigenvalue > 1 for eigenvalue in pca.explained_variance_)
        print(f"Number of components to retain (Kaiser criterion): {num_components_kaiser}")

        if plots:
            # Scree plot
            plt.figure(figsize=(12, 7))
            sns.lineplot(x=range(1, len(pca.explained_variance_ratio_) + 1),
                         y=pca.explained_variance_ratio_,
                         marker='o', linestyle='--', color='darkblue', label='Explained Variance Ratio')
            plt.xlabel('Principal Component', fontsize=12)
            plt.ylabel('Explained Variance Ratio', fontsize=12)
            plt.title('Scree Plot: Explained Variance Ratio', fontsize=16)
            plt.legend()
            plt.show()

            # Cumulative explained variance plot
            plt.figure(figsize=(12, 7))
            sns.lineplot(x=range(1, len(explained_variance) + 1),
                         y=explained_variance,
                         marker='o', linestyle='-', color='darkgreen', label='Cumulative Explained Variance')
            plt.axhline(y=0.90, color='red', linestyle='--', linewidth=2, label='90% Variance Threshold')
            plt.text(x=1, y=0.91, s='90% mark', color='red', fontsize=10, ha='left')
            plt.xlabel('Number of Principal Components', fontsize=12)
            plt.ylabel('Cumulative Explained Variance', fontsize=12)
            plt.title('Cumulative Explained Variance vs. Number of Principal Components', fontsize=16)
            plt.legend(loc='lower right')
            plt.show()

        # Determine the number of components to retain for 90% variance
        comp_90 = next(i for i, total in enumerate(explained_variance) if total >= 0.90) + 1
        print(f"\nNumber of components to retain for 90% variance: {comp_90}")
        print(f"Cumulative Explained Variance for the {comp_90}th component: {explained_variance[comp_90-1]:.4f}\n")

        # Transform training data into the reduced-dimensional PCA space
        v_data_pca_df = pd.DataFrame(v_data_pca[:, :comp_90], columns=[f'PC{i+1}' for i in range(comp_90)])
        data_final = pd.concat([self.traindata.drop(columns=v_cols).reset_index(drop=True), v_data_pca_df], axis=1)

        # Transform test data into the same reduced-dimensional PCA space
        v_test_data_pca_df = pd.DataFrame(v_test_data_pca[:, :comp_90], columns=[f'PC{i+1}' for i in range(comp_90)])
        test_data_final = pd.concat([self.testdata.drop(columns=v_cols).reset_index(drop=True), v_test_data_pca_df], axis=1)

        return data_final, test_data_final


class applyModel:
    """Model training and evaluation class."""
    
    def __init__(self, eval=True):
        self.eval = eval

    def evaluateModel(self, y_test, y_pred, y_pred_proba):
        """Evaluate the model using various metrics."""
        plt.style.use('seaborn-v0_8-whitegrid')

        # Confusion Matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', linewidths=.5, cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.ylabel('True Label', fontsize=12)
        plt.tight_layout()
        plt.show()

        # Basic Metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)

        # Additional Metrics
        balanced_acc = balanced_accuracy_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        kappa = cohen_kappa_score(y_test, y_pred)

        # Classification Report
        class_report = classification_report(y_test, y_pred)

        # Print Metrics
        print('\n\n -------------Model Evaluation Metrics-------------')
        print(f'Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1:.4f}')
        print(f'ROC-AUC: {roc_auc:.4f}')
        print(f'Balanced Accuracy: {balanced_acc:.4f}')
        print(f'Matthew\'s Correlation Coefficient: {mcc:.4f}')
        print(f'Cohen\'s Kappa: {kappa:.4f}')
        print('\nClassification Report:')
        print(class_report)

        metrics = [accuracy, precision, recall, f1, roc_auc, balanced_acc, mcc, kappa]
        return metrics

    def plot_training_curves(self, evals_result):
        """Plot training and validation curves to detect overfitting."""
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Extract metrics
        epochs = len(evals_result['validation_0']['auc'])
        x_axis = range(0, epochs)
        
        # Get training and validation AUC
        train_auc = evals_result['validation_0']['auc']  # validation_0 is training set
        val_auc = evals_result['validation_1']['auc']     # validation_1 is validation set
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(x_axis, train_auc, label='Train AUC', color='blue', linewidth=2)
        ax.plot(x_axis, val_auc, label='Validation AUC', color='red', linewidth=2)
        
        # Find best iteration (highest validation AUC)
        best_iter = np.argmax(val_auc)
        best_val_auc = val_auc[best_iter]
        best_train_auc = train_auc[best_iter]
        
        # Mark best iteration
        ax.axvline(x=best_iter, color='green', linestyle='--', linewidth=1.5, 
                   label=f'Best Iteration ({best_iter})')
        ax.plot(best_iter, best_val_auc, 'go', markersize=10, label=f'Best Val AUC: {best_val_auc:.4f}')
        
        ax.set_xlabel('Iteration (Boosting Round)', fontsize=12)
        ax.set_ylabel('AUC Score', fontsize=12)
        ax.set_title('Training vs Validation AUC - Overfitting Detection', fontsize=16)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Add gap analysis
        gap = train_auc[-1] - val_auc[-1]
        gap_pct = (gap / train_auc[-1]) * 100 if train_auc[-1] > 0 else 0
        
        # Add text box with gap information
        textstr = f'Final Gap: {gap:.4f} ({gap_pct:.2f}%)\n'
        textstr += f'Train AUC: {train_auc[-1]:.4f}\n'
        textstr += f'Val AUC: {val_auc[-1]:.4f}\n'
        textstr += f'Best Val AUC: {best_val_auc:.4f} @ iter {best_iter}'
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.show()
        
        # Print gap analysis
        print(f"\n{'='*60}")
        print("OVERFITTING ANALYSIS")
        print(f"{'='*60}")
        print(f"Final Training AUC: {train_auc[-1]:.4f}")
        print(f"Final Validation AUC: {val_auc[-1]:.4f}")
        print(f"Gap: {gap:.4f} ({gap_pct:.2f}%)")
        print(f"Best Validation AUC: {best_val_auc:.4f} at iteration {best_iter}")
        print(f"Early stopping triggered: {'Yes' if len(val_auc) < 5000 else 'No'}")
        
        if gap_pct > 5:
            print("⚠️  WARNING: Significant gap detected (>5%) - model may be overfitting")
        elif gap_pct > 2:
            print("⚠️  CAUTION: Moderate gap detected (>2%) - monitor closely")
        else:
            print("✓ Gap is acceptable (<2%)")
        print(f"{'='*60}\n")

    def trainModel(self, X_train_split: pd.DataFrame, X_val_split: pd.DataFrame,
                   y_train_split: pd.DataFrame, y_val_split: pd.DataFrame, y_train: pd.DataFrame):
        """Train XGBoost model."""
        # Weigh classes differently since majority are "safe" transactions
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
        print(f"  Class weight (scale_pos_weight): {scale_pos_weight:.4f}")
        print(f"  Training samples - Fraud: {(y_train_split == 1).sum()}, Non-fraud: {(y_train_split == 0).sum()}")

        # Define the XGBoost model
        xgb_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='auc',
            use_label_encoder=False,
            learning_rate=0.05,
            n_estimators=5000,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=10
        )

        # Train the XGBoost model with eval_set to track metrics
        xgb_model.fit(
            X_train_split,
            y_train_split,
            eval_set=[(X_train_split, y_train_split), (X_val_split, y_val_split)],  # Track both train and val
            verbose=False
        )

        # Extract evaluation results for plotting
        evals_result = xgb_model.evals_result()
        
        # Plot training vs validation metrics to detect overfitting
        if self.eval:
            self.plot_training_curves(evals_result)

        # Predict probabilities of fraud for the val/test set
        yval_pred = xgb_model.predict(X_val_split)
        yval_pred_proba = xgb_model.predict_proba(X_val_split)[:, 1]

        if self.eval:
            metrics = self.evaluateModel(y_val_split, yval_pred, yval_pred_proba)

        return xgb_model

    def pred_and_submit(self, model: xgb.XGBClassifier, Xtest: pd.DataFrame,
                       test_transaction_df: pd.DataFrame, plot_pred=True):
        """Apply model to predict and format predictions for submission."""
        # Predict probabilities of fraud for the test set
        y_pred_proba = model.predict_proba(Xtest)[:, 1]

        # Create the submission DataFrame
        submission = pd.DataFrame({
            'TransactionID': test_transaction_df['TransactionID'],
            'isFraud': y_pred_proba
        })

        if plot_pred:
            plt.style.use('seaborn-v0_8-whitegrid')
            plt.figure(figsize=(10, 6))
            sns.histplot(submission['isFraud'], bins=100, kde=True, color='green', edgecolor='black')
            plt.title('Histogram of Fraud Probability Predictions', fontsize=16)
            plt.xlabel('Predicted Fraud Probability', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.tight_layout()
            plt.show()

        # Save the submission DataFrame to CSV
        submission.to_csv('new_submission.csv', index=False)
        return submission


# ============================================================================
# Main retraining script
# ============================================================================

def main():
    """Main function to retrain model with synthetic data."""
    print("=" * 80)
    print("RETRAINING MODEL WITH SYNTHETIC DATA FROM TABDIFF")
    print("=" * 80)

    # Step 1: Load synthetic data from TabDiff results
    print("\nStep 1: Loading synthetic data from TabDiff results...")
    tabdiff_results_dir = Path("TabDiff/tabdiff/result/fraud_data/quick_fraud")
    synthetic_files = []

    # Check all epoch directories for samples.csv files (prefer EMA model)
    if tabdiff_results_dir.exists():
        for epoch_dir in sorted(tabdiff_results_dir.glob("*"), reverse=True):
            if epoch_dir.is_dir() and epoch_dir.name.isdigit():
                # Prefer EMA model samples
                ema_samples = epoch_dir / "ema" / "samples.csv"
                regular_samples = epoch_dir / "samples.csv"

                if ema_samples.exists():
                    synthetic_files.append((int(epoch_dir.name), ema_samples, "ema"))
                elif regular_samples.exists():
                    synthetic_files.append((int(epoch_dir.name), regular_samples, "regular"))

    if not synthetic_files:
        raise FileNotFoundError(f"No synthetic data files found in {tabdiff_results_dir}")

    # Use the latest epoch's samples
    synthetic_files.sort(key=lambda x: x[0], reverse=True)
    epoch_num, syn_file_path, model_type = synthetic_files[0]
    print(f"Loading synthetic data from epoch {epoch_num} ({model_type} model)")
    print(f"File: {syn_file_path}")

    syn_df = pd.read_csv(syn_file_path)
    print(f"Synthetic data shape: {syn_df.shape}")
    print(f"Fraud cases in synthetic data: {(syn_df['isFraud'] == 1).sum() if 'isFraud' in syn_df.columns else 'N/A'}")

    # Step 2: Process original training data through FULL pipeline
    print("\nStep 2: Processing original training data through full pipeline...")
    train_id_df_orig = pd.read_csv(r"data/ieee-fraud-detection/train_identity.csv")
    train_transaction_df_orig = pd.read_csv(r"data/ieee-fraud-detection/train_transaction.csv")

    PPD_combined = preprocessDatasets()

    # Remove empty cols
    PPD_combined.remove_empty_cols(train_id_df_orig)
    PPD_combined.remove_empty_cols(train_transaction_df_orig)

    # Join ID and transaction
    train_df_orig = PPD_combined.join_ID(train_transaction_df_orig, train_id_df_orig)

    # Remove outliers
    print("  Removing outliers...")
    train_df_orig = PPD_combined.remove_outliers(train_df_orig, 'TransactionAmt')

    # Feature engineer (this requires RAW data with string columns)
    print("  Feature engineering...")
    train_df_orig = PPD_combined.feature_engineer(train_df_orig)

    # Replace blanks
    print("  Replacing blanks...")
    PPD_combined.replace_blanks(train_df_orig)

    # Reduce memory
    print("  Reducing memory...")
    PPD_combined.reduce_memory(train_df_orig)

    # Encode (label encoding only, no standardization yet)
    print("  Encoding (label encoding only)...")
    for col in train_df_orig.columns:
        if col == 'TransactionID' or col == 'isFraud':
            continue
        if not pd.api.types.is_numeric_dtype(train_df_orig[col]):
            label_encoder = LabelEncoder()
            train_df_orig[col] = label_encoder.fit_transform(train_df_orig[col].astype(str))

    print(f"  Original data shape after label encoding: {train_df_orig.shape}")

    # Step 3: Align datasets and standardize both together
    print("\nStep 3: Aligning datasets and standardizing both together...")

    # Ensure synthetic data has TransactionID and isFraud
    if 'TransactionID' not in syn_df.columns:
        max_orig_id = train_df_orig['TransactionID'].max()
        syn_df['TransactionID'] = range(int(max_orig_id) + 1, int(max_orig_id) + 1 + len(syn_df))
        print("  Generated TransactionIDs for synthetic data")

    if 'isFraud' not in syn_df.columns:
        print("  WARNING: Synthetic data missing 'isFraud' column. Setting to 1.")
        syn_df['isFraud'] = 1

    # Find common columns (columns that exist in both datasets)
    common_cols = [col for col in train_df_orig.columns if col in syn_df.columns]

    # Ensure TransactionID and isFraud are included
    if 'TransactionID' not in common_cols:
        common_cols.insert(0, 'TransactionID')
    if 'isFraud' not in common_cols:
        common_cols.append('isFraud')

    print(f"  Common columns: {len(common_cols)}")
    print(f"  Columns in original: {len(train_df_orig.columns)}")
    print(f"  Columns in synthetic: {len(syn_df.columns)}")

    # Align both datasets to common columns (before standardization)
    train_df_aligned = train_df_orig[common_cols].copy()
    syn_df_aligned = syn_df[common_cols].copy()

    print(f"  Original data shape after alignment: {train_df_aligned.shape}")
    print(f"  Synthetic data shape after alignment: {syn_df_aligned.shape}")

    # Step 3b: Combine datasets BEFORE standardization, then standardize together
    print("\nStep 3b: Combining datasets, then standardizing together...")
    train_df_combined = pd.concat([train_df_aligned, syn_df_aligned], ignore_index=True)
    print(f"  Combined data shape before standardization: {train_df_combined.shape}")

    # Now apply standardization to the combined dataset
    scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    print("  Applying standardization to combined dataset...")
    for col in train_df_combined.columns:
        if col == 'TransactionID' or col == 'isFraud':
            continue

        # MinMax Scale D-columns
        if re.match(r'^D\d+$', col):
            train_df_combined[col] = minmax_scaler.fit_transform(train_df_combined[[col]])

        # StandardScaler for other numeric columns
        elif pd.api.types.is_numeric_dtype(train_df_combined[col]):
            train_df_combined[col] = scaler.fit_transform(train_df_combined[[col]])

    print("  ✓ Combined dataset standardized")

    # Step 4: Combined dataset is now ready
    print("\nStep 4: Combined dataset ready for further processing...")
    print(f"Combined training data shape: {train_df_combined.shape}")
    print(f"Fraud cases: {(train_df_combined['isFraud'] == 1).sum()}")
    print(f"Non-fraud cases: {(train_df_combined['isFraud'] == 0).sum()}")

    # Step 5: Process test data (same pipeline as original)
    print("\nStep 5: Processing test data...")
    test_id_df = pd.read_csv(r"data/ieee-fraud-detection/test_identity.csv")
    test_transaction_df = pd.read_csv(r"data/ieee-fraud-detection/test_transaction.csv")

    PPD_combined.remove_empty_cols(test_id_df)
    PPD_combined.remove_empty_cols(test_transaction_df)
    test_df = PPD_combined.join_ID(test_transaction_df, test_id_df)

    # Feature engineer (no outlier removal for test data, same as original script)
    print("  Feature engineering...")
    test_df = PPD_combined.feature_engineer(test_df)

    # Replace blanks
    print("  Replacing blanks...")
    PPD_combined.replace_blanks(test_df)

    # Reduce memory
    print("  Reducing memory...")
    PPD_combined.reduce_memory(test_df)

    # Encode (label encoding only, same as training)
    print("  Encoding (label encoding only)...")
    for col in test_df.columns:
        if col == 'TransactionID':
            continue
        if not pd.api.types.is_numeric_dtype(test_df[col]):
            label_encoder = LabelEncoder()
            test_df[col] = label_encoder.fit_transform(test_df[col].astype(str))

    # Standardize test data (same as training)
    print("  Standardizing test data...")
    scaler = StandardScaler()
    minmax_scaler = MinMaxScaler()

    for col in test_df.columns:
        if col == 'TransactionID':
            continue
        if re.match(r'^D\d+$', col):
            test_df[col] = minmax_scaler.fit_transform(test_df[[col]])
        elif pd.api.types.is_numeric_dtype(test_df[col]):
            test_df[col] = scaler.fit_transform(test_df[[col]])

    print(f"  Test data shape after preprocessing: {test_df.shape}")

    # Step 6: Apply PCA dimensionality reduction
    print("\nStep 6: Applying PCA dimensionality reduction...")
    # Pass full dataframes to PCA - only exclude isFraud from train for PCA input
    train_for_pca = train_df_combined.drop(columns=['isFraud'] if 'isFraud' in train_df_combined.columns else [])
    test_for_pca = test_df

    # Apply PCA - it will preserve all non-V columns automatically
    RD_combined = ReduceDeminesion(train_for_pca, test_for_pca)
    train_df_pca, test_df_pca = RD_combined.plot_and_reduceD(plots=False)

    # Add isFraud back to train_df_pca for final preprocessing
    if 'isFraud' in train_df_combined.columns:
        train_df_pca['isFraud'] = train_df_combined['isFraud'].values

    # Final preprocessing (handles column alignment automatically, with FIXED method)
    train_df_combined, test_df_processed = PPD_combined.final_preprocessing(train_df_pca, test_df_pca)

    print(f"Final training data shape after PCA: {train_df_combined.shape}")
    print(f"Final test data shape after PCA: {test_df_processed.shape}")

    # Step 7: Split data and retrain model
    print("\nStep 7: Splitting data and retraining model...")
    X_train_combined = train_df_combined.drop(columns=['isFraud'])
    y_train_combined = train_df_combined['isFraud']
    X_test_combined = test_df_processed

    # Split into train and validation
    X_train_split_combined, X_val_split_combined, y_train_split_combined, y_val_split_combined = train_test_split(
        X_train_combined, y_train_combined, test_size=0.2, stratify=y_train_combined, random_state=42
    )

    print(f"Training set size: {X_train_split_combined.shape[0]}")
    print(f"Validation set size: {X_val_split_combined.shape[0]}")
    print(f"Training fraud cases: {(y_train_split_combined == 1).sum()}")
    print(f"Training non-fraud cases: {(y_train_split_combined == 0).sum()}")

    # Train model
    print("\nStep 8: Training XGBoost model...")
    AM_combined = applyModel(eval=True)
    xgbModel_combined = AM_combined.trainModel(
        X_train_split_combined, X_val_split_combined,
        y_train_split_combined, y_val_split_combined,
        y_train_combined
    )

    # Step 9: Generate predictions and save submission
    print("\nStep 9: Generating predictions with retrained model...")
    # Predict probabilities
    y_pred_proba = xgbModel_combined.predict_proba(X_test_combined)[:, 1]

    # Create submission DataFrame
    final_predictions_combined = pd.DataFrame({
        'TransactionID': test_transaction_df['TransactionID'],
        'isFraud': y_pred_proba
    })

    # Plot predictions histogram
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))
    sns.histplot(final_predictions_combined['isFraud'], bins=100, kde=True, color='green', edgecolor='black')
    plt.title('Histogram of Fraud Probability Predictions (Retrained with Synthetic Data)', fontsize=16)
    plt.xlabel('Predicted Fraud Probability', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.tight_layout()
    plt.show()

    # Save as new_submission.csv
    final_predictions_combined.to_csv('new_submission.csv', index=False)
    print(f"\n✓ Successfully generated new_submission.csv!")
    print(f"Predictions shape: {final_predictions_combined.shape}")
    print(f"\nPrediction statistics:")
    print(f"  Min probability: {final_predictions_combined['isFraud'].min():.6f}")
    print(f"  Max probability: {final_predictions_combined['isFraud'].max():.6f}")
    print(f"  Mean probability: {final_predictions_combined['isFraud'].mean():.6f}")
    print(f"  Median probability: {final_predictions_combined['isFraud'].median():.6f}")
    print(f"  Std probability: {final_predictions_combined['isFraud'].std():.6f}")

    # Calculate and save model metrics
    print("\nStep 10: Calculating final model metrics...")
    y_val_pred = xgbModel_combined.predict(X_val_split_combined)
    y_val_pred_proba = xgbModel_combined.predict_proba(X_val_split_combined)[:, 1]

    metrics_dict = {
        'accuracy': float(accuracy_score(y_val_split_combined, y_val_pred)),
        'precision': float(precision_score(y_val_split_combined, y_val_pred)),
        'recall': float(recall_score(y_val_split_combined, y_val_pred)),
        'f1_score': float(f1_score(y_val_split_combined, y_val_pred)),
        'roc_auc': float(roc_auc_score(y_val_split_combined, y_val_pred_proba)),
        'balanced_accuracy': float(balanced_accuracy_score(y_val_split_combined, y_val_pred)),
        'matthews_corrcoef': float(matthews_corrcoef(y_val_split_combined, y_val_pred)),
        'cohen_kappa': float(cohen_kappa_score(y_val_split_combined, y_val_pred)),
        'training_samples': int(len(X_train_split_combined)),
        'validation_samples': int(len(X_val_split_combined)),
        'synthetic_samples': int(len(syn_df)),
        'original_samples': int(len(train_transaction_df_orig)),
        'combined_samples': int(len(train_df_combined)),
        'epoch_used': int(epoch_num),
        'model_type': model_type
    }

    # Save metrics to JSON
    with open('new_submission_metrics.json', 'w') as f:
        json.dump(metrics_dict, f, indent=2)

    print("\nModel Metrics:")
    for key, value in metrics_dict.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    print("\n✓ Metrics saved to new_submission_metrics.json")

    print("\n" + "=" * 80)
    print("RETRAINING COMPLETE!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - new_submission.csv (predictions)")
    print(f"  - new_submission_metrics.json (model metrics)")
    print("=" * 80)


if __name__ == '__main__':
    main()

