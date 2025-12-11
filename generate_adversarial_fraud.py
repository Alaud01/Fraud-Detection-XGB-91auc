#!/usr/bin/env python3
"""
Script to generate adversarial fraud examples using TabDiff-inspired noise and purify them with TabDiff.

This script:
1. Loads the IEEE Fraud Detection dataset.
2. Runs the exact preprocessing pipeline from detect_fraud.ipynb (including PCA and scaling).
3. Splits the data into Train/Validation sets (Validation set acts as the "test dataset" with known labels).
4. Loads the latest XGBoost model from xgb_saved/.
5. Loads TabDiff model (trained on processed data with PCA) for purification.
6. Selects 10% of the FRAUD cases from the validation set.
7. Iteratively adds TabDiff-style Gaussian noise (following the diffusion schedule) until the model predicts the case as "Safe".
8. Purifies adversarial samples using TabDiff's reverse diffusion process to make them more realistic.
9. Outputs the adversarial samples and purified adversarial samples (both in processed/PCA space).

Note: 
- Adversarial samples are created in processed feature space (with PC columns from PCA).
- TabDiff purification happens directly in processed space (no inverse PCA needed).
- Purified samples can be directly evaluated with XGBoost model.
- Categorical columns are treated as continuous in the encoded/scaled feature space for perturbation.
"""

import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy import stats
from datetime import datetime, timedelta
import warnings
from pathlib import Path
import time
import torch
import pickle
import json
import glob

# Add TabDiff to path
tabdiff_path = os.path.join(os.path.dirname(__file__), 'TabDiff')
if os.path.exists(tabdiff_path):
    sys.path.insert(0, tabdiff_path)
    try:
        from tabdiff.modules.main_modules import UniModMLP, Model
        from tabdiff.models.unified_ctime_diffusion import UnifiedCtimeDiffusion
        from utils_train import preprocess as tabdiff_preprocess
        import src
        TABDIFF_AVAILABLE = True
    except ImportError as e:
        print(f"Warning: TabDiff modules not available: {e}")
        TABDIFF_AVAILABLE = False
else:
    TABDIFF_AVAILABLE = False

# Filter warnings
warnings.filterwarnings('ignore')

# ==========================================
# Preprocessing Classes from detect_fraud.ipynb
# ==========================================

class preprocessDatasets:
    def __init__(self):
        self.cat_columns = []

    def join_ID(self, transaction_df: pd.DataFrame, ID_df: pd.DataFrame):  
        ''' 
        Function to join datasets based on transaction ID.
        '''
        merged_df = pd.merge(transaction_df, ID_df, on='TransactionID', how='outer')
        col_list = []
        for col in merged_df.columns:
            if '-' in col:  # Replace '-' and '_' since column names between the two datasets does not align
                col = col.replace('-', '_')
            col_list.append(col)
        merged_df.columns = col_list
        return merged_df

    def replace_blanks(self, df: pd.DataFrame):
        ''' 
        Function to replace blanks with "holder" value of -999.
        '''
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):  # Check if the column is numeric
                df[col] = df[col].fillna(-999)
            else:
                df[col] = df[col].fillna('-999')
        return df
  
    def encode_df(self, df: pd.DataFrame):
        ''' 
        Encode/scale dataframe columns to maintain data quality prior to feeding into model.
        '''
        # Initialize stores if empty (first call)
        if not hasattr(self, 'scalers'):
            self.scalers = {}
            self.label_encoders = {}
            self.minmax_scalers = {}

        # Loop through each column and determine its type, then encode/scale accordingly
        for col in df.columns:
            if col == 'TransactionID' or col == 'isFraud':  # These two columns can be left as is
                continue
            
            # MinMax Scale D-columns as indicated from the plots and correlation matrix
            elif re.match(r'^D\d+$', col):
                if col not in self.minmax_scalers:
                    self.minmax_scalers[col] = MinMaxScaler()
                    df[col] = self.minmax_scalers[col].fit_transform(df[[col]])
                else:
                    df[col] = self.minmax_scalers[col].transform(df[[col]])

            # Check if the column is numeric
            elif pd.api.types.is_numeric_dtype(df[col]):
                if col not in self.scalers:
                    self.scalers[col] = StandardScaler()
                    df[col] = self.scalers[col].fit_transform(df[[col]])
                else:
                    df[col] = self.scalers[col].transform(df[[col]])

            # Process non-numeric columns (object types)
            else:
                if col not in self.cat_columns:
                    self.cat_columns.append(col)
                
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen labels robustly
                    le = self.label_encoders[col]
                    
                    # Efficiently identify and replace unseen values
                    # Get the set of known classes
                    valid_classes = set(le.classes_)
                    
                    # Get the values to transform
                    values = df[col].astype(str).values
                    
                    # Replace unseen values with the first class (safe fallback)
                    # This avoids the ValueError and keeps the pipeline running
                    # Using a list comprehension is generally efficient for string arrays
                    fallback_val = le.classes_[0]
                    safe_values = [x if x in valid_classes else fallback_val for x in values]
                    
                    df[col] = le.transform(safe_values)

        return df
  
    def remove_outliers(self, data: pd.DataFrame, column: str):
        '''
        Remove outliers in transaction amount col that are >3sigma based on a single column.
        '''
        z_scores = stats.zscore(data[column])  # z-score each of the columns
        non_outliers = np.abs(z_scores) < 3
        original_len = len(data)  # calculate to initial length to note count of values removed
        data = data[non_outliers]
        final_len = len(data)  # calculate to final length to note count of values removed
        print(f'{original_len - final_len} values were removed, since they contianed outliers (z-score over 3)')
        return data
  
    def remove_empty_cols(self, data: pd.DataFrame):
        '''
        Drop col if more than 90% of the col is missing (NaN).
        '''
        threshold = 0.90
        data.dropna(thresh=int((1 - threshold) * len(data)), axis=1, inplace=True)
        return data
  
    def feature_engineer(self, data: pd.DataFrame):
        '''
        Feature engineer dataset to extract more details from certain columns.
        '''
        # Handle P_emaildomain split - convert to string first to handle encoded numeric values
        if 'P_emaildomain' in data.columns:
            email_p_str = data['P_emaildomain'].astype(str)
            split_p = email_p_str.str.split('.', n=1, expand=True)
            if split_p.shape[1] == 1:
                split_p[1] = ''
            data['P_emailserver'] = split_p[0].fillna('')
            data['P_suffix'] = split_p[1].fillna('')
        
        # Handle R_emaildomain split - convert to string first to handle encoded numeric values
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
                # Apply regex pattern to match and replace values in the column
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

        data = matchPatterns(data, browser_patterns, 'browser')
        data = matchPatterns(data, device_patterns, 'device_name')

        # Modify date format to split day, month, etc.
        start_date = datetime(2017, 11, 30)
        data['TransactionFullDate'] = data['TransactionDT'].apply(lambda x: start_date + timedelta(seconds=x))
        data['TransactionDate'] = data['TransactionFullDate'].dt.date
        data['DayOfWeek'] = data['TransactionFullDate'].dt.dayofweek.apply(lambda x: (x + 1) % 7)    # Sunday=0, Monday=1, etc
        data['HourOfDay'] = data['TransactionFullDate'].dt.hour   # from 0 to 23
        data['Month'] = data['TransactionFullDate'].dt.month
        return data
  
    def reduce_memory(self, df: pd.DataFrame):
        '''
        Reduce memory by analyzing max/min value in cols and minimizing column data type
        '''
        start = df.memory_usage().sum() / 1024**2  # Get initial memory
        print('Starting memory usage of the dataframe: {:.2f} MB'.format(start))
        
        for col in df.columns:  # Apply memory reduction steps
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
        
        end = df.memory_usage().sum() / 1024**2  # Get final memory
        print('Memory usage after downsizing: {:.2f} MB'.format(end))
        print('Memory usage decreased by {:.1f}%'.format(100 * (start - end) / start))
        return df

    def final_preprocessing(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        '''
        Apply final preprocessing steps such as dropping cols, sorting, reformatting, etc.
        '''
        # Sort values to keep predictions in submission format
        train_df = train_df.sort_values(by='TransactionID', ascending=True).reset_index(drop=True)
        test_df = test_df.sort_values(by='TransactionID', ascending=True).reset_index(drop=True)

        # Drop unnecessary columns
        cols_to_drop = ['P_emaildomain', 'R_emaildomain', 'id_30', 'id_31', 'id_33', 'DeviceInfo', 'TransactionDT', 'TransactionFullDate', 'TransactionDate', 'TransactionID']
        train_df.drop(columns=[c for c in cols_to_drop if c in train_df.columns], inplace=True)
        test_df.drop(columns=[c for c in cols_to_drop if c in test_df.columns], inplace=True)

        # Due to removal of cols with >90% missing values, the column removals need to be carried over to test dataset as well
        col_drop =[]
        for col in test_df.columns:
            if col not in train_df.columns:
                col_drop.append(col)
        
        dropped_df = test_df.drop(columns=col_drop)
        return train_df, dropped_df

class ReduceDeminesion():  # Due to size of dataset, apply dimensionality reduction steps
    def __init__(self, traindata: pd.DataFrame, testdata: pd.DataFrame):
        '''
        Initialize class
        '''
        self.traindata = traindata
        self.testdata = testdata
        self.v_cols = []
    
    def plot_and_reduceD(self, plots=False):
        '''
        Plot scree plot and apply Principal-Component-Analysis to ensure important correlations are extracted
        '''
        v_cols = [col for col in self.traindata.columns if re.match(r'^V\d+$', col)]
        
        # PCA on training data
        v_data = self.traindata[v_cols]
        self.pca = PCA().fit(v_data)  # Fit PCA on training data
        v_data_pca = self.pca.transform(v_data)  # Transform training data
        explained_variance = self.pca.explained_variance_ratio_.cumsum()

        # Apply PCA to test data using the same PCA model
        v_test_data = self.testdata[v_cols]
        v_test_data_pca = self.pca.transform(v_test_data)  # Transform test data using fitted PCA model

        # Number of components to retain based on Kaiser criterion
        num_components_kaiser = sum(eigenvalue > 1 for eigenvalue in self.pca.explained_variance_)
        
        # Determine the number of components to retain for 90% variance
        comp_90 = next(i for i, total in enumerate(explained_variance) if total >= 0.90) + 1
        print(f"\nNumber of components to retain for 90% variance: {comp_90}")

        # Transform training data into the reduced-dimensional PCA space
        v_data_pca_df = pd.DataFrame(v_data_pca[:, :comp_90], columns=[f'PC{i+1}' for i in range(comp_90)])
        data_final = pd.concat([self.traindata.drop(columns=v_cols).reset_index(drop=True), v_data_pca_df], axis=1)

        # Transform test data into the same reduced-dimensional PCA space
        v_test_data_pca_df = pd.DataFrame(v_test_data_pca[:, :comp_90], columns=[f'PC{i+1}' for i in range(comp_90)])
        test_data_final = pd.concat([self.testdata.drop(columns=v_cols).reset_index(drop=True), v_test_data_pca_df], axis=1)

        # Return both the transformed training and test datasets
        return data_final, test_data_final

# ==========================================
# Helper Functions
# ==========================================

def get_latest_model(model_dir='xgb_saved'):
    """Finds the latest XGBoost JSON model in the directory."""
    files = list(Path(model_dir).glob('*.json'))
    if not files:
        return None
    latest_file = max(files, key=os.path.getctime)
    return str(latest_file)

def get_tabdiff_noise_sigma(t, sigma_min=0.002, sigma_max=80, rho=7):
    """
    Calculates the noise standard deviation (sigma) for a given diffusion time step t.
    Based on TabDiff's default PowerMean noise schedule.
    """
    sigma = (sigma_min ** (1/rho) + t * (sigma_max ** (1/rho) - sigma_min ** (1/rho))) ** rho
    return sigma

def apply_tabdiff_noise(sample, t, sigma_num, cat_indices, unique_vals_list):
    """
    Applies TabDiff-style noise:
    - Numerical: Gaussian noise N(0, sigma_num)
    - Categorical: Random replacement with prob t (linear schedule approximation)
    """
    noisy_sample = sample.copy()
    
    # 1. Numerical Noise (Gaussian)
    # We assume all indices NOT in cat_indices are numerical
    is_cat = np.zeros(len(sample), dtype=bool)
    if cat_indices:
        is_cat[cat_indices] = True
    
    # Add Gaussian noise to numerical columns
    # We generate noise for all, but only apply to numerical
    noise = np.random.normal(0, 1, size=len(sample)) * sigma_num
    noisy_sample[~is_cat] += noise[~is_cat]
        
    # 2. Categorical Noise (Discrete Flipping) - DISABLED as per user request
    # Probability of replacement p = t (bounded [0, 1])
    # move_chance = np.clip(t, 0, 1)
    
    # for i, col_idx in enumerate(cat_indices):
    #     if np.random.random() < move_chance:
    #         # Replace with random valid value from the column's distribution
    #         valid_vals = unique_vals_list[i]
    #         noisy_sample[col_idx] = np.random.choice(valid_vals)
            
    return noisy_sample

def inverse_transform_sample(sample, feature_names, PPD, RD):
    """
    Maps a processed sample back to the input space (Inverse PCA -> Inverse Scale -> Inverse Encode).
    """
    # 1. Reconstruct DataFrame with column names
    # Sample is 1D array
    df_sample = pd.DataFrame([sample], columns=feature_names)
    
    # 2. Inverse PCA
    # Identify PC columns
    pc_cols = [c for c in feature_names if c.startswith('PC')]
    if pc_cols and hasattr(RD, 'pca'):
        # Extract PC values
        pc_values = df_sample[pc_cols].values
        
        # Inverse transform to get V columns
        # PCA was fit on v_cols. inverse_transform returns array of shape (1, n_v_cols)
        # BUT, we only kept 'comp_90' components.
        # The PCA object has n_components = n_features (full rank)? No, sklearn PCA keeps all unless specified.
        # We truncated the OUTPUT df to comp_90.
        # So we only have values for PC1...PC9.
        # We need to pad the rest with zeros to inverse transform?
        # Sklearn PCA inverse_transform requires input shape (n_samples, n_components).
        # If we truncated, we are missing data. We cannot perfectly reconstruct.
        # However, typically 'PC' cols in dataframe are subset.
        # If we can't inverse PCA, we can't reconstruct V cols.
        # We will skip V col reconstruction if dimensions don't match.
        
        n_components_kept = len(pc_cols)
        if n_components_kept == RD.pca.n_components_:
             v_values = RD.pca.inverse_transform(pc_values)
             # Map back to V column names? RD doesn't store v_cols explicitly in self.v_cols (it does in init)
             # We need the list of V columns used.
             # We can regex 'V\d+' from traindata columns if we have them.
             pass
    
    # 3. Inverse Scale and Decode
    # We iterate over available columns in df_sample (excluding PCs for now)
    df_reconstructed = df_sample.copy()
    
    for col in df_reconstructed.columns:
        if col in PPD.minmax_scalers:
            val = df_reconstructed[col].values.reshape(-1, 1)
            df_reconstructed[col] = PPD.minmax_scalers[col].inverse_transform(val).flatten()
            
        elif col in PPD.scalers:
            val = df_reconstructed[col].values.reshape(-1, 1)
            df_reconstructed[col] = PPD.scalers[col].inverse_transform(val).flatten()
            
        elif col in PPD.label_encoders:
            # Inverse Label Encode
            # Need to round to nearest integer first
            val = df_reconstructed[col].values
            val_int = np.round(val).astype(int)
            
            # Clip to valid range
            le = PPD.label_encoders[col]
            n_classes = len(le.classes_)
            val_int = np.clip(val_int, 0, n_classes - 1)
            
            df_reconstructed[col] = le.inverse_transform(val_int)
            
    return df_reconstructed.iloc[0]

def load_tabdiff_model(tabdiff_dir, dataname='fraud_data', exp_name='quick_fraud', device='cpu'):
    """
    Loads the trained TabDiff model checkpoint.
    """
    if not TABDIFF_AVAILABLE:
        return None, None, None
    
    ckpt_dir = Path(tabdiff_dir) / 'tabdiff' / 'ckpt' / dataname / exp_name
    if not ckpt_dir.exists():
        alt_paths = [
            Path(tabdiff_dir) / 'tabdiff' / 'ckpt' / dataname / 'learnable_schedule',
            Path(tabdiff_dir) / 'tabdiff' / 'ckpt' / dataname
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                ckpt_dir = alt_path
                break
    
    if not ckpt_dir.exists():
        print(f"Warning: TabDiff checkpoint not found at {ckpt_dir}")
        return None, None, None
    
    # Find latest checkpoint
    best_ema_files = list(ckpt_dir.glob('best_ema_model_*.pt'))
    best_model_files = list(ckpt_dir.glob('best_model_*.pt'))
    model_files = list(ckpt_dir.glob('model_*.pt'))
    
    latest_model = None
    if best_ema_files:
        latest_model = max(best_ema_files, key=lambda x: int(x.stem.split('_')[-1]))
    elif best_model_files:
        latest_model = max(best_model_files, key=lambda x: int(x.stem.split('_')[-1]))
    elif model_files:
        latest_model = max(model_files, key=lambda x: int(x.stem.split('_')[-1]))
    
    if latest_model is None:
        print(f"Warning: No TabDiff checkpoint found")
        return None, None, None
    
    # Load config
    config_path = ckpt_dir / 'config.pkl'
    if not config_path.exists():
        print(f"Warning: Config file not found at {config_path}")
        return None, None, None
    
    with open(config_path, 'rb') as f:
        config = pickle.load(f)
    
    # First, try to get dimensions from config (saved when model was trained)
    d_numerical = None
    categories = None
    
    if 'unimodmlp_params' in config and 'd_numerical' in config['unimodmlp_params']:
        d_numerical = config['unimodmlp_params']['d_numerical']
        categories_list = config['unimodmlp_params'].get('categories', [])
        # categories in config are +1 (for padding), so subtract 1
        categories = np.array([c - 1 for c in categories_list])
        print(f"Using dimensions from config: d_numerical={d_numerical}, categories={categories}")
    
    # If not in config, infer from checkpoint
    if d_numerical is None:
        print(f"Loading checkpoint to infer dimensions: {latest_model}")
        state_dicts = torch.load(latest_model, map_location=device)
        
        # Infer dimensions from checkpoint
        # The tokenizer weight shape is [total_features, d_token]
        # total_features = d_numerical + sum(categories)
        checkpoint_total_features = None
        checkpoint_d_token = None
        
        if 'denoise_fn' in state_dicts:
            denoise_fn_state = state_dicts['denoise_fn']
            # Look for tokenizer weight to infer dimensions
            for key in denoise_fn_state.keys():
                if 'tokenizer.weight' in key:
                    checkpoint_total_features = denoise_fn_state[key].shape[0]
                    checkpoint_d_token = denoise_fn_state[key].shape[1]
                    print(f"Checkpoint expects {checkpoint_total_features} total features (d_token={checkpoint_d_token})")
                    break
        
        if checkpoint_total_features is None:
            print("Warning: Could not infer dimensions from checkpoint. Falling back to raw data dimensions.")
            # Fallback to raw data
            data_dir = Path(tabdiff_dir) / 'data' / dataname
            info_path = data_dir / 'info.json'
            with open(info_path, 'r') as f:
                info = json.load(f)
            
            X_num, X_cat, categories, d_numerical, num_inverse, int_inverse, cat_inverse = tabdiff_preprocess(
                str(data_dir), y_only=False, dequant_dist=config['data']['dequant_dist'],
                int_dequant_factor=config['data']['int_dequant_factor'],
                task_type=info['task_type'], inverse=True
            )
            categories = np.array(categories)
            
            # Load checkpoint for weights (already loaded above)
            # state_dicts is already loaded
        else:
            # Infer from checkpoint: for processed data with PCA and binary classification
            # total_features = d_numerical + 2 (target has 2 classes)
            # So: d_numerical = checkpoint_total_features - 2
            d_numerical = checkpoint_total_features - 2  # Subtract target (2 classes for binary)
            categories = np.array([2])  # Binary classification target
            print(f"Inferred from checkpoint: d_numerical={d_numerical}, categories={categories}")
            
            # Create dummy inverse transforms (not used for processed data)
            num_inverse = lambda x: x
            int_inverse = lambda x: x
            cat_inverse = lambda x: x
            
            # Create dummy info
            info = {'task_type': 'binclass'}
    else:
        # Dimensions from config, load checkpoint for weights
        print(f"Loading checkpoint: {latest_model}")
        state_dicts = torch.load(latest_model, map_location=device)
        
        # Create dummy inverse transforms (not used for processed data)
        num_inverse = lambda x: x
        int_inverse = lambda x: x
        cat_inverse = lambda x: x
        
        # Create dummy info
        info = {'task_type': 'binclass'}
    
    # Build model with inferred dimensions
    config['unimodmlp_params']['d_numerical'] = d_numerical
    config['unimodmlp_params']['categories'] = (categories + 1).tolist()
    
    backbone = UniModMLP(**config['unimodmlp_params'])
    model = Model(backbone, **config['diffusion_params']['edm_params'])
    model.to(device)
    
    # Create diffusion
    diffusion = UnifiedCtimeDiffusion(
        num_classes=categories,
        num_numerical_features=d_numerical,
        denoise_fn=model,
        y_only_model=None,
        **config['diffusion_params'],
        device=torch.device(device),
    )
    diffusion.to(device)
    diffusion.eval()
    
    # Load weights (state_dicts already loaded earlier)
    diffusion._denoise_fn.load_state_dict(state_dicts['denoise_fn'])
    if 'num_schedule' in state_dicts:
        diffusion.num_schedule.load_state_dict(state_dicts['num_schedule'])
    if 'cat_schedule' in state_dicts:
        diffusion.cat_schedule.load_state_dict(state_dicts['cat_schedule'])
    
    print(f"Loaded TabDiff model from {latest_model}")
    
    return diffusion, info, (num_inverse, int_inverse, cat_inverse)

def purify_with_tabdiff(sample_tabdiff_space, diffusion, d_numerical, device='cpu', t_purify=0.1, num_steps=5):
    """
    Purifies a sample using TabDiff's reverse diffusion process.
    
    Args:
        sample_tabdiff_space: Sample already in TabDiff space (numpy array: [num_features])
        diffusion: Loaded TabDiff diffusion model
        d_numerical: Number of numerical features
        device: Device to run on
        t_purify: Timestep to start purification from (small = less noise added)
        num_steps: Number of reverse diffusion steps
    
    Returns:
        purified_sample: Purified sample in TabDiff space (numpy array)
    """
    if diffusion is None:
        return None
    
    # Convert to tensor and split num/cat
    sample_tensor = torch.tensor(sample_tabdiff_space).float().unsqueeze(0).to(device)
    x_num = sample_tensor[:, :d_numerical]
    
    # Handle categorical features (may be empty)
    has_cat = sample_tensor.shape[1] > d_numerical
    if has_cat:
        x_cat = sample_tensor[:, d_numerical:].long()
    else:
        x_cat = torch.zeros((1, 0), dtype=torch.long).to(device)
    
    # Add noise (forward process to t_purify)
    t = torch.tensor([t_purify]).to(device)
    sigma_num = diffusion.num_schedule.total_noise(t)
    sigma_cat = diffusion.cat_schedule.total_noise(t)
    
    # Forward process
    noise = torch.randn_like(x_num)
    x_num_t = x_num + noise * sigma_num
    
    if has_cat and len(diffusion.num_classes) > 0:
        move_chance = -torch.expm1(-sigma_cat)
        x_cat_t, _ = diffusion.q_xt(x_cat, move_chance)
    else:
        x_cat_t = x_cat
    
    # Reverse diffusion (purification)
    # Create timestep schedule for reverse
    t_reverse = torch.linspace(t_purify, 0.0, num_steps + 1, device=device)
    
    z_num = x_num_t
    z_cat = x_cat_t
    
    # S_noise is defined in the module (S_noise=1)
    from tabdiff.models.unified_ctime_diffusion import S_noise
    
    for i in range(num_steps):
        t_cur = t_reverse[i]
        t_next = t_reverse[i + 1]
        t_hat = (t_cur + t_next) / 2
        
        sigma_num_cur = diffusion.num_schedule.total_noise(t_cur.unsqueeze(0))
        sigma_num_next = diffusion.num_schedule.total_noise(t_next.unsqueeze(0))
        sigma_num_hat = diffusion.num_schedule.total_noise(t_hat.unsqueeze(0))
        sigma_cat_cur = diffusion.cat_schedule.total_noise(t_cur.unsqueeze(0))
        sigma_cat_next = diffusion.cat_schedule.total_noise(t_next.unsqueeze(0))
        sigma_cat_hat = diffusion.cat_schedule.total_noise(t_hat.unsqueeze(0))
        
        z_num, z_cat, _ = diffusion.edm_update(
            z_num, z_cat, num_steps - i - 1,
            t_cur.unsqueeze(0), t_next.unsqueeze(0), t_hat.unsqueeze(0),
            sigma_num_cur, sigma_num_next, sigma_num_hat,
            sigma_cat_cur, sigma_cat_next, sigma_cat_hat
        )
    
    # Return purified sample
    if has_cat:
        purified = torch.cat([z_num, z_cat.float()], dim=1).detach().cpu().numpy()[0]
    else:
        purified = z_num.detach().cpu().numpy()[0]
    
    return purified

def main():
    print("=" * 60)
    print("GENERATE ADVERSARIAL FRAUD SAMPLES (TabDiff Noise)")
    print("=" * 60)

    # 1. Load and Process Data
    print("\nStep 1: Loading and Processing Data (Replicating Pipeline)...")
    data_dir = Path('data/ieee-fraud-detection')
    
    train_id_df = pd.read_csv(data_dir / 'train_identity.csv')
    train_transaction_df = pd.read_csv(data_dir / 'train_transaction.csv')
    test_id_df = pd.read_csv(data_dir / 'test_identity.csv')
    test_transaction_df = pd.read_csv(data_dir / 'test_transaction.csv')

    dfs_list = [train_id_df, train_transaction_df, test_id_df, test_transaction_df]
    PPD = preprocessDatasets()

    # Remove empty cols
    for df in dfs_list:
        PPD.remove_empty_cols(df)

    # Join ID and transaction
    train_df = PPD.join_ID(train_transaction_df, train_id_df)
    test_df = PPD.join_ID(test_transaction_df, test_id_df)
    
    # Preserve IDs for tracking (since final_preprocessing drops them)
    train_ids = train_df['TransactionID'].copy()
    
    # Store raw validation samples BEFORE preprocessing (for TabDiff purification)
    # We'll need to recreate the validation split later, so store the full train_df before outlier removal
    train_df_raw = train_df.copy()

    # Remove outliers
    print("Removing outliers...")
    train_df = PPD.remove_outliers(train_df, 'TransactionAmt')

    # Feature engineer
    print("Feature engineering...")
    train_df = PPD.feature_engineer(train_df)
    test_df = PPD.feature_engineer(test_df)

    # Replace blanks
    PPD.replace_blanks(train_df)
    PPD.replace_blanks(test_df)

    # Reduce memory
    PPD.reduce_memory(train_df)
    PPD.reduce_memory(test_df)

    # Encode and scale
    print("Encoding and Scaling...")
    train_df = PPD.encode_df(train_df)
    test_df = PPD.encode_df(test_df)

    # Reduce Dimension (PCA)
    print("Applying PCA...")
    RD = ReduceDeminesion(train_df, test_df)
    train_df, test_df = RD.plot_and_reduceD(plots=False)
    
    # Final preprocessing
    print("Final Preprocessing...")
    train_df, test_df = PPD.final_preprocessing(train_df, test_df)

    # Split Data (Replicating detect_fraud.ipynb split logic)
    print("\nStep 2: Splitting Data to get Validation Set (Ground Truth)...")
    X_train = train_df.drop(columns=['isFraud'])
    y_train = train_df['isFraud']
    
    # Note: indices might have shifted due to drops/sorts. 
    # X_train index corresponds to the processed train_df.
    
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    print(f"Validation Set Shape: {X_val_split.shape}")
    print(f"Validation Fraud Cases: {(y_val_split == 1).sum()}")
    
    # Store raw validation samples (before XGBoost preprocessing) for TabDiff purification
    # We need to recreate the split on the raw data
    # Get indices that correspond to validation set
    train_df_raw_processed = train_df_raw.copy()
    train_df_raw_processed = PPD.feature_engineer(train_df_raw_processed)
    PPD.replace_blanks(train_df_raw_processed)
    
    # Recreate the same split on raw data
    train_df_raw_processed = train_df_raw_processed.sort_values(by='TransactionID', ascending=True).reset_index(drop=True)
    X_train_raw = train_df_raw_processed.drop(columns=['isFraud'])
    y_train_raw = train_df_raw_processed['isFraud']
    
    X_train_split_raw, X_val_split_raw, y_train_split_raw, y_val_split_raw = train_test_split(
        X_train_raw, y_train_raw, test_size=0.2, stratify=y_train_raw, random_state=42
    )
    
    # Store mapping from processed indices to raw samples
    val_raw_samples = X_val_split_raw.copy()
    val_raw_labels = y_val_split_raw.copy()

    # 3. Load TabDiff Model for Purification
    print("\nStep 3: Loading TabDiff Model for Purification...")
    tabdiff_dir = Path('TabDiff')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    diffusion, tabdiff_info, tabdiff_inverses = load_tabdiff_model(
        tabdiff_dir, dataname='fraud_data', exp_name='quick_fraud', device=device
    )
    
    if diffusion is None:
        print("Warning: TabDiff model not available. Skipping purification step.")
        purification_enabled = False
    else:
        purification_enabled = True
        print("TabDiff model loaded successfully for purification.")

    # 4. Load XGBoost Model and Align Features
    print("\nStep 3: Loading XGBoost Model and Aligning Features...")
    model_path = get_latest_model()
    if not model_path:
        print("ERROR: No XGBoost model found in xgb_saved/")
        sys.exit(1)
    
    print(f"Loading model from: {model_path}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # Align Features
    # Check feature alignment
    booster = model.get_booster()
    expected_features = booster.feature_names
    
    if expected_features is None:
        # Try feature_names_in_ if available
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_
    
    if expected_features:
        print(f"\nAligning features with model (Expected: {len(expected_features)}, Got: {X_val_split.shape[1]})...")
        
        # Identify missing and extra columns
        current_features = X_val_split.columns.tolist()
        missing_cols = set(expected_features) - set(current_features)
        extra_cols = set(current_features) - set(expected_features)
        
        if missing_cols:
            print(f"Adding {len(missing_cols)} missing columns (filled with NaN): {list(missing_cols)[:5]}...")
            for col in missing_cols:
                X_val_split[col] = np.nan
                
        if extra_cols:
            print(f"Dropping {len(extra_cols)} extra columns: {list(extra_cols)[:5]}...")
            X_val_split.drop(columns=list(extra_cols), inplace=True)
            
        # Reorder columns to match model
        X_val_split = X_val_split[expected_features]
        
        print(f"New shape: {X_val_split.shape}")

    # 5. Select 10% of Fraud Cases
    print("\nStep 5: Selecting 10% of Fraud Cases from Validation Set...")
    fraud_indices = y_val_split[y_val_split == 1].index
    num_to_sample = int(len(fraud_indices) * 0.2)
    selected_indices = np.random.choice(fraud_indices, num_to_sample, replace=False)
    
    print(f"Selected {len(selected_indices)} fraud cases for adversarial attack.")
    
    X_fraud_samples = X_val_split.loc[selected_indices].copy()
    
    # Identify Categorical Columns for TabDiff Noise
    # Intersect tracked cat columns with aligned feature columns
    final_cat_cols = [c for c in PPD.cat_columns if c in X_fraud_samples.columns]
    cat_indices = [X_fraud_samples.columns.get_loc(c) for c in final_cat_cols]
    
    print(f"Identified {len(final_cat_cols)} categorical columns for discrete noise application.")
    
    # Precompute unique values for categorical columns to sample from
    unique_vals_list = [X_val_split[col].unique() for col in final_cat_cols]

    # 6. Attack Loop
    print("\nStep 6: Running Adversarial Attack (TabDiff Noise)...")
    print("Iteratively adding noise until model predicts 'Safe'...")
    print("  - Numerical cols: Gaussian noise (PowerMean schedule)")
    print("  - Categorical cols: Random replacement (Linear schedule)")

    adversarial_results = []
    
    # Progress bar
    try:
        from tqdm import tqdm
        iterator = tqdm(X_fraud_samples.iterrows(), total=len(X_fraud_samples))
    except ImportError:
        iterator = X_fraud_samples.iterrows()

    success_count = 0
    skipped_count = 0
    
    # TabDiff noise schedule parameters
    # Loop t from 0 to 1.0
    t_steps = np.linspace(0.01, 1.0, 100) # 100 steps of increasing noise

    for idx, row in iterator:
        # Get original prediction
        # XGBoost expects 2D array
        sample = row.values.reshape(1, -1)
        sample_flat = row.values # 1D for helper
        
        # Initial prediction
        pred_prob = model.predict_proba(sample)[:, 1][0]
        pred_label = int(pred_prob > 0.5)
        
        if pred_label == 0:
            # Model already thinks it's safe (False Negative or weak fraud)
            skipped_count += 1
            continue
            
        # Attack: Add noise iteratively
        attack_successful = False
        final_sample = None
        final_noise = None
        final_t = 0
        final_sigma = 0
        
        for t in t_steps:
            sigma = get_tabdiff_noise_sigma(t)
            
            # Apply TabDiff Noise
            perturbed_sample_flat = apply_tabdiff_noise(sample_flat, t, sigma, cat_indices, unique_vals_list)
            perturbed_sample = perturbed_sample_flat.reshape(1, -1)
            
            # Check prediction
            new_prob = model.predict_proba(perturbed_sample)[:, 1][0]
            new_label = int(new_prob > 0.5)
            
            if new_label == 0:
                # Success! It predicts Safe
                attack_successful = True
                final_sample = perturbed_sample.flatten()
                final_noise = final_sample - sample.flatten()
                final_t = t
                final_sigma = sigma
                break
        
        if attack_successful:
            success_count += 1
            # Save result
            # We save the new sample features + metadata
            result_row = {
                'Original_Index': idx,
                'Original_Prob': pred_prob,
                'Adversarial_Prob': new_prob,
                'Noise_t': final_t,
                'Noise_Sigma': final_sigma,
                'Noise_L2': np.linalg.norm(final_noise),
            }
            # Add feature columns (New Sample)
            for i, col in enumerate(X_fraud_samples.columns):
                result_row[col] = final_sample[i]
            
            adversarial_results.append(result_row)
    
    print(f"\nAttack Complete.")
    print(f"Total Samples Attempted: {len(selected_indices)}")
    print(f"Skipped (Already Predicted Safe): {skipped_count}")
    print(f"Successfully Attacked: {success_count}")
    print(f"Failed (Could not fool model): {len(selected_indices) - skipped_count - success_count}")

    # 7. TabDiff Purification (if enabled)
    purified_results = []
    if purification_enabled and adversarial_results:
        print("\n" + "=" * 60)
        print("Step 7: TabDiff Purification of Adversarial Samples...")
        print("=" * 60)
        print("Purifying adversarial samples (with added noise) using TabDiff's reverse diffusion...")
        print("Adversarial samples are in processed space (with PCA). TabDiff will purify them directly.")
        print("This makes the adversarial samples more realistic while potentially still fooling the model.")
        
        # Extract adversarial samples from results (they're already in processed feature space with PCA)
        feature_names = X_fraud_samples.columns.tolist()
        adversarial_samples_processed = []
        adversarial_indices = []
        
        print(f"Preparing {len(adversarial_results)} adversarial samples for purification (already in processed/PCA space)...")
        for result_row in adversarial_results:
            orig_idx = result_row['Original_Index']
            # Extract adversarial sample vector (in processed feature space with PC columns)
            sample_vec = [result_row[col] for col in feature_names]
            sample_np = np.array(sample_vec)
            
            # Get the original label
            if orig_idx in y_val_split.index:
                sample_dict = {col: sample_np[i] for i, col in enumerate(feature_names)}
                sample_dict['isFraud'] = y_val_split.loc[orig_idx]
                adversarial_samples_processed.append(sample_dict)
                adversarial_indices.append(orig_idx)
        
        if not adversarial_samples_processed:
            print("Warning: No adversarial samples found for purification.")
        else:
            print(f"Processing {len(adversarial_samples_processed)} adversarial samples for purification...")
            
            # Save samples to temporary directory in TabDiff format
            temp_dir = Path('temp_purify_adversarial')
            temp_dir.mkdir(exist_ok=True)
            
            # Create DataFrame from processed adversarial samples
            adversarial_processed_df = pd.DataFrame(adversarial_samples_processed)
            
            # Save as CSV
            temp_csv = temp_dir / 'fraud_data.csv'
            adversarial_processed_df.to_csv(temp_csv, index=False)
            
            # Create info.json matching TabDiff's format
            num_col_idx = [i for i, col in enumerate(adversarial_processed_df.columns) 
                          if pd.api.types.is_numeric_dtype(adversarial_processed_df[col]) and col != 'isFraud']
            cat_col_idx = [i for i, col in enumerate(adversarial_processed_df.columns) 
                          if not pd.api.types.is_numeric_dtype(adversarial_processed_df[col]) and col != 'isFraud']
            target_col_idx = [adversarial_processed_df.columns.get_loc('isFraud')]
            
            temp_info = {
                "name": "temp_purify_adversarial",
                "task_type": "binclass",
                "header": "infer",
                "column_names": adversarial_processed_df.columns.tolist(),
                "num_col_idx": num_col_idx,
                "cat_col_idx": cat_col_idx,
                "target_col_idx": target_col_idx,
                "file_type": "csv",
                "data_path": "temp_purify_adversarial/fraud_data.csv",
            }
            
            temp_info_path = temp_dir / 'info.json'
            with open(temp_info_path, 'w') as f:
                json.dump(temp_info, f, indent=4)
            
            try:
                # Preprocess for TabDiff - manually create Dataset to avoid bug in dataset_from_csv
                T_dict = {
                    'normalization': "quantile",
                    'num_nan_policy': 'mean',
                    'cat_nan_policy': None,
                    'cat_min_frequency': None,
                    'cat_encoding': None,
                    'y_policy': "default",
                    'dequant_dist': 'none',
                    'int_dequant_factor': 0.0,
                }
                T = src.Transformations(**T_dict)
                
                # Get categorical features (columns that are not numeric and not target)
                cat_features = [col for col in adversarial_processed_df.columns 
                               if col != 'isFraud' and not pd.api.types.is_numeric_dtype(adversarial_processed_df[col])]
                target = 'isFraud'
                
                # Manually create Dataset (workaround for bug in dataset_from_csv)
                y = {}
                X_num = {}
                X_cat = {} if len(cat_features) > 0 else None
                
                for split in ['train', 'test']:
                    df = pd.read_csv(temp_csv)
                    y[split] = df[target].to_numpy().astype(float)
                    if X_cat is not None:
                        X_cat[split] = df[cat_features].to_numpy().astype(str)
                    X_num[split] = df.drop(cat_features + [target], axis=1).to_numpy().astype(float)
                
                # Create Dataset with correct parameters (fixing TabDiff bug)
                from src.util import TaskType
                n_classes = len(np.unique(y['train']))
                dataset = src.Dataset(
                    X_num, 
                    X_cat, 
                    y, 
                    {},  # int_col_idx_wrt_num
                    None,  # y_info
                    TaskType.BINCLASS,  # task_type
                    n_classes  # n_classes
                )
                
                # Transform dataset (TabDiff's quantile normalization)
                dataset = src.transform_dataset(dataset, T, None)
                
                # Extract processed data
                X_num_processed = dataset.X_num['train']
                X_cat_processed = dataset.X_cat['train'] if dataset.X_cat is not None else None
                y_processed = dataset.y['train']
                d_numerical = X_num_processed.shape[1]
                
                # TabDiff concatenates target to categorical features for classification
                if X_cat_processed is not None:
                    X_cat_with_target = np.concatenate([y_processed.reshape(-1, 1), X_cat_processed], axis=1)
                    categories = src.get_categories({'train': X_cat_with_target})
                else:
                    X_cat_with_target = y_processed.reshape(-1, 1)
                    categories = np.array([len(np.unique(y_processed))])
                
                # Store inverse transforms for converting back to processed space
                num_inverse = dataset.num_transform.inverse_transform if dataset.num_transform is not None else lambda x: x
                cat_inverse = dataset.cat_transform.inverse_transform if dataset.cat_transform is not None else lambda x: x
                
                # Purify each adversarial sample
                purified_count = 0
                failed_purification = 0
                
                try:
                    from tqdm import tqdm
                    iterator = tqdm(range(len(adversarial_samples_processed)), desc="Purifying adversarial samples")
                except ImportError:
                    iterator = range(len(adversarial_samples_processed))
                
                for i in iterator:
                    orig_idx = adversarial_indices[i]
                    result_row = next(r for r in adversarial_results if r['Original_Index'] == orig_idx)
                    
                    try:
                        # Convert to TabDiff space (with target concatenated to categorical)
                        sample_num = X_num_processed[i]
                        if X_cat_with_target is not None:
                            sample_cat = X_cat_with_target[i].astype(float)
                            sample_tabdiff = np.concatenate([sample_num, sample_cat])
                        else:
                            sample_tabdiff = sample_num
                        
                        # Purify using TabDiff reverse diffusion
                        purified_tabdiff = purify_with_tabdiff(
                            sample_tabdiff, diffusion, d_numerical, 
                            device=device, t_purify=0.1, num_steps=5
                        )
                        
                        if purified_tabdiff is not None:
                            # Convert purified sample back to processed space format
                            # Split num and cat (cat includes target as first element)
                            purified_num = purified_tabdiff[:d_numerical]
                            purified_cat_with_target = purified_tabdiff[d_numerical:]
                            
                            # Inverse transform numerical features to get back to processed space
                            try:
                                purified_num_processed = num_inverse(purified_num.reshape(1, -1))[0]
                            except Exception:
                                # If inverse transform fails, use purified values directly
                                purified_num_processed = purified_num
                            
                            # Extract target and categorical
                            if len(purified_cat_with_target) > 0:
                                purified_target = purified_cat_with_target[0]
                                if len(purified_cat_with_target) > 1:
                                    purified_cat = purified_cat_with_target[1:]
                                    try:
                                        if cat_inverse is not None and len(purified_cat) > 0:
                                            # Categorical inverse transform
                                            purified_cat_processed = cat_inverse(purified_cat.reshape(1, -1))[0]
                                        else:
                                            purified_cat_processed = purified_cat
                                    except Exception:
                                        purified_cat_processed = purified_cat
                                else:
                                    purified_cat_processed = np.array([])
                            else:
                                purified_target = 1.0  # Default to fraud
                                purified_cat_processed = np.array([])
                            
                            # Reconstruct sample in processed space (with PC columns)
                            # Map back to feature names - use the order from adversarial_processed_df
                            purified_sample_processed = {}
                            
                            # Get the order of features as they appear in the DataFrame
                            num_feature_names = [col for col in adversarial_processed_df.columns 
                                               if col != 'isFraud' and col not in cat_features]
                            
                            # Map numerical features back
                            for j, col in enumerate(num_feature_names):
                                if j < len(purified_num_processed) and pd.notna(purified_num_processed[j]):
                                    purified_sample_processed[col] = float(purified_num_processed[j])
                                else:
                                    # Fallback to original adversarial value
                                    purified_sample_processed[col] = result_row.get(col, 0.0)
                            
                            # Map categorical features back
                            if len(purified_cat_processed) > 0:
                                for j, col in enumerate(cat_features):
                                    if j < len(purified_cat_processed) and pd.notna(purified_cat_processed[j]):
                                        purified_sample_processed[col] = purified_cat_processed[j]
                                    else:
                                        # Fallback to original adversarial value
                                        purified_sample_processed[col] = result_row.get(col, 0)
                            
                            # Add target
                            purified_sample_processed['isFraud'] = int(round(purified_target))
                            
                            purified_count += 1
                            purified_row = result_row.copy()
                            purified_row['Purified'] = True
                            purified_row['Purification_Success'] = True
                            
                            # Store purified sample in processed space (with PC columns)
                            for col in feature_names:
                                if col in purified_sample_processed:
                                    purified_row[f'Purified_{col}'] = purified_sample_processed[col]
                            
                            purified_row['Purified_isFraud'] = purified_sample_processed.get('isFraud', 1)
                            purified_results.append(purified_row)
                        else:
                            failed_purification += 1
                            
                    except Exception as e:
                        print(f"  Error purifying adversarial sample {orig_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        failed_purification += 1
                
                print(f"\nPurification Complete:")
                print(f"  Successfully Purified: {purified_count} adversarial samples")
                print(f"  Failed: {failed_purification}")
                print(f"\nNote: Purified samples are stored in processed space (with PC columns).")
                print(f"      They can be directly evaluated with XGBoost model.")
                
            except Exception as e:
                print(f"Error in TabDiff preprocessing: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # Cleanup
                import shutil
                if temp_dir.exists():
                    shutil.rmtree(temp_dir)
    
    # 8. Save Results
    if adversarial_results:
        results_df = pd.DataFrame(adversarial_results)
        output_path = 'adversarial_samples.csv'
        results_df.to_csv(output_path, index=False)
        print(f"\nSaved adversarial samples to {output_path}")
        
        # Map back to input space
        print("\nMapping adversarial samples back to Input Space (Inverse Transform)...")
        print("Note: This reverses scaling and encoding to show human-readable values.")
        
        raw_results = []
        # Iterate over successful attacks
        # We need feature names
        feature_names = X_fraud_samples.columns.tolist()
        
        for row in adversarial_results:
            # Reconstruct sample vector from result_row
            # It contains 'Original_Index', etc., and then feature columns
            sample_vec = [row[col] for col in feature_names]
            sample_np = np.array(sample_vec)
            
            # Inverse transform
            try:
                raw_sample = inverse_transform_sample(sample_np, feature_names, PPD, RD)
                # Add metadata
                raw_row = {k: v for k, v in row.items() if k not in feature_names}
                # Add raw features
                for col, val in raw_sample.items():
                    raw_row[col] = val
                raw_results.append(raw_row)
            except Exception as e:
                # PCA inverse transform might fail if dimensions mismatch (truncated)
                pass

        if raw_results:
            raw_df = pd.DataFrame(raw_results)
            raw_output_path = 'adversarial_samples_input_space.csv'
            raw_df.to_csv(raw_output_path, index=False)
            print(f"Saved input-space samples to {raw_output_path}")
        
        # Save purified results if available
        if purified_results:
            purified_df = pd.DataFrame(purified_results)
            purified_output_path = 'adversarial_samples_purified.csv'
            purified_df.to_csv(purified_output_path, index=False)
            print(f"Saved purified samples (with metadata) to {purified_output_path}")
            
            # Also create a clean CSV with purified samples in processed space (ready for XGBoost)
            purified_samples_clean = []
            for row in purified_results:
                clean_sample = {}
                # Extract purified feature values
                for col in feature_names:
                    purified_col = f'Purified_{col}'
                    if purified_col in row:
                        clean_sample[col] = row[purified_col]
                    else:
                        # Fallback to original adversarial value if purification failed for this feature
                        clean_sample[col] = row.get(col, 0.0)
                # Add label
                clean_sample['isFraud'] = row.get('Purified_isFraud', row.get('isFraud', 1))
                purified_samples_clean.append(clean_sample)
            
            if purified_samples_clean:
                purified_clean_df = pd.DataFrame(purified_samples_clean)
                purified_clean_output_path = 'adversarial_samples_purified_clean.csv'
                purified_clean_df.to_csv(purified_clean_output_path, index=False)
                print(f"Saved purified samples (clean format, ready for XGBoost) to {purified_clean_output_path}")
                print(f"  Shape: {purified_clean_df.shape}")
                print(f"  Columns: {list(purified_clean_df.columns[:5])}... (includes PC columns)")
    else:
        print("\nNo adversarial samples generated.")

if __name__ == "__main__":
    main()

