#!/usr/bin/env python3
"""
Script to merge train_identity.csv and train_transaction.csv, apply FULL preprocessing pipeline
(including PCA) from detect_fraud.ipynb, filter fraud cases from processed dataset, and train TabDiff
to synthetically generate more fraud samples in PCA space.

This script:
1. Applies the EXACT preprocessing pipeline from detect_fraud.ipynb:
   - Remove empty columns (>90% missing)
   - Join ID and transaction datasets
   - Remove outliers (>3sigma in TransactionAmt)
   - Feature engineering
   - Replace blanks with -999
   - Reduce memory usage
   - Encode/scale (StandardScaler for numeric, MinMaxScaler for D columns, LabelEncoder for categorical)
   - PCA dimensionality reduction (V columns → PC columns)
   - Final preprocessing (drop columns, align columns)
2. Extracts fraud cases from PROCESSED dataset (after PCA)
3. Trains TabDiff on processed fraud cases (in PCA space)
4. Generates synthetic fraud cases in PCA space

Requirements:
- TabDiff must be installed in the TabDiff/ directory
- Required Python packages: pandas, numpy, sklearn, tomli, tomli-w
- Works with any Python environment (conda, venv, or system Python)

Usage:
    python generate_tabdiff_fraud.py
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
import re
from datetime import datetime, timedelta
from scipy import stats

# Check for required packages
try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
    from sklearn.decomposition import PCA
except ImportError as e:
    print(f"ERROR: Missing required package: {e}")
    print("Please install required packages: pip install pandas numpy scikit-learn")
    sys.exit(1)

# Add TabDiff to path
tabdiff_path = os.path.join(os.path.dirname(__file__), 'TabDiff')
if os.path.exists(tabdiff_path):
    sys.path.insert(0, tabdiff_path)
else:
    print(f"WARNING: TabDiff directory not found at {tabdiff_path}")
    print("Make sure TabDiff is installed in the TabDiff/ directory")

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
                    valid_classes = set(le.classes_)
                    values = df[col].astype(str).values
                    fallback_val = le.classes_[0]
                    safe_values = [x if x in valid_classes else fallback_val for x in values]
                    df[col] = le.transform(safe_values)

        return df
  
    def remove_outliers(self, data: pd.DataFrame, column: str):
        '''
        Remove outliers in transaction amount col that are >3sigma based on a single column.
        '''
        z_scores = stats.zscore(data[column])
        non_outliers = np.abs(z_scores) < 3
        original_len = len(data)
        data = data[non_outliers]
        final_len = len(data)
        print(f'{original_len - final_len} values were removed, since they contained outliers (z-score over 3)')
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

        data = matchPatterns(data, browser_patterns, 'browser')
        data = matchPatterns(data, device_patterns, 'device_name')

        # Modify date format to split day, month, etc.
        start_date = datetime(2017, 11, 30)
        data['TransactionFullDate'] = data['TransactionDT'].apply(lambda x: start_date + timedelta(seconds=x))
        data['TransactionDate'] = data['TransactionFullDate'].dt.date
        data['DayOfWeek'] = data['TransactionFullDate'].dt.dayofweek.apply(lambda x: (x + 1) % 7)
        data['HourOfDay'] = data['TransactionFullDate'].dt.hour
        data['Month'] = data['TransactionFullDate'].dt.month
        return data
  
    def reduce_memory(self, df: pd.DataFrame):
        '''
        Reduce memory by analyzing max/min value in cols and minimizing column data type
        '''
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

class ReduceDeminesion():
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

def prepare_tabdiff_data(fraud_df, output_dir, dataname='fraud_data'):
    """
    Prepare data for TabDiff by creating the required directory structure
    and info.json file.
    """
    # Create directories
    data_dir = Path(output_dir) / 'data' / dataname
    info_dir = Path(output_dir) / 'data' / 'Info'
    data_dir.mkdir(parents=True, exist_ok=True)
    info_dir.mkdir(parents=True, exist_ok=True)
    
    # Save fraud data
    fraud_df.to_csv(data_dir / f'{dataname}.csv', index=False)
    
    # Determine column types
    num_col_idx = []
    cat_col_idx = []
    target_col_idx = []
    
    column_names = fraud_df.columns.tolist()
    
    for idx, col in enumerate(column_names):
        if col == 'isFraud':
            target_col_idx.append(idx)
        elif pd.api.types.is_numeric_dtype(fraud_df[col]):
            num_col_idx.append(idx)
        else:
            cat_col_idx.append(idx)
    
    # Create info.json
    info = {
        "name": dataname,
        "task_type": "binclass",
        "header": "infer",
        "column_names": column_names,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "file_type": "csv",
        "data_path": f"data/{dataname}/{dataname}.csv",
        "val_path": None,
        "test_path": None
    }
    
    info_path = info_dir / f'{dataname}.json'
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=4)
    
    print(f"Created info.json at {info_path}")
    print(f"Numerical columns: {len(num_col_idx)}, Categorical columns: {len(cat_col_idx)}, Target: {len(target_col_idx)}")
    
    return info_path, data_dir

def update_tabdiff_config(config_path, steps=1, checkpoint_freq=250, epochs=None):
    """Update TabDiff config for training with specified steps and checkpoint frequency."""
    try:
        import tomli_w
        import tomli
    except ImportError:
        print("Warning: tomli not found, trying toml...")
        try:
            import toml
            with open(config_path, 'r') as f:
                config = toml.load(f)
            
            config['train']['main']['steps'] = steps
            config['train']['main']['check_val_every'] = checkpoint_freq
            config['train']['main']['batch_size'] = 512
            config['diffusion_params']['num_timesteps'] = 5
            
            if 'sample' not in config:
                config['sample'] = {}
            config['sample']['batch_size'] = 256
            
            with open(config_path, 'w') as f:
                toml.dump(config, f)
            
            print(f"Updated config: steps={steps}, check_val_every={checkpoint_freq}, batch_size=512, sample_batch_size=256, num_timesteps=5")
            return True
        except ImportError:
            print("ERROR: Neither tomli nor toml found. Please install tomli: pip install tomli tomli-w")
            return False
    
    # Use tomli (binary read, binary write)
    with open(config_path, 'rb') as f:
        config = tomli.load(f)
    
    config['train']['main']['steps'] = steps
    config['train']['main']['check_val_every'] = checkpoint_freq
    config['train']['main']['batch_size'] = 512
    
    config['diffusion_params']['num_timesteps'] = 5
    
    if 'sample' not in config:
        config['sample'] = {}
    config['sample']['batch_size'] = 256
    
    with open(config_path, 'wb') as f:
        tomli_w.dump(config, f)
    
    print(f"Updated config: steps={steps}, check_val_every={checkpoint_freq}, batch_size=512, sample_batch_size=256, num_timesteps=5")
    return True

def main():
    # Paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data' / 'ieee-fraud-detection'
    tabdiff_dir = base_dir / 'TabDiff'
    output_dir = base_dir
    
    print("=" * 80)
    print("GENERATE TABDIFF FRAUD SAMPLES (WITH FULL PREPROCESSING PIPELINE)")
    print("=" * 80)
    
    # Step 1: Load and merge datasets
    print("\nStep 1: Loading and merging datasets...")
    train_id_df = pd.read_csv(data_dir / 'train_identity.csv')
    train_transaction_df = pd.read_csv(data_dir / 'train_transaction.csv')
    test_id_df = pd.read_csv(data_dir / 'test_identity.csv')
    test_transaction_df = pd.read_csv(data_dir / 'test_transaction.csv')
    
    dfs_list = [train_id_df, train_transaction_df, test_id_df, test_transaction_df]
    PPD = preprocessDatasets()
    
    # Step 2: Remove empty columns
    print("\nStep 2: Removing empty columns...")
    for df in dfs_list:
        PPD.remove_empty_cols(df)
    
    # Step 3: Join ID and transaction datasets
    print("\nStep 3: Joining ID and transaction datasets...")
    train_df = PPD.join_ID(train_transaction_df, train_id_df)
    test_df = PPD.join_ID(test_transaction_df, test_id_df)
    print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")
    
    # Step 4: Remove outliers (train only)
    print("\nStep 4: Removing outliers from training data...")
    train_df = PPD.remove_outliers(train_df, 'TransactionAmt')
    
    # Step 5: Feature engineering
    print("\nStep 5: Feature engineering...")
    train_df = PPD.feature_engineer(train_df)
    test_df = PPD.feature_engineer(test_df)
    
    # Step 6: Replace blanks
    print("\nStep 6: Replacing blanks...")
    PPD.replace_blanks(train_df)
    PPD.replace_blanks(test_df)
    
    # Step 7: Reduce memory
    print("\nStep 7: Reducing memory usage...")
    PPD.reduce_memory(train_df)
    PPD.reduce_memory(test_df)
    
    # Step 8: Encode and scale
    print("\nStep 8: Encoding and scaling...")
    train_df = PPD.encode_df(train_df)
    test_df = PPD.encode_df(test_df)
    
    # Step 9: Apply PCA dimensionality reduction
    print("\nStep 9: Applying PCA dimensionality reduction...")
    RD = ReduceDeminesion(train_df, test_df)
    train_df, test_df = RD.plot_and_reduceD(plots=False)
    
    # Step 10: Final preprocessing
    print("\nStep 10: Final preprocessing...")
    train_df, test_df = PPD.final_preprocessing(train_df, test_df)
    
    print(f"\nFinal processed train shape: {train_df.shape}")
    print(f"Final processed test shape: {test_df.shape}")
    
    # Step 11: Extract fraud cases from PROCESSED dataset (after PCA)
    print("\nStep 11: Extracting fraud cases from processed dataset (after PCA)...")
    fraud_df = train_df[train_df['isFraud'] == 1].copy()
    print(f"Number of fraud cases after preprocessing: {len(fraud_df)}")
    print(f"Fraud cases shape: {fraud_df.shape}")
    
    if len(fraud_df) == 0:
        print("ERROR: No fraud cases found after preprocessing!")
        return
    
    # Check if we have PC columns (PCA was applied)
    pc_cols = [col for col in fraud_df.columns if col.startswith('PC')]
    print(f"PCA columns found: {len(pc_cols)} (PC1-PC{len(pc_cols)})")
    
    # Step 12: Prepare data for TabDiff
    print("\nStep 12: Preparing processed fraud data for TabDiff...")
    info_path, tabdiff_data_dir = prepare_tabdiff_data(
        fraud_df, 
        tabdiff_dir,
        dataname='fraud_data'
    )
    
    # Step 13: Process dataset using TabDiff's process_dataset.py
    print("\nStep 13: Processing dataset for TabDiff...")
    original_cwd = os.getcwd()
    try:
        os.chdir(tabdiff_dir)
        python_exe = sys.executable
        result = subprocess.run(
            [python_exe, 'process_dataset.py', '--dataname', 'fraud_data'],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"Error processing dataset: {result.stderr}")
            if result.stdout:
                print(f"stdout: {result.stdout}")
            return
        print("Dataset processed successfully")
    finally:
        os.chdir(original_cwd)
    
    # Step 14: Update TabDiff config for training with 500 steps and checkpoints every 250 steps
    print("\nStep 14: Updating TabDiff config for training (500 steps, checkpoints every 250 steps)...")
    config_path = tabdiff_dir / 'tabdiff' / 'configs' / 'tabdiff_configs.toml'
    if not update_tabdiff_config(config_path, steps=500, checkpoint_freq=250):
        print("Warning: Could not update config file, continuing with default settings...")
    
    # Step 15: Train TabDiff
    print("\nStep 15: Training TabDiff on processed fraud cases (PCA space)...")
    print("=" * 60)
    original_cwd = os.getcwd()
    python_exe = sys.executable
    
    env = os.environ.copy()
    env['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    
    try:
        os.chdir(tabdiff_dir)
        process = subprocess.Popen(
            [python_exe, 'main.py', '--dataname', 'fraud_data', '--mode', 'train', '--no_wandb', '--exp_name', 'quick_fraud'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )
        
        for line in process.stdout:
            print(line, end='', flush=True)
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\nERROR: TabDiff training failed with return code {process.returncode}")
            return
        print("\n" + "=" * 60)
        print("TabDiff training completed successfully!")
    finally:
        os.chdir(original_cwd)
    
    # Step 16: Generate synthetic fraud cases
    print("\nStep 16: Generating 1000 synthetic fraud cases in PCA space...")
    ckpt_dir = tabdiff_dir / 'tabdiff' / 'ckpt' / 'fraud_data' / 'quick_fraud'
    if not ckpt_dir.exists():
        alt_paths = [
            tabdiff_dir / 'tabdiff' / 'ckpt' / 'fraud_data' / 'learnable_schedule',
            tabdiff_dir / 'tabdiff' / 'ckpt' / 'fraud_data'
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                ckpt_dir = alt_path
                break
    
    if not ckpt_dir.exists():
        print(f"ERROR: Checkpoint directory not found.")
        return
    
    best_ema_files = list(ckpt_dir.glob('best_ema_model_*.pt'))
    best_model_files = list(ckpt_dir.glob('best_model_*.pt'))
    model_files = list(ckpt_dir.glob('model_*.pt'))
    
    latest_model = None
    if best_ema_files:
        latest_model = max(best_ema_files, key=lambda x: int(x.stem.split('_')[-1]))
        print(f"Using best EMA checkpoint: {latest_model}")
    elif best_model_files:
        latest_model = max(best_model_files, key=lambda x: int(x.stem.split('_')[-1]))
        print(f"Using best model checkpoint: {latest_model}")
    elif model_files:
        latest_model = max(model_files, key=lambda x: int(x.stem.split('_')[-1]))
        print(f"Using model checkpoint: {latest_model}")
    else:
        print(f"ERROR: No compatible model checkpoint found")
        return
    
    python_exe = sys.executable
    print("=" * 60)
    
    env = os.environ.copy()
    env['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    
    process = subprocess.Popen(
        [
            python_exe, 'main.py',
            '--dataname', 'fraud_data',
            '--mode', 'test',
            '--no_wandb',
            '--exp_name', 'quick_fraud',
            '--ckpt_path', str(latest_model),
            '--num_samples_to_generate', '1000'
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
        cwd=str(tabdiff_dir),
        env=env
    )
    
    for line in process.stdout:
        print(line, end='', flush=True)
    
    process.wait()
    
    if process.returncode != 0:
        print(f"\nERROR: Synthetic data generation failed with return code {process.returncode}")
        return
    print("\n" + "=" * 60)
    print("Synthetic data generation completed successfully!")
    
    # Step 17: Load synthetic data
    print("\nStep 17: Loading synthetic fraud cases (in PCA space)...")
    result_dirs = [
        tabdiff_dir / 'tabdiff' / 'result' / 'fraud_data' / 'quick_fraud',
        tabdiff_dir / 'tabdiff' / 'result' / 'fraud_data' / 'learnable_schedule',
        tabdiff_dir / 'tabdiff' / 'result' / 'fraud_data'
    ]
    
    syn_file = None
    for result_dir in result_dirs:
        if result_dir.exists():
            # Look for samples.csv in subdirectories (epoch directories)
            for epoch_dir in result_dir.glob('*'):
                if epoch_dir.is_dir():
                    ema_samples = epoch_dir / 'ema' / 'samples.csv'
                    regular_samples = epoch_dir / 'samples.csv'
                    if ema_samples.exists():
                        syn_file = ema_samples
                        break
                    elif regular_samples.exists():
                        syn_file = regular_samples
                        break
            if syn_file:
                break
    
    if syn_file is None:
        print(f"ERROR: No synthetic data files found.")
        return
    
    print(f"Loading synthetic data from: {syn_file}")
    syn_df = pd.read_csv(syn_file)
    print(f"Synthetic data shape: {syn_df.shape}")
    print(f"Fraud cases in synthetic data: {(syn_df['isFraud'] == 1).sum()}")
    
    # Step 18: Save synthetic data (already in PCA space)
    output_path = output_dir / 'data' / 'synthetic' / 'tabdiff_synthetic_fraud_pca_1000.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    syn_df.to_csv(output_path, index=False)
    print(f"\n✓ Successfully generated 1000 synthetic fraud cases in PCA space!")
    print(f"Saved to: {output_path}")
    print(f"Final synthetic data shape: {syn_df.shape}")
    print(f"\nNote: Synthetic samples are in PCA space (with PC columns), matching the processed feature space.")
    print(f"      They can be directly used with the XGBoost model trained on processed data.")

if __name__ == '__main__':
    main()
