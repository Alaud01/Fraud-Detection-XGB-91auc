#!/usr/bin/env python3
"""
Script to merge train_identity.csv and train_transaction.csv, filter fraud cases,
apply label encoding, train TabDiff, and generate 1000 synthetic fraud cases.

This script:
1. Merges train_identity.csv and train_transaction.csv on TransactionID
2. Filters fraud cases (isFraud == 1)
3. Applies label encoding to categorical columns (same as detect_fraud.ipynb)
4. Prepares data for TabDiff format
5. Trains TabDiff with minimal steps/epochs for quick training on M4 Mac
6. Generates 1000 synthetic fraud cases
7. Applies inverse label encoding to restore categorical values

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

# Check for required packages
try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import LabelEncoder
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

def join_ID(transaction_df, ID_df):
    """Join datasets based on TransactionID."""
    merged_df = pd.merge(transaction_df, ID_df, on='TransactionID', how='outer')
    col_list = []
    for col in merged_df.columns:
        if '-' in col:
            col = col.replace('-', '_')
        col_list.append(col)
    merged_df.columns = col_list
    return merged_df

def remove_empty_cols(data, threshold=0.90):
    """Drop columns if more than threshold% is missing."""
    data.dropna(thresh=int((1 - threshold) * len(data)), axis=1, inplace=True)
    return data

def replace_blanks(df):
    """Replace blanks with -999."""
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col].fillna(-999, inplace=True)
        else:
            df[col].fillna('-999', inplace=True)
    return df

def encode_df(df, label_encoders=None):
    """
    Encode dataframe columns using LabelEncoder for categorical columns.
    Returns encoded dataframe and dictionary of label encoders for inverse transform.
    """
    if label_encoders is None:
        label_encoders = {}
    
    for col in df.columns:
        if col == 'TransactionID' or col == 'isFraud':
            continue
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Categorical column - use label encoding
            if col not in label_encoders:
                label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen values during inverse transform
                unique_values = df[col].astype(str).unique()
                known_classes = set(label_encoders[col].classes_)
                for val in unique_values:
                    if val not in known_classes:
                        # Add to encoder
                        label_encoders[col].classes_ = np.append(label_encoders[col].classes_, val)
                df[col] = label_encoders[col].transform(df[col].astype(str))
    
    return df, label_encoders

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

def update_tabdiff_config(config_path, steps=50, epochs=None):
    """Update TabDiff config for quick training."""
    try:
        import tomli_w
        import tomli
    except ImportError:
        print("Warning: tomli not found, trying toml...")
        try:
            import toml
            # Fallback to toml if tomli not available
            with open(config_path, 'r') as f:
                config = toml.load(f)
            
            config['train']['main']['steps'] = steps
            config['train']['main']['check_val_every'] = 10
            config['train']['main']['batch_size'] = 512
            config['diffusion_params']['num_timesteps'] = 5
            
            # Reduce sample batch size to avoid CUDA OOM
            if 'sample' not in config:
                config['sample'] = {}
            config['sample']['batch_size'] = 256
            
            with open(config_path, 'w') as f:
                toml.dump(config, f)
            
            print(f"Updated config: steps={steps}, batch_size=512, sample_batch_size=256, num_timesteps=5")
            return True
        except ImportError:
            print("ERROR: Neither tomli nor toml found. Please install tomli: pip install tomli tomli-w")
            return False
    
    # Use tomli (binary read, binary write)
    with open(config_path, 'rb') as f:
        config = tomli.load(f)
    
    # Minimize training steps for quick training
    config['train']['main']['steps'] = steps
    config['train']['main']['check_val_every'] = 10
    config['train']['main']['batch_size'] = 512  # Smaller batch for M4 Mac
    
    # Reduce diffusion timesteps
    config['diffusion_params']['num_timesteps'] = 5
    
    # Reduce sample batch size to avoid CUDA OOM during evaluation/sampling
    if 'sample' not in config:
        config['sample'] = {}
    config['sample']['batch_size'] = 256  # Much smaller for sampling to avoid OOM
    
    with open(config_path, 'wb') as f:
        tomli_w.dump(config, f)
    
    print(f"Updated config: steps={steps}, batch_size=512, sample_batch_size=256, num_timesteps=5")
    return True

def main():
    # Paths
    base_dir = Path(__file__).parent
    data_dir = base_dir / 'data' / 'ieee-fraud-detection'
    tabdiff_dir = base_dir / 'TabDiff'
    output_dir = base_dir
    
    # Step 1: Load and merge datasets
    print("Step 1: Loading and merging datasets...")
    train_id_df = pd.read_csv(data_dir / 'train_identity.csv')
    train_transaction_df = pd.read_csv(data_dir / 'train_transaction.csv')
    
    # Merge datasets
    merged_df = join_ID(train_transaction_df, train_id_df)
    print(f"Merged dataset shape: {merged_df.shape}")
    
    # Step 2: Remove empty columns
    print("\nStep 2: Removing empty columns...")
    merged_df = remove_empty_cols(merged_df)
    print(f"After removing empty columns: {merged_df.shape}")
    
    # Step 3: Replace blanks
    print("\nStep 3: Replacing blanks...")
    merged_df = replace_blanks(merged_df)
    
    # Step 4: Filter fraud cases
    print("\nStep 4: Filtering fraud cases...")
    fraud_df = merged_df[merged_df['isFraud'] == 1].copy()
    print(f"Number of fraud cases: {len(fraud_df)}")
    
    if len(fraud_df) == 0:
        print("ERROR: No fraud cases found!")
        return
    
    # Step 5: Apply label encoding (same as detect_fraud.ipynb)
    print("\nStep 5: Applying label encoding...")
    fraud_df_encoded, label_encoders = encode_df(fraud_df.copy())
    
    # Save label encoders for inverse transform
    encoders_path = output_dir / 'label_encoders.json'
    encoders_dict = {}
    for col, encoder in label_encoders.items():
        encoders_dict[col] = {
            'classes': encoder.classes_.tolist()
        }
    with open(encoders_path, 'w') as f:
        json.dump(encoders_dict, f, indent=2)
    print(f"Saved label encoders to {encoders_path}")
    
    # Step 6: Prepare data for TabDiff
    print("\nStep 6: Preparing data for TabDiff...")
    info_path, tabdiff_data_dir = prepare_tabdiff_data(
        fraud_df_encoded, 
        tabdiff_dir,
        dataname='fraud_data'
    )
    
    # Step 7: Process dataset using TabDiff's process_dataset.py
    print("\nStep 7: Processing dataset for TabDiff...")
    original_cwd = os.getcwd()
    try:
        os.chdir(tabdiff_dir)
        # Use sys.executable to use the same Python interpreter
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
    
    # Step 8: Update TabDiff config for quick training
    print("\nStep 8: Updating TabDiff config for quick training...")
    config_path = tabdiff_dir / 'tabdiff' / 'configs' / 'tabdiff_configs.toml'
    if not update_tabdiff_config(config_path, steps=50):
        print("Warning: Could not update config file, continuing with default settings...")
    
    # Step 9: Train TabDiff
    print("\nStep 9: Training TabDiff (this may take a while)...")
    print("=" * 60)
    original_cwd = os.getcwd()
    python_exe = sys.executable
    
    # Set CUDA memory allocation config to reduce fragmentation
    env = os.environ.copy()
    env['PYTORCH_ALLOC_CONF'] = 'expandable_segments:True'
    
    try:
        os.chdir(tabdiff_dir)
        # Run with real-time output streaming
        process = subprocess.Popen(
            [python_exe, 'main.py', '--dataname', 'fraud_data', '--mode', 'train', '--no_wandb', '--exp_name', 'quick_fraud'],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
            env=env
        )
        
        # Stream output in real-time
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
    
    # Step 10: Generate synthetic fraud cases
    print("\nStep 10: Generating 1000 synthetic fraud cases...")
    # Find the latest checkpoint
    ckpt_dir = tabdiff_dir / 'tabdiff' / 'ckpt' / 'fraud_data' / 'quick_fraud'
    if not ckpt_dir.exists():
        # Try alternative paths
        alt_paths = [
            tabdiff_dir / 'tabdiff' / 'ckpt' / 'fraud_data' / 'learnable_schedule',
            tabdiff_dir / 'tabdiff' / 'ckpt' / 'fraud_data'
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                ckpt_dir = alt_path
                break
    
    if not ckpt_dir.exists():
        print(f"ERROR: Checkpoint directory not found. Searched:")
        print(f"  - {tabdiff_dir / 'tabdiff' / 'ckpt' / 'fraud_data' / 'quick_fraud'}")
        print(f"  - {tabdiff_dir / 'tabdiff' / 'ckpt' / 'fraud_data' / 'learnable_schedule'}")
        return
    
    # Find latest model checkpoint
    # Priority: best_ema_model > best_model > model > ema_model
    # Note: ema_model_*.pt files are saved as raw state_dict, not wrapped dict, so they won't work
    best_ema_files = list(ckpt_dir.glob('best_ema_model_*.pt'))
    best_model_files = list(ckpt_dir.glob('best_model_*.pt'))
    model_files = list(ckpt_dir.glob('model_*.pt'))
    
    latest_model = None
    if best_ema_files:
        # Extract epoch number from filename like "best_ema_model_0.1234_50.pt"
        latest_model = max(best_ema_files, key=lambda x: int(x.stem.split('_')[-1]))
        print(f"Using best EMA checkpoint: {latest_model}")
    elif best_model_files:
        latest_model = max(best_model_files, key=lambda x: int(x.stem.split('_')[-1]))
        print(f"Using best model checkpoint: {latest_model}")
    elif model_files:
        latest_model = max(model_files, key=lambda x: int(x.stem.split('_')[-1]))
        print(f"Using model checkpoint: {latest_model}")
    else:
        print(f"ERROR: No compatible model checkpoint found in {ckpt_dir}")
        print("Looking for: best_ema_model_*.pt, best_model_*.pt, or model_*.pt")
        print(f"Found files: {list(ckpt_dir.glob('*.pt'))}")
        return
    
    python_exe = sys.executable
    print("=" * 60)
    
    # Set CUDA memory allocation config to reduce fragmentation
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
    
    # Stream output in real-time
    for line in process.stdout:
        print(line, end='', flush=True)
    
    process.wait()
    
    if process.returncode != 0:
        print(f"\nERROR: Synthetic data generation failed with return code {process.returncode}")
        return
    print("\n" + "=" * 60)
    print("Synthetic data generation completed successfully!")
    
    # Step 11: Load and decode synthetic data
    print("\nStep 11: Loading and decoding synthetic data...")
    # Find the generated synthetic data
    result_dirs = [
        tabdiff_dir / 'tabdiff' / 'result' / 'fraud_data' / 'quick_fraud',
        tabdiff_dir / 'tabdiff' / 'result' / 'fraud_data' / 'learnable_schedule',
        tabdiff_dir / 'tabdiff' / 'result' / 'fraud_data'
    ]
    
    syn_file = None
    for result_dir in result_dirs:
        if result_dir.exists():
            syn_files = list(result_dir.glob('*.csv'))
            if syn_files:
                syn_file = max(syn_files, key=lambda x: x.stat().st_mtime)
                break
    
    if syn_file is None:
        print(f"ERROR: No synthetic data files found. Searched:")
        for rd in result_dirs:
            print(f"  - {rd}")
        return
    
    print(f"Loading synthetic data from: {syn_file}")
    syn_df = pd.read_csv(syn_file)
    print(f"Synthetic data shape: {syn_df.shape}")
    
    # Apply inverse label encoding
    print("Applying inverse label encoding...")
    with open(encoders_path, 'r') as f:
        encoders_dict = json.load(f)
    
    # Reconstruct label encoders
    for col, encoder_info in encoders_dict.items():
        if col in syn_df.columns:
            encoder = LabelEncoder()
            encoder.classes_ = np.array(encoder_info['classes'])
            # Get unique values in synthetic data
            unique_vals = syn_df[col].unique()
            # Map to original labels
            syn_df[col] = syn_df[col].apply(
                lambda x: encoder.classes_[int(x)] if int(x) < len(encoder.classes_) else encoder.classes_[0]
            )
    
    # Step 12: Save final synthetic data
    output_path = output_dir / 'data' / 'synthetic' / 'tabdiff_synthetic_fraud_1000.csv'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    syn_df.to_csv(output_path, index=False)
    print(f"\n✓ Successfully generated 1000 synthetic fraud cases!")
    print(f"Saved to: {output_path}")
    print(f"Final synthetic data shape: {syn_df.shape}")
    print(f"Fraud cases in synthetic data: {(syn_df['isFraud'] == 1).sum()}")

if __name__ == '__main__':
    main()

