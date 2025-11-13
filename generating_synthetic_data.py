import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os
import json
import subprocess
import matplotlib.pyplot as plt
import torch
import platform
import glob
import re

def detect_and_setup_gpu():
    """
    Detects available GPU acceleration and returns the appropriate device string.
    Supports CUDA and Mac Metal Performance Shaders (MPS).
    """
    device = 'cpu'
    
    # Check for CUDA
    if torch.cuda.is_available():
        device = 'cuda:0'
        print(f"✅ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    # Check for Mac GPU (Metal Performance Shaders)
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = 'mps'
        print("✅ Mac GPU (Metal Performance Shaders) detected and enabled")
    else:
        print("⚠️  No GPU detected. Training will use CPU (slower)")
    
    return device

def enable_mac_gpu_in_tabdiff():
    """
    Patches TabDiff's main.py to support Mac GPU (MPS) detection.
    """
    tabdiff_path = 'TabDiff'
    main_py_path = os.path.join(tabdiff_path, 'main.py')
    
    with open(main_py_path, 'r') as f:
        content = f.read()
    
    # Replace the CUDA-only device detection with MPS-aware version
    old_device_check = """    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'"""
    
    new_device_check = """    # check cuda and mac gpu
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        args.device = 'mps'
    else:
        args.device = 'cpu'"""
    
    if old_device_check in content:
        content = content.replace(old_device_check, new_device_check)
        with open(main_py_path, 'w') as f:
            f.write(content)
        print("✅ TabDiff patched for Mac GPU support")
    else:
        print("ℹ️  TabDiff already supports Mac GPU or patch not needed")

def preprocess_data():
    """
    Loads and preprocesses the IEEE-CIS fraud detection data.
    This function mirrors the preprocessing steps from detect_fraud.ipynb.
    """
    print("Loading data...")
    train_transaction = pd.read_csv('data/ieee-fraud-detection/train_transaction.csv')
    train_identity = pd.read_csv('data/ieee-fraud-detection/train_identity.csv')

    print("Merging data...")
    train = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')

    # Drop unnecessary columns (as in notebook)
    many_null_cols = [col for col in train.columns if train[col].isnull().sum() / train.shape[0] > 0.9]
    big_top_value_cols = [col for col in train.columns if train[col].value_counts(dropna=False, normalize=True).values[0] > 0.9]
    one_value_cols = [col for col in train.columns if train[col].nunique() <= 1]
    cols_to_drop = list(set(many_null_cols + big_top_value_cols + one_value_cols))
    cols_to_drop.remove('isFraud')
    train = train.drop(cols_to_drop, axis=1)

    print("Preprocessing and filling NaNs...")
    # Fill NA values
    train['card4'].fillna('unknown', inplace=True)
    train['card6'].fillna('unknown', inplace=True)
    train.fillna(-999, inplace=True)

    print("Label encoding categorical features...")
    for col in train.columns:
        if train[col].dtype == 'object':
            le = LabelEncoder()
            le.fit(list(train[col].astype(str).values))
            train[col] = le.transform(list(train[col].astype(str).values))
    
    return train

def prepare_for_tabdiff(df):
    """
    Prepares the data and metadata for TabDiff.
    """
    print("Preparing data for TabDiff...")
    
    tabdiff_path = 'TabDiff'
    if not os.path.exists(tabdiff_path):
        raise FileNotFoundError("TabDiff directory not found. Please clone it first using 'git clone https://github.com/MinkaiXu/TabDiff.git'")
    
    # Create directories in TabDiff if they don't exist
    tabdiff_data_dir = os.path.join(tabdiff_path, 'data')
    tabdiff_info_dir = os.path.join(tabdiff_data_dir, 'Info')
    tabdiff_fraud_processed_dir = os.path.join(tabdiff_data_dir, 'fraud_data')
    os.makedirs(tabdiff_data_dir, exist_ok=True)
    os.makedirs(tabdiff_info_dir, exist_ok=True)
    os.makedirs(tabdiff_fraud_processed_dir, exist_ok=True)
    
    # Also create our own data/synthetic directory for backup
    os.makedirs('data/synthetic', exist_ok=True)
    os.makedirs('data/synthetic/Info', exist_ok=True)
    
    # Isolate fraud cases
    fraud_df = df[df['isFraud'] == 1].copy()
    
    # Save fraud data to both locations
    fraud_data_path_tabdiff = os.path.join(tabdiff_data_dir, 'fraud_data.csv')
    fraud_data_path_local = 'data/synthetic/fraud_data.csv'
    
    fraud_df.to_csv(fraud_data_path_tabdiff, index=False)
    fraud_df.to_csv(fraud_data_path_local, index=False)
    
    print(f"Saved {len(fraud_df)} fraud cases to {fraud_data_path_tabdiff}")

    # Create metadata file
    num_col_idx = [i for i, dtype in enumerate(fraud_df.dtypes) if dtype in ['int64', 'float64'] and fraud_df.columns[i] not in ['isFraud']]
    cat_col_idx = [i for i, dtype in enumerate(fraud_df.dtypes) if dtype not in ['int64', 'float64']]
    target_col_idx = [fraud_df.columns.get_loc('isFraud')]

    info = {
        "name": "fraud_data",
        "task_type": "binclass",
        "header": "infer",
        "column_names": None,
        "num_col_idx": num_col_idx,
        "cat_col_idx": cat_col_idx,
        "target_col_idx": target_col_idx,
        "file_type": "csv",
        "data_path": "data/fraud_data.csv",
        "val_path": None,
        "test_path": None,
    }

    info_path_tabdiff = os.path.join(tabdiff_info_dir, 'fraud_data.json')
    info_path_local = 'data/synthetic/Info/fraud_data.json'
    
    with open(info_path_tabdiff, 'w') as f:
        json.dump(info, f, indent=4)
    with open(info_path_local, 'w') as f:
        json.dump(info, f, indent=4)
        
    print(f"Saved metadata to {info_path_tabdiff}")
    
    # Process dataset with TabDiff's script
    print("Running TabDiff's process_dataset.py...")
    process_cmd = ['python', 'process_dataset.py', '--dataname', 'fraud_data']
    subprocess.run(process_cmd, cwd=tabdiff_path, check=True)
    
    return fraud_df

def run_tabdiff(lightweight=True):
    """
    Runs TabDiff training and sampling.
    If lightweight=True, uses smaller model for faster training (10-20 min on CPU/GPU).
    Automatically detects and uses available GPU (CUDA or Mac MPS).
    """
    print("\n" + "="*70)
    print("GPU SETUP")
    print("="*70)
    
    # Detect and setup GPU
    device = detect_and_setup_gpu()
    
    # Enable Mac GPU support in TabDiff
    enable_mac_gpu_in_tabdiff()
    
    print("="*70 + "\n")
    
    tabdiff_path = 'TabDiff'
    dataname = 'fraud_data'
    config_path = os.path.join(tabdiff_path, 'tabdiff', 'configs', 'tabdiff_configs.toml')
    
    # Create a lightweight config if requested
    if lightweight:
        print("\nUsing lightweight model configuration for faster training...")
        # Read the original config
        with open(config_path, 'r') as f:
            config_content = f.read()
        
        # Modify config for lightweight training using regex replacements (handles existing values e.g. 1200)
        replacements = [
            (r"steps\s*=\s*\d+", "steps = 50"),
            (r"num_timesteps\s*=\s*\d+", "num_timesteps = 10"),
            (r"dim_t\s*=\s*\d+", "dim_t = 256"),
            (r"batch_size\s*=\s*\d+", "batch_size = 2048"),
            (r"factor\s*=\s*\d+", "factor = 16"),
            (r"check_val_every\s*=\s*\d+", "check_val_every = 10"),
            (r"d_token\s*=\s*\d+", "d_token = 2"),
            (r"num_layers\s*=\s*\d+", "num_layers = 1"),
        ]

        lightweight_config = config_content
        for pattern, replacement in replacements:
            lightweight_config, count = re.subn(pattern, replacement, lightweight_config, count=1)
            if count == 0:
                print(f"⚠️  Warning: pattern '{pattern}' not found when applying lightweight config.")
        
        # Write the lightweight config
        with open(config_path, 'w') as f:
            f.write(lightweight_config)
        
        # Verify config was written correctly
        with open(config_path, 'r') as f:
            verify_config = f.read()
        
        if 'steps = 50' in verify_config and 'check_val_every = 10' in verify_config:
            print("✅ Lightweight config applied (50 steps, checkpoint every 10 steps).")
        else:
            print("⚠️  Config verification failed! Checking what was written...")
            print("steps = 50 found:", 'steps = 50' in verify_config)
            print("check_val_every = 10 found:", 'check_val_every = 10' in verify_config)
    
    print("\nTraining TabDiff model...")
    print("="*70)
    # Don't use custom exp_name - keep it default so checkpoint path is predictable
    train_cmd = ['python', 'main.py', '--dataname', dataname, '--mode', 'train', 
                 '--no_wandb']
    
    try:
        result = subprocess.run(train_cmd, cwd=tabdiff_path, check=False, capture_output=False)
        if result.returncode != 0:
            print(f"\n⚠️  Training exited with code {result.returncode}")
        else:
            print("\n✅ Training completed successfully")
    except Exception as e:
        print(f"⚠️  Training error: {e}")
    
    print("="*70)
    
    # Check if checkpoint exists before testing
    checkpoint_dir = os.path.join(tabdiff_path, 'tabdiff', 'ckpt', dataname, 'learnable_schedule')
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'best_ema_model*'))
    
    if checkpoint_files:
        print(f"\n✅ Checkpoint found: {checkpoint_files[0]}")
        print("\nSampling from TabDiff model...")
        print("="*70)
        # Test command finds the checkpoint automatically from the default location
        test_cmd = ['python', 'main.py', '--dataname', dataname, '--mode', 'test', 
                    '--no_wandb', '--report']
        subprocess.run(test_cmd, cwd=tabdiff_path, check=True)
        print("="*70)
    else:
        print(f"\n⚠️  No checkpoint found in {checkpoint_dir}")
        print("Available checkpoint directories:")
        ckpt_base = os.path.join(tabdiff_path, 'tabdiff', 'ckpt')
        if os.path.exists(ckpt_base):
            for root, dirs, files in os.walk(ckpt_base):
                if files:
                    print(f"  {root}: {files}")
        print("\nSkipping test phase...")
    
    # Restore original config
    if lightweight:
        print("\nRestoring original config...")
        with open(config_path, 'w') as f:
            f.write(config_content)

def visualize_and_save(original_fraud_df, lightweight=True):
    """
    Loads synthetic data, visualizes comparison, and saves the data.
    """
    print("\nLoading synthetic data...")
    
    # Find the generated samples. With default exp_name, path is predictable
    exp_name = 'learnable_schedule'  # Default when no --exp_name is specified
    sample_dir = f'TabDiff/eval/report_runs/{exp_name}/fraud_data'
    
    # Find the first sample file
    sample_files = glob.glob(f'{sample_dir}/sample_*.csv')
    
    if not sample_files:
        print(f"\n⚠️  Could not find sample files in {sample_dir}")
        print("This is expected if training didn't complete or produce samples.")
        print("\nAvailable report directories:")
        if os.path.exists('TabDiff/eval/report_runs'):
            for root, dirs, files in os.walk('TabDiff/eval/report_runs'):
                if files:
                    print(f"  {root}:")
                    for f in files[:5]:  # Show first 5 files
                        print(f"    - {f}")
        print("\nSkipping visualization...")
        return
    
    sample_path = sorted(sample_files)[0]
    print(f"✅ Found sample at: {sample_path}")
    
    synthetic_df = pd.read_csv(sample_path)
    print(f"✅ Loaded {len(synthetic_df)} synthetic fraud cases.")
    
    # Save synthetic data
    synthetic_data_path = 'data/synthetic/synthetic_fraud_data.csv'
    synthetic_df.to_csv(synthetic_data_path, index=False)
    print(f"Saved synthetic fraud data to {synthetic_data_path}")

    # Create plot
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Original Fraud Cases', 'Synthetic Fraud Cases'], [len(original_fraud_df), len(synthetic_df)], color=['blue', 'green'])
    plt.ylabel('Number of Cases')
    plt.title('Comparison of Original and Synthetic Fraud Cases')
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, yval, int(yval), va='bottom') # va: vertical alignment

    plot_path = 'synthetic_fraud_cases_comparison.png'
    plt.savefig(plot_path)
    print(f"Saved comparison plot to {plot_path}")
    plt.show()


if __name__ == '__main__':
    print("\n" + "="*70)
    print("FRAUD DETECTION SYNTHETIC DATA GENERATION")
    print("="*70)
    print(f"Platform: {platform.system()}")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MPS Available: {hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()}")
    print("="*70 + "\n")
    
    processed_data = preprocess_data()
    original_fraud_df = prepare_for_tabdiff(processed_data)
    # Use lightweight=True for faster training (10-20 min with GPU), lightweight=False for full model
    run_tabdiff(lightweight=True)
    visualize_and_save(original_fraud_df, lightweight=True)
    print("\n✅ Synthetic data generation process completed.")
