#!/usr/bin/env python3
"""
Retrain the fraud detection models from detect_fraud.ipynb using the
full IEEE-CIS dataset plus TabDiff-generated synthetic fraud samples.

This script reproduces the preprocessing, visualization, model training,
evaluation, and submission generation workflow, ensuring a like-for-like
comparison against the original notebook while augmenting the training
set with TabDiff outputs.
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


warnings.filterwarnings("ignore", category=FutureWarning)
pd.options.mode.copy_on_write = True  # Avoid chained-assignment warnings
sns.set_theme(style="whitegrid")


class PreprocessDatasets:
    """Data preparation utilities (port of detect_fraud.ipynb helpers)."""

    def join_ID(self, transaction_df: pd.DataFrame, ID_df: pd.DataFrame) -> pd.DataFrame:
        merged_df = pd.merge(transaction_df, ID_df, on="TransactionID", how="outer")
        merged_df.columns = [col.replace("-", "_") for col in merged_df.columns]
        return merged_df

    def replace_blanks(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(-999)
        non_numeric_cols = [col for col in df.columns if col not in numeric_cols]
        if non_numeric_cols:
            df[non_numeric_cols] = df[non_numeric_cols].fillna("-999")
        return df

    def encode_df(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        label_encoder = LabelEncoder()
        scaler = StandardScaler()
        minmax_scaler = MinMaxScaler()

        for col in df.columns:
            if col in {"TransactionID", "isFraud"}:
                continue

            if re.match(r"^D\d+$", col):
                df[col] = minmax_scaler.fit_transform(df[[col]])
            elif pd.api.types.is_numeric_dtype(df[col]):
                df[col] = scaler.fit_transform(df[[col]])
            else:
                df[col] = label_encoder.fit_transform(df[col].astype(str))

        return df

    def remove_outliers(self, data: pd.DataFrame, column: str) -> pd.DataFrame:
        data = data.copy()
        z_scores = stats.zscore(data[column].astype(float), nan_policy="omit")
        non_outliers = np.abs(z_scores) < 3
        removed = (~non_outliers).sum()
        print(f"{removed} values were removed as outliers (>3σ) in {column}")
        return data.loc[non_outliers].reset_index(drop=True)

    def remove_empty_cols(self, data: pd.DataFrame) -> pd.DataFrame:
        threshold = 0.90
        min_non_null = int((1 - threshold) * len(data))
        return data.dropna(axis=1, thresh=min_non_null)

    def feature_engineer(self, data: pd.DataFrame) -> pd.DataFrame:
        data = data.copy()

        def safe_split(series: pd.Series, pattern: str, idx: int) -> pd.Series:
            return (
                series.astype(str)
                .str.split(pattern, expand=True)
                .iloc[:, idx]
                .replace("nan", np.nan)
            )

        if {"P_emaildomain", "R_emaildomain"}.issubset(data.columns):
            data[["P_emailserver", "P_suffix"]] = data["P_emaildomain"].astype(str).str.split(
                ".", n=1, expand=True
            )
            data[["R_emailserver", "R_suffix"]] = data["R_emaildomain"].astype(str).str.split(
                ".", n=1, expand=True
            )

        if "id_30" in data.columns:
            data["os"] = data["id_30"].astype(str).str.split(" ", expand=True)[0]
        if "id_33" in data.columns:
            screen = data["id_33"].astype(str).str.split("x", expand=True)
            data["screen_width"] = pd.to_numeric(screen[0], errors="coerce")
            data["screen_height"] = pd.to_numeric(screen[1], errors="coerce")
        if "id_31" in data.columns:
            data["browser"] = data["id_31"].astype(str).str.split(" ", expand=True)[0].str.lower()
        if "DeviceInfo" in data.columns:
            data["device_name"] = data["DeviceInfo"].astype(str).str.split(" ", expand=True)[0].str.lower()

        browser_patterns = {
            r"samsung/sm-g532m|samsung/sch|samsung/sm-g531h": "samsung",
            r"generic/android": "android",
            r"mozilla/firefox": "firefox",
            r"nokia/lumia": "nokia",
            r"zte/blade": "zte",
            r"lg/k-200": "lg",
            r"lanix/ilium": "lanix",
            r"blu/dash": "blu",
            r"m4tel/m4": "m4",
        }

        device_patterns = {
            r"samsung|sgh|sm|gt-": "samsung",
            r"mot": "motorola",
            r"ale-|.*-l|hi": "huawei",
            r"lg": "lg",
            r"rv:": "rv",
            r"blade": "zte",
            r"xt": "sony",
            r"iphone": "ios",
            r"lenovo": "lenovo",
            r"mi|redmi": "xiaomi",
            r"ilium": "ilium",
            r"alcatel": "alcatel",
            r"asus": "asus",
        }

        def match_patterns(df: pd.DataFrame, patterns: Dict[str, str], col_name: str) -> pd.DataFrame:
            if col_name not in df.columns:
                return df
            for pattern, value in patterns.items():
                df[col_name] = df[col_name].str.replace(pattern, value, regex=True)
            return df

        data = match_patterns(data, browser_patterns, "browser")
        data = match_patterns(data, device_patterns, "device_name")

        if "TransactionDT" in data.columns:
            start_date = datetime(2017, 11, 30)
            full_date = data["TransactionDT"].astype(float).apply(lambda x: start_date + timedelta(seconds=x))
            data["TransactionFullDate"] = full_date
            data["TransactionDate"] = full_date.dt.date
            data["DayOfWeek"] = full_date.dt.dayofweek.apply(lambda x: (x + 1) % 7)
            data["HourOfDay"] = full_date.dt.hour
            data["Month"] = full_date.dt.month

        return data

    def reduce_memory(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        start = df.memory_usage().sum() / 1024**2
        print(f"Starting memory usage: {start:.2f} MB")

        for col in df.columns:
            col_type = df[col].dtype
            col_min, col_max = df[col].min(), df[col].max()

            if pd.api.types.is_float_dtype(col_type):
                if col_min > np.finfo(np.float32).min and col_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
            elif pd.api.types.is_integer_dtype(col_type):
                if col_min > np.iinfo(np.int8).min and col_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif col_min > np.iinfo(np.int16).min and col_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif col_min > np.iinfo(np.int32).min and col_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = df[col].astype("category")

        end = df.memory_usage().sum() / 1024**2
        print(f"Memory usage after downsizing: {end:.2f} MB ({100 * (start - end) / start:.1f}% reduction)")
        return df

    def final_preprocessing(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df = train_df.sort_values(by="TransactionID", ascending=True).reset_index(drop=True)
        test_df = test_df.sort_values(by="TransactionID", ascending=True).reset_index(drop=True)

        drop_cols = [
            "P_emaildomain",
            "R_emaildomain",
            "id_30",
            "id_31",
            "id_33",
            "DeviceInfo",
            "TransactionDT",
            "TransactionFullDate",
            "TransactionDate",
            "TransactionID",
        ]
        keep_cols = [col for col in drop_cols if col in train_df.columns]
        train_df = train_df.drop(columns=keep_cols, errors="ignore")
        test_df = test_df.drop(columns=keep_cols, errors="ignore")

        drop_from_test = [col for col in test_df.columns if col not in train_df.columns]
        test_df = test_df.drop(columns=drop_from_test, errors="ignore")
        return train_df, test_df


class VisualizeDataset:
    """Visualization helpers that both show and persist artifacts."""

    def __init__(self, save_dir: Path, show_plots: bool):
        self.save_dir = save_dir
        self.show_plots = show_plots
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def _finalize(self, fig: plt.Figure, filename: str) -> None:
        fig.savefig(self.save_dir / filename, dpi=200, bbox_inches="tight")
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def plot_hist(self, df: pd.DataFrame, col: str, plot_fraud: bool) -> None:
        if plot_fraud:
            fig, axes = plt.subplots(1, 2, figsize=(16, 6))
            sns.histplot(df[df["isFraud"] == 1], x=col, bins=100, kde=True, color="red", ax=axes[0], edgecolor="black")
            axes[0].set_title("Fraudulent Transactions")
            sns.histplot(df[df["isFraud"] == 0], x=col, bins=100, kde=True, color="blue", ax=axes[1], edgecolor="black")
            axes[1].set_title("Non-Fraudulent Transactions")
            fig.suptitle(f"Distribution of {col} by Transaction Type")
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df, x=col, bins=100, kde=True, edgecolor="black", color="green", ax=ax)
            ax.set_title(f"Histogram of {col}")
        self._finalize(fig, f"hist_{col}.png")

    def plot_fraud_per_period(self, df: pd.DataFrame, period_cols: list[str]) -> None:
        fig, axes = plt.subplots(len(period_cols), 1, figsize=(10, 4 * len(period_cols)))
        axes = np.atleast_1d(axes)
        for idx, col in enumerate(period_cols):
            fraud_grouped = df[df["isFraud"] == 1].groupby(col)["isFraud"].size()
            total_grouped = df.groupby(col)["isFraud"].size()
            fraud_percentage = (fraud_grouped / total_grouped * 100).fillna(0)
            sns.barplot(x=fraud_percentage.index, y=fraud_percentage.values, color="red", ax=axes[idx])
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel("Fraud Percentage (%)")
        fig.suptitle("Fraud Share by Period")
        fig.tight_layout()
        self._finalize(fig, "fraud_period.png")

    def plot_correlation(self, df: pd.DataFrame, cols: list[str], label: str) -> None:
        subset = df[cols]
        corr_matrix = subset.corr()
        fig, ax = plt.subplots(figsize=(13, 11))
        sns.heatmap(corr_matrix, annot=True, cmap="viridis", fmt=".2f", vmin=-1, vmax=1, square=True, ax=ax)
        ax.set_title(f"Correlation Heatmap ({label})")
        fig.tight_layout()
        self._finalize(fig, f"corr_{label}.png")


class ReduceDimension:
    """PCA-based dimensionality reduction for V-columns."""

    def __init__(self, traindata: pd.DataFrame, testdata: pd.DataFrame):
        self.traindata = traindata
        self.testdata = testdata

    def plot_and_reduce(self, plot_dir: Path, show_plots: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
        plot_dir.mkdir(parents=True, exist_ok=True)
        v_cols = [col for col in self.traindata.columns if re.match(r"^V\d+$", col)]
        if not v_cols:
            return self.traindata, self.testdata

        v_data = self.traindata[v_cols]
        pca = PCA().fit(v_data)
        explained_variance = pca.explained_variance_ratio_.cumsum()
        comp_90 = next(i for i, total in enumerate(explained_variance) if total >= 0.90) + 1
        print(f"Number of PCA components to retain for 90% variance: {comp_90}")

        plots = {
            "pca_scree.png": (range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_),
            "pca_cumulative.png": (range(1, len(explained_variance) + 1), explained_variance),
        }
        for filename, (x_vals, y_vals) in plots.items():
            fig, ax = plt.subplots(figsize=(12, 7))
            sns.lineplot(x=list(x_vals), y=list(y_vals), marker="o", ax=ax)
            ax.set_title(filename.replace("_", " ").replace(".png", "").title())
            if "cumulative" in filename:
                ax.axhline(y=0.90, color="red", linestyle="--")
            fig.tight_layout()
            fig.savefig(plot_dir / filename, dpi=200, bbox_inches="tight")
            if show_plots:
                plt.show()
            else:
                plt.close(fig)

        v_data_pca_df = pd.DataFrame(pca.transform(v_data)[:, :comp_90], columns=[f"PC{i+1}" for i in range(comp_90)])
        data_final = pd.concat([self.traindata.drop(columns=v_cols).reset_index(drop=True), v_data_pca_df], axis=1)

        v_test_data_pca_df = pd.DataFrame(
            pca.transform(self.testdata[v_cols])[:, :comp_90], columns=[f"PC{i+1}" for i in range(comp_90)]
        )
        test_data_final = pd.concat([self.testdata.drop(columns=v_cols).reset_index(drop=True), v_test_data_pca_df], axis=1)
        return data_final, test_data_final


class ApplyModel:
    """Model training, evaluation, and submission utilities."""

    def __init__(self, eval_: bool, plot_dir: Path, show_plots: bool):
        self.eval = eval_
        self.plot_dir = plot_dir
        self.show_plots = show_plots
        self.plot_dir.mkdir(parents=True, exist_ok=True)
        self.last_metrics: Optional[Dict[str, float]] = None

    def _finalize(self, fig: plt.Figure, filename: str) -> None:
        fig.savefig(self.plot_dir / filename, dpi=200, bbox_inches="tight")
        if self.show_plots:
            plt.show()
        else:
            plt.close(fig)

    def evaluate_model(self, y_test: pd.Series, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title("Confusion Matrix")
        fig.tight_layout()
        self._finalize(fig, "confusion_matrix.png")

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_pred_proba),
            "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            "matthews_corrcoef": matthews_corrcoef(y_test, y_pred),
            "cohen_kappa": cohen_kappa_score(y_test, y_pred),
        }

        print("\n------------- Model Evaluation Metrics -------------")
        for name, value in metrics.items():
            print(f"{name}: {value:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))
        return metrics

    def train_model(
        self,
        X_train_split: pd.DataFrame,
        X_val_split: pd.DataFrame,
        y_train_split: pd.Series,
        y_val_split: pd.Series,
        y_full: pd.Series,
    ) -> xgb.XGBClassifier:
        scale_pos_weight = len(y_full[y_full == 0]) / max(1, len(y_full[y_full == 1]))
        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            use_label_encoder=False,
            learning_rate=0.05,
            n_estimators=5000,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            early_stopping_rounds=10,
        )
        xgb_model.fit(X_train_split, y_train_split, eval_set=[(X_val_split, y_val_split)], verbose=False)

        if self.eval:
            yval_pred = xgb_model.predict(X_val_split)
            yval_pred_proba = xgb_model.predict_proba(X_val_split)[:, 1]
            self.last_metrics = self.evaluate_model(y_val_split, yval_pred, yval_pred_proba)
        else:
            self.last_metrics = None
        return xgb_model

    def pred_and_submit(
        self,
        model: xgb.XGBClassifier,
        Xtest: pd.DataFrame,
        test_transaction_df: pd.DataFrame,
        submission_path: Path,
    ) -> pd.DataFrame:
        y_pred_proba = model.predict_proba(Xtest)[:, 1]
        submission = pd.DataFrame({"TransactionID": test_transaction_df["TransactionID"], "isFraud": y_pred_proba})

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(submission["isFraud"], bins=100, kde=True, color="green", edgecolor="black", ax=ax)
        ax.set_title("Histogram of Fraud Probability Predictions")
        fig.tight_layout()
        self._finalize(fig, "prediction_hist.png")

        if submission_path.exists():
            backup_path = submission_path.with_suffix(submission_path.suffix + ".bak")
            submission_path.rename(backup_path)
            print(f"Existing submission backed up to {backup_path}")

        submission.to_csv(submission_path, index=False)
        print(f"Submission saved to {submission_path}")
        return submission


def build_label_decoders(train_df: pd.DataFrame) -> Dict[str, LabelEncoder]:
    decoders: Dict[str, LabelEncoder] = {}
    for col in train_df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        le.fit(train_df[col].astype(str).fillna("nan"))
        decoders[col] = le
    return decoders


def decode_tabdiff_samples(
    samples_df: pd.DataFrame, decoders: Dict[str, LabelEncoder], template_columns: list[str]
) -> pd.DataFrame:
    samples_df = samples_df.copy()
    missing_cols = [col for col in template_columns if col not in samples_df.columns]
    for col in missing_cols:
        samples_df[col] = np.nan
    samples_df = samples_df[template_columns]

    for col, encoder in decoders.items():
        if col not in samples_df.columns:
            continue
        classes = encoder.classes_
        # Round continuous outputs back to the nearest valid label index
        encoded = pd.to_numeric(samples_df[col], errors="coerce").round().astype("Int64")
        encoded = encoded.clip(lower=0, upper=len(classes) - 1)
        encoded_filled = encoded.fillna(0).astype(int)
        decoded = encoder.inverse_transform(encoded_filled)
        decoded_series = pd.Series(decoded, index=samples_df.index)
        decoded_series[encoded.isna()] = np.nan
        samples_df[col] = decoded_series

    if "TransactionID" in samples_df.columns:
        samples_df["TransactionID"] = pd.to_numeric(samples_df["TransactionID"], errors="coerce").round()
    return samples_df


def load_tabdiff_samples(samples_path: Path) -> pd.DataFrame:
    if not samples_path.exists():
        raise FileNotFoundError(f"TabDiff samples not found at {samples_path}")
    print(f"Loading TabDiff synthetic data from {samples_path}")
    return pd.read_csv(samples_path)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parent
    default_data_root = project_root / "data" / "ieee-fraud-detection"
    default_tabdiff = (
        project_root
        / "TabDiff"
        / "tabdiff"
        / "result"
        / "fraud_data"
        / "learnable_schedule"
        / "50"
        / "ema"
        / "samples.csv"
    )
    if not default_tabdiff.exists():
        default_tabdiff = default_tabdiff.parent / "samples.csv"

    parser = argparse.ArgumentParser(description="Retrain fraud model with TabDiff synthetic data.")
    parser.add_argument("--data-root", type=Path, default=default_data_root, help="Path to IEEE-CIS data directory")
    parser.add_argument("--tabdiff-samples", type=Path, default=default_tabdiff, help="Path to TabDiff samples.csv")
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=project_root / "artifacts" / "plots_tabdiff",
        help="Directory to store generated plots",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=project_root / "artifacts" / "metrics_tabdiff.json",
        help="Where to write evaluation metrics",
    )
    parser.add_argument(
        "--submission-path",
        type=Path,
        default=project_root / "submission.csv",
        help="Output path for submission.csv",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="Validation split size")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for splitting")
    parser.add_argument("--show-plots", action="store_true", help="Display plots in addition to saving them")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    plots_dir = args.plots_dir.resolve()
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.metrics_path.resolve()
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    submission_path = args.submission_path.resolve()

    # Load base datasets
    train_id_df = pd.read_csv(data_root / "train_identity.csv")
    train_transaction_df = pd.read_csv(data_root / "train_transaction.csv")
    test_id_df = pd.read_csv(data_root / "test_identity.csv")
    test_transaction_df = pd.read_csv(data_root / "test_transaction.csv")

    preprocessor = PreprocessDatasets()
    train_transaction_df = preprocessor.remove_empty_cols(train_transaction_df)
    train_id_df = preprocessor.remove_empty_cols(train_id_df)
    test_transaction_df = preprocessor.remove_empty_cols(test_transaction_df)
    test_id_df = preprocessor.remove_empty_cols(test_id_df)

    train_df = preprocessor.join_ID(train_transaction_df, train_id_df)
    test_df = preprocessor.join_ID(test_transaction_df, test_id_df)

    # Build label decoders before feature engineering (raw string space)
    decoders = build_label_decoders(train_df)

    tabdiff_samples = load_tabdiff_samples(args.tabdiff_samples.resolve())
    tabdiff_samples = decode_tabdiff_samples(tabdiff_samples, decoders, train_df.columns.tolist())

    print(f"Original train rows: {len(train_df):,}")
    print(f"TabDiff synthetic rows: {len(tabdiff_samples):,}")
    train_df = pd.concat([train_df, tabdiff_samples], ignore_index=True)
    print(f"Combined train rows: {len(train_df):,}")

    train_df = preprocessor.remove_outliers(train_df, "TransactionAmt")
    train_df = preprocessor.feature_engineer(train_df)
    test_df = preprocessor.feature_engineer(test_df)

    train_df = preprocessor.replace_blanks(train_df)
    test_df = preprocessor.replace_blanks(test_df)

    visualizer = VisualizeDataset(plots_dir / "eda", args.show_plots)
    visualizer.plot_hist(train_df, "TransactionAmt", plot_fraud=True)
    visualizer.plot_fraud_per_period(train_df, ["DayOfWeek", "HourOfDay"])

    c_cols = [col for col in train_df.columns if col.startswith("C")]
    d_cols = [f"D{i}" for i in list(range(1, 7)) + list(range(8, 16))]
    if c_cols:
        visualizer.plot_correlation(train_df, c_cols[:20], "C_columns")  # limit for readability
    if d_cols:
        visualizer.plot_correlation(train_df, d_cols, "D_columns")

    train_df = preprocessor.reduce_memory(train_df)
    test_df = preprocessor.reduce_memory(test_df)

    train_df = preprocessor.encode_df(train_df)
    test_df = preprocessor.encode_df(test_df)

    reducer = ReduceDimension(train_df, test_df)
    train_df, test_df = reducer.plot_and_reduce(plots_dir / "pca", args.show_plots)
    train_df, test_df = preprocessor.final_preprocessing(train_df, test_df)

    X_train = train_df.drop(columns=["isFraud"])
    y_train = train_df["isFraud"]
    X_test = test_df

    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train, y_train, test_size=args.test_size, stratify=y_train, random_state=args.random_state
    )

    model_runner = ApplyModel(eval_=True, plot_dir=plots_dir / "model", show_plots=args.show_plots)
    xgb_model = model_runner.train_model(X_train_split, X_val_split, y_train_split, y_val_split, y_train)
    submission = model_runner.pred_and_submit(xgb_model, X_test, test_transaction_df, submission_path)

    metrics = model_runner.last_metrics
    if metrics is None:
        y_val_pred = xgb_model.predict(X_val_split)
        y_val_pred_proba = xgb_model.predict_proba(X_val_split)[:, 1]
        metrics = model_runner.evaluate_model(y_val_split, y_val_pred, y_val_pred_proba)
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")

    submission_summary = submission["isFraud"].describe().to_dict()
    print("\nSubmission probability summary:")
    for key, value in submission_summary.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

