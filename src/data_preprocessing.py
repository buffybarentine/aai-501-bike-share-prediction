"""Data cleaning and preprocessing pipeline for AAI-501 Bike Share project."""

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_and_validate_data(raw_path: Path) -> pd.DataFrame:
    """Load hourly dataset, validate integrity, and sort chronologically."""
    df = pd.read_csv(raw_path)

    # 1. Check for missing values and duplicates:
    # Confirm zero blank cells or duplicate rows and print proof for report.
    missing_count = int(df.isnull().sum().sum())
    duplicate_count = int(df.duplicated().sum())
    print(
        f"Loaded {len(df)} rows | Missing values: {missing_count} | "
        f"Duplicates: {duplicate_count}"
    )

    # 2. Sort the data in time order:
    # Sort rows chronologically by dteday and hr from Jan 2011 to Dec 2012.
    df = df.sort_values(by=["dteday", "hr"]).reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode cyclical time features and one-hot categorical variables."""
    processed = df.copy()

    # 5. Turn category numbers into separate yes/no columns (One-Hot Encoding):
    # Convert season (1-4) and weathersit (1-4) into binary dummy columns
    # so the model treats them as distinct categories rather than math scores.
    # Note: weathersit=4 only appears 3 times; keeping it documents rare events.
    processed = pd.get_dummies(
        processed,
        columns=["season", "weathersit"],
        drop_first=True,
        dtype=int,
    )

    # 6. Make clock hours and months wrap around properly (Cyclical Encoding):
    # Convert hr (0-23) and mnth (1-12) into sine and cosine columns so
    # hour 23 -> 0 and month 12 -> 1 loop smoothly in a circle.
    processed["hr_sin"] = np.sin(2 * np.pi * processed["hr"] / 24.0)
    processed["hr_cos"] = np.cos(2 * np.pi * processed["hr"] / 24.0)
    processed["mnth_sin"] = np.sin(2 * np.pi * processed["mnth"] / 12.0)
    processed["mnth_cos"] = np.cos(2 * np.pi * processed["mnth"] / 12.0)

    return processed


def prepare_regression_splits(
    df: pd.DataFrame, output_dir: Path, train_ratio: float = 0.8
) -> None:
    """Create leakage-free chronological train/test splits for regression."""
    # 3. Drop useless and "cheating" columns (Target Leakage):
    # Drop instant (row ID) plus casual and registered (since casual +
    # registered = cnt, leaving them in lets the model cheat).
    leakage_and_id_cols = ["instant", "casual", "registered"]

    # 4. Fix duplicate temperature columns (Multicollinearity):
    # Drop atemp ("feels-like" temp) because it is ~99% correlated with temp.
    collinear_cols = ["atemp"]

    # 7. Drop dteday after extracting everything you need:
    # Drop raw text date dteday (and raw hr/mnth replaced by sin/cos)
    # because machine learning models only accept numeric inputs.
    raw_time_cols = ["dteday", "hr", "mnth"]

    reg_df = df.drop(
        columns=leakage_and_id_cols + collinear_cols + raw_time_cols
    )

    # 8. Split the Regression data by time (Chronological Train/Test Split):
    # Put the first 80% of the timeline into regression_train.csv and the
    # final 20% into regression_test.csv to test predicting future demand.
    # Keep train_ratio >= 0.6 so yr=1 (2012) appears in the training set.
    # Note for modeling: wrap Ridge/Lasso in a Pipeline with StandardScaler.
    split_idx = int(len(reg_df) * train_ratio)
    train_df = reg_df.iloc[:split_idx]
    test_df = reg_df.iloc[split_idx:]

    train_df.to_csv(output_dir / "regression_train.csv", index=False)
    test_df.to_csv(output_dir / "regression_test.csv", index=False)
    print(
        f"Saved regression splits -> Train: {len(train_df)} rows | "
        f"Test: {len(test_df)} rows"
    )


def prepare_clustering_data(df: pd.DataFrame, output_dir: Path) -> None:
    """Save scaled features for K-Means and unscaled copy for profiling."""
    # 9. Scale the numbers for Clustering (StandardScaler):
    # Keep casual, registered, workingday, temp, hum, windspeed, cyclical
    # time, and encoded season/weathersit; drop IDs, raw dates, atemp, and cnt.
    drop_cols = ["instant", "dteday", "atemp", "cnt", "hr", "mnth"]
    clust_df = df.drop(columns=drop_cols)

    # Apply StandardScaler so large counts (e.g., registered 0-886) do not
    # overpower small decimals (e.g., hum 0.0-1.0) in distance calculations.
    scaler = StandardScaler()
    scaled_array = scaler.fit_transform(clust_df)
    scaled_df = pd.DataFrame(scaled_array, columns=clust_df.columns)
    scaled_df.to_csv(output_dir / "clustering_scaled.csv", index=False)

    # 10. Export unscaled clustering reference for human interpretation:
    # Save original units (plus readable hr, mnth, and cnt) so cluster labels
    # from K-Means can be mapped back to real rider counts and hours.
    unscaled_drop_cols = ["instant", "dteday", "atemp"]
    unscaled_df = df.drop(columns=unscaled_drop_cols)
    unscaled_df.to_csv(output_dir / "clustering_unscaled.csv", index=False)

    print(
        f"Saved clustering data -> Scaled: {scaled_df.shape} | "
        f"Unscaled reference: {unscaled_df.shape}"
    )


def main() -> None:
    """Run the full preprocessing pipeline."""
    raw_file = Path("data/raw/hour.csv")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_and_validate_data(raw_file)
    featured_df = engineer_features(raw_df)

    prepare_regression_splits(featured_df, processed_dir)
    prepare_clustering_data(featured_df, processed_dir)
    print("Preprocessing complete.")


if __name__ == "__main__":
    main()

# =============================================================================
# AI Disclosure & Attribution:
# Portions of this code structure, PEP 8 formatting, and inline documentation
# were generated with assistance from Google Gemini (Model: Gemini 3.1 Pro)
# and reviewed, verified, and adapted by the AAI-501 Group 3 team.
# =============================================================================
