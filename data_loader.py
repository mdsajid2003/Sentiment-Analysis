import pandas as pd
import os
from model.preprocess import clean_tweet  # Import the clean_tweet function

def prepare_datasets(train_file, val_file):
    """
    Load and preprocess training and validation datasets.

    Parameters:
    - train_file (str): Path to the training dataset file.
    - val_file (str): Path to the validation dataset file.

    Returns:
    - train_df (pd.DataFrame): Preprocessed training dataset.
    - val_df (pd.DataFrame): Preprocessed validation dataset.
    """
    # Load datasets
    try:
        train_df = pd.read_csv(train_file)
        val_df = pd.read_csv(val_file)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Dataset file not found: {e}")
    except pd.errors.EmptyDataError as e:
        raise ValueError(f"Dataset file is empty: {e}")

    # Debug: Print dataset shapes and column names
    print("Initial training data shape:", train_df.shape)
    print("Initial validation data shape:", val_df.shape)
    print("Training dataset columns:", train_df.columns)
    print("Validation dataset columns:", val_df.columns)

    # Ensure datasets have the expected structure
    if train_df.shape[1] < 4 or val_df.shape[1] < 4:
        raise ValueError("Both datasets must have at least 4 columns (including 'Tweet' and 'Sentiment').")

    # Adjust column names dynamically
    try:
        train_df.rename(columns={train_df.columns[3]: 'Tweet', train_df.columns[2]: 'Sentiment'}, inplace=True)
        val_df.rename(columns={val_df.columns[3]: 'Tweet', val_df.columns[2]: 'Sentiment'}, inplace=True)
    except IndexError:
        raise ValueError("Column indices for renaming are out of range. Please check the dataset structure.")

    # Validate required columns exist after renaming
    for df, name in [(train_df, 'training'), (val_df, 'validation')]:
        if 'Tweet' not in df.columns or 'Sentiment' not in df.columns:
            raise ValueError(f"Required columns ('Tweet', 'Sentiment') are missing in the {name} dataset.")

    # Preprocess 'Tweet' column
    train_df['Tweet'] = train_df['Tweet'].astype(str).apply(clean_tweet)
    val_df['Tweet'] = val_df['Tweet'].astype(str).apply(clean_tweet)

    # Standardize sentiment labels and map them to numeric values
    label_mapping = {'negative': 0, 'neutral': 1, 'positive': 2}
    train_df['Sentiment'] = train_df['Sentiment'].str.lower().map(label_mapping)
    val_df['Sentiment'] = val_df['Sentiment'].str.lower().map(label_mapping)

    # Drop rows with missing Sentiment values
    train_df.dropna(subset=['Sentiment'], inplace=True)
    val_df.dropna(subset=['Sentiment'], inplace=True)

    # Debug: Check dataset shape after preprocessing
    print(f"Training data shape after preprocessing: {train_df.shape}")
    print(f"Validation data shape after preprocessing: {val_df.shape}")

    # Ensure datasets are not empty after preprocessing
    if train_df.empty:
        raise ValueError("Training dataset is empty after preprocessing.")
    if val_df.empty:
        raise ValueError("Validation dataset is empty after preprocessing.")

    return train_df, val_df
