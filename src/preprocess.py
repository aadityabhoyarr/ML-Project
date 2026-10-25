"""Data Preprocessing and Train/Test Splitting Module.

This script loads the raw dataset, creates cleaned and truncated text columns
for both evaluation conditions (Condition A: Full text, Condition B: Truncated
to first 50 words), and performs an 80/20 stratified train/test split.
"""

import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text_for_tfidf(text: str) -> str:
    """Cleans raw text for TF-IDF vectorization.

    Performs:
    1. Lowercasing
    2. Removal of URLs
    3. Removal of non-alphanumeric noise (preserving words and spaces)
    4. Normalization of whitespace
    """
    if not isinstance(text, str):
        return ""
    # Convert to lowercase
    cleaned = text.lower()
    # Remove URLs
    cleaned = re.sub(r"https?://\S+|www\.\S+", " ", cleaned)
    # Remove emails
    cleaned = re.sub(r"\S+@\S+", " ", cleaned)
    # Remove non-alphanumeric characters (keep letters, numbers, spaces)
    cleaned = re.sub(r"[^a-z0-9\s]", " ", cleaned)
    # Collapse multiple whitespaces
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def truncate_text(text: str, max_words: int = 50) -> str:
    """Truncates text to the first `max_words` words based on whitespace split.

    If the text has fewer words than max_words, returns the entire text.
    Preserves original casing and punctuation of those words.
    """
    if not isinstance(text, str):
        return ""
    words = text.split()
    return " ".join(words[:max_words])


def load_dataset(data_path: str = None) -> pd.DataFrame:
    """Locates and loads the raw dataset without modifying the source file.

    Strips leading and trailing spaces from the 'text' column.
    """
    if data_path and os.path.exists(data_path):
        target_path = data_path
    else:
        # Search candidate paths
        candidate_paths = [
            os.path.join("data", "clean.csv"),
            os.path.join("data", "data.csv"),
            os.path.join("raw", "data.csv"),
            os.path.join("..", "data", "clean.csv"),
            os.path.join("..", "data", "data.csv"),
            os.path.join("..", "raw", "data.csv"),
        ]
        target_path = None
        for p in candidate_paths:
            if os.path.exists(p):
                target_path = p
                break
        if not target_path:
            raise FileNotFoundError(
                f"Could not find dataset in candidate paths: {candidate_paths}"
            )

    print(f"[Preprocessing] Loading dataset from: {target_path}")
    df = pd.read_csv(target_path)

    # Validate columns
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError(f"Dataset must contain 'text' and 'label' columns. Found: {df.columns}")

    # Strip leading/trailing whitespace from text without modifying original CSV file
    df["text_raw"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(int)

    return df


def preprocess_and_split(
    data_path: str = None,
    output_dir: str = "data",
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocesses text columns for Conditions A and B and creates a 80/20 stratified split.

    Columns produced:
    - label: 0 (Human) or 1 (AI)
    - text_raw: Full raw text (for handcrafted features in Condition A)
    - text_clean: Full cleaned text (for TF-IDF in Condition A)
    - text_raw_trunc: First 50 words of raw text (for handcrafted features in Condition B)
    - text_clean_trunc: First 50 words cleaned (for TF-IDF in Condition B)
    """
    df = load_dataset(data_path)

    print("[Preprocessing] Generating cleaned and truncated text columns...")
    # Cleaned text for TF-IDF (Condition A)
    df["text_clean"] = df["text_raw"].apply(clean_text_for_tfidf)

    # Truncated versions (Condition B - first 50 words)
    df["text_raw_trunc"] = df["text_raw"].apply(lambda t: truncate_text(t, max_words=50))
    df["text_clean_trunc"] = df["text_raw_trunc"].apply(clean_text_for_tfidf)

    # Select only required columns
    cols_to_keep = [
        "label",
        "text_raw",
        "text_clean",
        "text_raw_trunc",
        "text_clean_trunc",
    ]
    df_processed = df[cols_to_keep].copy()

    # Stratified 80/20 train/test split
    print(f"[Preprocessing] Splitting dataset ({len(df_processed)} rows) 80/20 with stratify=label, random_state={random_state}...")
    train_df, test_df = train_test_split(
        df_processed,
        test_size=test_size,
        random_state=random_state,
        stratify=df_processed["label"],
    )

    # Save to output_dir
    os.makedirs(output_dir, exist_ok=True)
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"[Preprocessing] Train set saved to: {train_path} ({len(train_df)} rows)")
    print(f"[Preprocessing] Test set saved to:  {test_path} ({len(test_df)} rows)")
    print(f"[Preprocessing] Train class distribution: Human (0) = {(train_df['label']==0).sum()}, AI (1) = {(train_df['label']==1).sum()}")
    print(f"[Preprocessing] Test class distribution:  Human (0) = {(test_df['label']==0).sum()}, AI (1) = {(test_df['label']==1).sum()}")

    return train_df, test_df


if __name__ == "__main__":
    preprocess_and_split()
