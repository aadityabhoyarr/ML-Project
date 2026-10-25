"""Feature Engineering Module for AI vs Human Text Detection.

This module provides:
1. TF-IDF vectorizer configuration (unigrams + bigrams, top 5000 features).
2. Length-independent handcrafted feature extraction:
   - Average sentence length (words per sentence)
   - Standard deviation of sentence length (uniformity metric; 0 for single sentence)
   - Vocabulary richness: Type-Token Ratio (unique words / total words)
   - Average word length
   - Punctuation ratio (punctuation characters / total characters)
   - Comma ratio (commas / total words)
   - Uppercase-letter ratio (uppercase characters / total characters)
   - Digit ratio (digit characters / total characters)
3. Scikit-learn compatible transformer `HandcraftedFeatureExtractor` for use in sklearn Pipelines.
"""

import string
import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

# NLTK sentence tokenization with regex fallback
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    # Quick probe to ensure tokenizer models are loaded
    sent_tokenize("Test sentence.")
    NLTK_AVAILABLE = True
except Exception:
    NLTK_AVAILABLE = False


def split_sentences(text: str) -> list[str]:
    """Splits text into sentences using NLTK sent_tokenize if available,

    falling back to a robust regex pattern.
    """
    if not text or not isinstance(text, str):
        return []
    if NLTK_AVAILABLE:
        try:
            sentences = sent_tokenize(text)
            if sentences:
                return sentences
        except Exception:
            pass

    # Regex fallback: split by punctuation (.!?) followed by space or end of line
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def extract_handcrafted_features(text: str) -> list[float]:
    """Computes 8 length-independent linguistic and stylistic features from raw text.

    Features returned (in exact order):
    1. avg_sentence_len: Average words per sentence
    2. std_sentence_len: Standard deviation of sentence length in words (0 if <= 1 sentence)
    3. type_token_ratio: Vocabulary richness (unique words / total words)
    4. avg_word_len: Average characters per word
    5. punct_ratio: Punctuation characters / total characters
    6. comma_ratio: Commas per word
    7. upper_ratio: Uppercase letters / total characters
    8. digit_ratio: Digits / total characters

    All features use ratios and averages to ensure independence from raw text length.
    """
    if not isinstance(text, str) or not text.strip():
        return [0.0] * 8

    text_clean = text.strip()
    total_chars = len(text_clean)

    # 1. Sentences
    sentences = split_sentences(text_clean)
    if not sentences:
        sentences = [text_clean]

    sentence_word_counts = [len(s.split()) for s in sentences]
    num_sentences = len(sentences)

    # Feature 1: Average sentence length
    avg_sentence_len = sum(sentence_word_counts) / num_sentences if num_sentences > 0 else 0.0

    # Feature 2: Standard deviation of sentence length (0 if 1 sentence or empty)
    std_sentence_len = float(np.std(sentence_word_counts)) if num_sentences > 1 else 0.0

    # Words for lexical features (lowercased, stripped of surrounding punctuation)
    tokens = text_clean.split()
    words = [t.lower().strip(string.punctuation) for t in tokens]
    words = [w for w in words if w]
    num_words = len(words)

    # Feature 3: Type-Token Ratio (Vocabulary richness)
    if num_words > 0:
        unique_words = len(set(words))
        type_token_ratio = unique_words / num_words
    else:
        type_token_ratio = 0.0

    # Feature 4: Average word length
    if num_words > 0:
        avg_word_len = sum(len(w) for w in words) / num_words
    else:
        avg_word_len = 0.0

    # Feature 5: Punctuation ratio (punctuation characters / total characters)
    punct_count = sum(1 for c in text_clean if c in string.punctuation)
    punct_ratio = punct_count / total_chars if total_chars > 0 else 0.0

    # Feature 6: Comma ratio (commas per word)
    comma_count = text_clean.count(",")
    comma_ratio = comma_count / num_words if num_words > 0 else 0.0

    # Feature 7: Uppercase letter ratio
    upper_count = sum(1 for c in text_clean if c.isupper())
    upper_ratio = upper_count / total_chars if total_chars > 0 else 0.0

    # Feature 8: Digit ratio
    digit_count = sum(1 for c in text_clean if c.isdigit())
    digit_ratio = digit_count / total_chars if total_chars > 0 else 0.0

    return [
        float(avg_sentence_len),
        float(std_sentence_len),
        float(type_token_ratio),
        float(avg_word_len),
        float(punct_ratio),
        float(comma_ratio),
        float(upper_ratio),
        float(digit_ratio),
    ]


HANDCRAFTED_FEATURE_NAMES = [
    "avg_sentence_len",
    "std_sentence_len",
    "type_token_ratio",
    "avg_word_len",
    "punct_ratio",
    "comma_ratio",
    "upper_ratio",
    "digit_ratio",
]

HANDCRAFTED_FEATURE_LABELS = [
    "Avg Sentence Length (words/sent)",
    "Std Sentence Length (sent variation)",
    "Vocabulary Richness (Type-Token Ratio)",
    "Avg Word Length (chars/word)",
    "Punctuation Ratio (punct/chars)",
    "Comma Ratio (commas/words)",
    "Uppercase Ratio (upper/chars)",
    "Digit Ratio (digits/chars)",
]


def extract_handcrafted_dict(text: str) -> dict[str, float]:
    """Returns handcrafted features as a clean dictionary with readable keys."""
    vals = extract_handcrafted_features(text)
    return dict(zip(HANDCRAFTED_FEATURE_LABELS, vals))


class HandcraftedFeatureExtractor(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer that converts raw text into an (N, 8)

    matrix of handcrafted linguistic and stylistic features.
    """

    def __init__(self):
        self.feature_names = HANDCRAFTED_FEATURE_NAMES

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "values"):
            texts = X.values
        else:
            texts = list(X)
        features = [extract_handcrafted_features(t) for t in texts]
        return np.array(features, dtype=np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names)


def get_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    """Configures TF-IDF vectorizer with unigrams + bigrams and English stop words."""
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
    )


if __name__ == "__main__":
    sample_text = (
        "Artificial Intelligence represents a transformative leap in computational technology. "
        "However, does it write like a human? In this study, we evaluate stylistic divergence."
    )
    print("Testing handcrafted feature extraction:")
    for k, v in extract_handcrafted_dict(sample_text).items():
        print(f"  {k}: {v:.4f}")
