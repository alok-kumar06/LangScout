"""Central configuration: paths, seeds and constants used across the project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "Language_Detection.csv"
EVERYDAY_CSV = ROOT / "data" / "samples" / "everyday_sentences.csv"
HINGLISH_TEST_CSV = ROOT / "data" / "samples" / "hinglish_test.csv"
CONVERSATIONAL_HINDI = ROOT / "data" / "samples" / "conversational_hindi.txt"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "langscout_model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"
REPORT_DIR = ROOT / "reports"

SEED = 42

# The Kaggle dataset contains two spelling mistakes in its labels.
LABEL_FIXES = {"Portugeese": "Portuguese", "Sweedish": "Swedish"}

# Long Wikipedia paragraphs are cut into chunks of at most this many characters,
# so that training data looks like the short sentences users actually type.
MAX_CHUNK_CHARS = 200
MIN_CHUNK_CHARS = 5

# Row-level split (done BEFORE chunking, so chunks of one article never leak across splits).
TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.70, 0.15, 0.15

# Fallback used when no threshold was tuned on the validation set.
DEFAULT_CONFIDENCE_THRESHOLD = 0.50

# Input lengths (characters) used for the short-text robustness study.
TRUNCATION_LENGTHS = [3, 5, 8, 12, 20, 40, 80, 160]

HINGLISH = "Hinglish"          # Hindi written in English letters (an 18th class)
SEED_VARIANTS = 3              # spelling variants generated per conversational seed sentence
