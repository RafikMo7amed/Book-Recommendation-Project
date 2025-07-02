import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

# --- PATHS ---
CLASSIFIED_BOOKS_PATH = os.path.join(DATA_DIR, "classified_books.json")
BEST_PARAMS_PATH = os.path.join(DATA_DIR, "best_summary_params.json")

# --- MODEL NAME ---
SUMMARIZATION_MODEL = "google/pegasus-large"

# --- API & RECOMMENDATION SETTINGS ---
N_RECOMMENDATIONS = 7

# --- Categories considered for the "Top Rated" section ---
TOP_RATED_CATEGORIES = [
    "Personal Development", "Career Success", "Productivity Enhancement",
    "Happiness and Well-Being", "Finance and Investment", "Leadership",
    "Critical Thinking"
]
