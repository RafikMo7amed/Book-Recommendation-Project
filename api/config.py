import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")


CLASSIFIED_BOOKS_PATH = os.path.join(DATA_DIR, "classified_books.json")
BEST_PARAMS_PATH = os.path.join(DATA_DIR, "best_summary_params.json")


SUMMARIZATION_MODEL = "google/pegasus-large"


N_RECOMMENDATIONS = 7


TOP_RATED_CATEGORIES = [
    "Personal Development", "Career Success", "Productivity Enhancement",
    "Happiness and Well-Being", "Finance and Investment", "Leadership",
    "Critical Thinking"
]