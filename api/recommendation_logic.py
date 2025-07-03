import pandas as pd

from . import config
def get_for_you_recommendations(df: pd.DataFrame, preferences: dict) -> pd.DataFrame:
    """Gets personalized recommendations based on user preferences."""
    user_labels = set(preferences.get('goals', []) + preferences.get('skills', []) + preferences.get('content_types', []))
    if preferences.get('habit_building', False):
        user_labels.add("Habit Improvement")
    
    if not user_labels:
        return pd.DataFrame() # Return empty if no preferences

    df['relevance_score'] = df['classifications'].apply(
        lambda scores: sum(scores.get(label, 0) for label in user_labels) / len(user_labels) if user_labels else 0
    )
    
   
    final_books = df.sort_values(by='relevance_score', ascending=False).head(config.N_RECOMMENDATIONS)
    return final_books

def get_top_rated_books(df: pd.DataFrame) -> pd.DataFrame:
    """Gets a general list of top-rated books based on key categories."""
    top_rated_categories = config.TOP_RATED_CATEGORIES
    
    # Calculate a "top_rated_score" for each book by summing scores of important categories
    df['top_rated_score'] = df['classifications'].apply(
        lambda scores: sum(scores.get(cat, 0) for cat in top_rated_categories)
    )
    
    # Return the top N books based on this new score
    return df.sort_values(by='top_rated_score', ascending=False).head(config.N_RECOMMENDATIONS)