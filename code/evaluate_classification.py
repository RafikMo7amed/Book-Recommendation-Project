# development_scripts/evaluate_classification.py
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
CLASSIFIED_BOOKS_PATH = os.path.join(DATA_DIR, "classified_books.json")

def load_classified_data(file_path):
    try:
        df = pd.read_json(file_path)
        print(f"Successfully loaded {len(df)} books.")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_confidence_scores(df):
    print("\n--- 1. Analyzing Model Confidence ---")
    df['max_score'] = df['classifications'].apply(lambda x: max(x.values()) if isinstance(x, dict) and x else 0)
    print(df['max_score'].describe())
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['max_score'], bins=30, kde=True)
    plt.title("Distribution of Max Confidence Scores per Book")
    plt.xlabel("Max Confidence Score")
    plt.show()

def analyze_label_distribution(df, threshold=0.7):
    print(f"\n--- 2. Analyzing Label Distribution (Threshold > {threshold}) ---")
    df['top_labels'] = df['classifications'].apply(
        lambda scores: [label for label, score in scores.items() if score >= threshold] if isinstance(scores, dict) else []
    )
    label_counts = pd.Series([label for sublist in df['top_labels'] for label in sublist]).value_counts()
    if label_counts.empty:
        print("No labels found above the threshold.")
        return
        
    print("Frequency of each label being assigned:")
    print(label_counts)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=label_counts.values, y=label_counts.index, orient='h')
    plt.title(f"Label Distribution (Scores > {threshold})")
    plt.show()

if __name__ == "__main__":
    df = load_classified_data(CLASSIFIED_BOOKS_PATH)
    if df is not None:
        analyze_confidence_scores(df)
        analyze_label_distribution(df)