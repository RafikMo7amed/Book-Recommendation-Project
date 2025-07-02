import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np 

def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"Successfully loaded {len(data)} books from {file_path}")
        return data
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}. Check file format.")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading the file: {e}")
        return None

def analyze_content_lengths(file_path):
    
    json_data = load_json_data(file_path)
    if json_data is None:
        return

    df = pd.DataFrame(json_data)
    
    if 'content' not in df.columns:
        print("Error: 'content' column not found in the DataFrame. Please ensure the input JSON has a 'content' field.")
        return

   
    df['word_count'] = df['content'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

    
    print("\n--- Content Word Count Statistics ---")
    print(df['word_count'].describe())

    
    print("\n--- Percentiles of Word Count ---")
    percentiles = [50, 75, 80, 90, 95, 99, 100]
    for p in percentiles:
        val = np.percentile(df['word_count'], p)
        print(f"{p}th percentile: {val:.0f} words")

    
    plt.figure(figsize=(12, 6))
    sns.histplot(df['word_count'], bins=50, kde=True) # kde=True لعمل تقدير لكثافة الكيرف
    plt.title('Distribution of Book Content Word Counts')
    plt.xlabel('Word Count')
    plt.ylabel('Number of Books')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    
    plt.figure(figsize=(10, 4))
    sns.boxplot(x=df['word_count'])
    plt.title('Box Plot of Book Content Word Counts')
    plt.xlabel('Word Count')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    
    print("\n--- Top 10 Longest Contents ---")
    print(df.nlargest(10, 'word_count')[['title', 'word_count']])


if __name__ == "__main__":
   
    input_data_file = "D:\\Graduation Project\\project\\data\\enriched_books_only_covers.json" 
    
    analyze_content_lengths(input_data_file)
