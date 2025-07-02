import json
import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import logging
import os
from datetime import datetime
from bs4 import BeautifulSoup
import re
import numpy as np

# --- Setup Logging ---
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f'classification_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Temporary Saves Directory ---
temp_save_dir = "temp_saves"
os.makedirs(temp_save_dir, exist_ok=True)

def load_json_data(file_path):
    """Load JSON file safely."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        logger.info(f"Loaded {len(data)} books from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading file {file_path}: {e}")
        return None

def save_classified_data(df, output_file):
    """Save classified data as JSON."""
    data = df.to_dict(orient='records')
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        logger.info(f"Final classified data saved to: {output_file}")
    except Exception as e:
        logger.error(f"Error saving final file: {e}")

def clean_text_for_classification(text):
    """Cleanning text before classification."""
    if not isinstance(text, str) or not text.strip():
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s.,!?\'\"-]", "", text)
    text = re.sub(r'\s+', ' ', text).strip().lower()
    return text

def classify_books(json_data, candidate_labels, batch_size=8, max_length=512, save_interval=100, chunk_stride=128):
    """
    Classify books using Zero-Shot Classification with text chunking and mean score aggregation.
    """
    device = 0 if torch.cuda.is_available() else -1
    if device == 0:
        logger.info("Using GPU for classification.")
        torch.cuda.empty_cache()
    else:
        logger.info("Using CPU for classification.")

    try:
        model_name = "facebook/bart-large-mnli"
        classifier = pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        logger.info(f"Loaded model and tokenizer: {model_name}.")
    except Exception as e:
        logger.error(f"Error loading model or tokenizer: {e}", exc_info=True)
        return None

    df = pd.DataFrame(json_data)
    if df.empty:
        logger.error("Empty data, cannot classify.")
        return None

    df['content'] = df['content'].apply(clean_text_for_classification)
    df = df[df['content'].str.strip() != ''].copy()
    df['word_count'] = df['content'].apply(lambda x: len(x.split()))
    logger.info(f"Number of books to classify after cleaning: {len(df)}")

    final_classifications = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Classifying books"):
        book_content = row['content']
        
        # This will hold scores for each label from all chunks
        # e.g., {'Leadership': [0.9, 0.85], 'Productivity': [0.2, 0.3]}
        chunk_scores = {label: [] for label in candidate_labels}

        if not book_content.strip():
            # If content is empty, append zero scores for all labels
            final_classifications.append({label: 0.0 for label in candidate_labels})
            continue
        
        # Tokenize the entire content and create chunks
        tokenized_content = tokenizer(
            book_content,
            max_length=max_length,
            truncation=True,
            return_overflowing_tokens=True,
            stride=chunk_stride,
            padding=True,
            return_tensors="pt"
        )
        
        input_ids_chunks = tokenized_content['input_ids']

        if input_ids_chunks.shape[0] == 0:
            final_classifications.append({label: 0.0 for label in candidate_labels})
            continue

        # Convert token chunks back to string chunks for the pipeline
        chunk_texts = [tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids_chunks]

        # Classify each chunk
        for chunk_text in chunk_texts:
            try:
                result = classifier(
                    chunk_text,
                    candidate_labels,
                    multi_label=True
                )
                # Store the score for each label from this chunk
                for label, score in zip(result['labels'], result['scores']):
                    chunk_scores[label].append(score)
            except Exception as e:
                logger.error(f"Error classifying a chunk for book '{row['title']}': {e}")
        
        
        aggregated_scores = {label: 0.0 for label in candidate_labels}
        for label, scores in chunk_scores.items():
            if scores:
                aggregated_scores[label] = np.mean(scores)

        final_classifications.append(aggregated_scores)

        # Save progress intermittently
        if (len(final_classifications) % save_interval == 0) or (len(final_classifications) == len(df)):
            temp_df_to_save = df.iloc[:len(final_classifications)].copy()
            temp_df_to_save['classifications'] = final_classifications
            save_classified_data(temp_df_to_save, os.path.join(temp_save_dir, f'temp_classified_books_{len(final_classifications)}.json'))
            logger.info(f"Saved temporary data up to book {len(final_classifications)}")

    df['classifications'] = final_classifications
    logger.info(f"Finished classifying all {len(df)} books.")

    return df

if __name__ == "__main__":
    input_file = "D:\\Graduation Project\\project\\data\\enriched_books_only_covers.json"
    output_file = "D:\\Graduation Project\\project\\data\\classified_books.json"
    
    candidate_labels = [
        "Personal Development", "Career Success", "Strengthening Relationships",
        "Habit Improvement", "Productivity Enhancement", "Building Self-Confidence",
        "Leadership", "Time Management", "Emotional Intelligence",
        "Critical Thinking", "Finance and Investment", "Happiness and Well-Being",
        "Real-Life Stories", "Practical Steps", "Inspiration and Motivation"
    ]
    
    json_data = load_json_data(input_file)
    if json_data:
        df_classified = classify_books(json_data, candidate_labels)
        if df_classified is not None:
            save_classified_data(df_classified, output_file)
            print("\nClassification process completed.")
            print(df_classified[['title', 'classifications']].head())