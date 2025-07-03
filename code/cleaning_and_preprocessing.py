import json
import pandas as pd
import re
from tqdm import tqdm
from rake_nltk import Rake
from langdetect import detect, DetectorFactory
import nltk
from bs4 import BeautifulSoup


nltk.download('stopwords')
nltk.download('punkt')
DetectorFactory.seed = 0  

def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error loading the file: {e}")
        return None

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text()

    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    text = re.sub(r"[^a-zA-Z0-9\s.,!?\'\"-]", "", text)

    text = re.sub(r'\s+', ' ', text).strip()

    text = text.lower()

    return text

def is_english(text):

    if not isinstance(text, str) or len(text) < 20:
        return False
    try:

        detections = detect(text)
        return detections == 'en'
    except:
        return False

def extract_keywords(text, num_keywords=7):
    rake = Rake()
    rake.extract_keywords_from_text(text)
    
    ranked_phrases = rake.get_ranked_phrases()
    return ranked_phrases[:num_keywords]

def preprocess_books_data(json_data, min_length=100, max_length=6000):

    titles = []
    urls = []
    contents = []
    keywords_list = []
    
    for book in tqdm(json_data, desc="Processing books"):
        title = book.get('title', '')
        url = book.get('url', '')
        content = book.get('content', '')
        
        cleaned_content = clean_text(content)
        
        if (cleaned_content and 
            min_length <= len(cleaned_content) <= max_length and 
            is_english(cleaned_content) and 
            title.strip()):
            
            titles.append(title)
            urls.append(url)
            contents.append(cleaned_content)
            keywords_list.append(extract_keywords(cleaned_content))
    
    df = pd.DataFrame({
        'title': titles,
        'url': urls,
        'content': contents,
        'keywords': keywords_list
    })
    
    df = df.drop_duplicates(subset=['title', 'content'])
    
    return df

def save_preprocessed_data(df, output_file):
    data = df.to_dict(orient='records')
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Books saved in: {output_file}")
    except Exception as e:
        print(f"Error saving the file: {e}")


if __name__ == "__main__":
    input_file = "D:\\Graduation Project\\project\\data\\books.json"
    output_file = "D:\\Graduation Project\\project\\data\\preprocessed_books.json"

    json_data = load_json_data(input_file)
    if json_data:

        df = preprocess_books_data(json_data, min_length=100, max_length=6000)

        save_preprocessed_data(df, output_file)
        print(f"Number of books after preprocessing: {len(df)}")
        print("\nSample of preprocessed data:")
        print(df.head())

        print("\nKeywords sample for a book:")
        if not df.empty:
            print(df['keywords'].iloc[0])
        else:
            print("No books processed to show keywords sample.")