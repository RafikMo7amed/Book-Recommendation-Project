import pandas as pd
import requests
import json
import time
import os
import re
from tqdm import tqdm
import unicodedata
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

# **ملاحظة هامة:**
# تأكد من تثبيت fuzzywuzzy و python-Levenshtein
# pip install fuzzywuzzy python-Levenshtein

def load_json_data(file_path):
    """تحميل ملف JSON مع التعامل مع الأخطاء"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        print(f"Successfully loaded data from {file_path}")
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

def save_dataframe_to_json(df, output_file):
    """حفظ DataFrame كـ JSON"""
    data = df.to_dict(orient='records')
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Data saved successfully to: {output_file}")
    except Exception as e:
        print(f"Error saving the DataFrame to JSON: {e}")

def normalize_title_for_comparison(title):
    """ينظف العنوان للمقارنة: lowercase، إزالة المسافات الزائدة، إزالة علامات التشكيل."""
    if not isinstance(title, str):
        return ""
    title = title.lower().strip()
    title = re.sub(r'\s+', ' ', title)
    # إزالة علامات التشكيل (diacritics) للحصول على مطابقة أفضل
    title = unicodedata.normalize('NFKD', title).encode('ascii', 'ignore').decode('utf-8')
    return title

def get_book_cover_from_google_books(title, retries=5, delay_between_retries=5): # تم تغيير اسم الدالة
    """
    يبحث عن رابط غلاف الكتاب فقط باستخدام Google Books API.
    يحاول تحسين الدقة باختيار أفضل تطابق من نتائج متعددة باستخدام fuzzy matching.
    """
    base_url = "https://www.googleapis.com/books/v1/volumes"
    
    normalized_search_title = normalize_title_for_comparison(title)

    for attempt in range(retries):
        try:
            params = {
                'q': normalized_search_title,
                'langRestrict': 'en',
                'maxResults': 15 
            }

            response = requests.get(base_url, params=params, timeout=25) 
            response.raise_for_status()

            data = response.json()

            best_match_cover_url = None
            best_match_score = -1 

            if 'items' in data and data['items']:
                potential_matches = [] 

                for item in data['items']:
                    volume_info = item.get('volumeInfo', {})
                    
                    if volume_info.get('printType') != 'BOOK':
                        continue

                    current_title = normalize_title_for_comparison(volume_info.get('title', ''))
                    
                    # استخدام fuzzy matching لمقارنة العنوان
                    title_fuzz_score = fuzz.ratio(normalized_search_title, current_title)
                    
                    if 'subtitle' in volume_info:
                        normalized_subtitle = normalize_title_for_comparison(volume_info['subtitle'])
                        subtitle_fuzz_score = fuzz.ratio(normalized_search_title, normalized_subtitle)
                        title_fuzz_score = max(title_fuzz_score, subtitle_fuzz_score)

                    # درجة المؤلف هنا لم تعد تُستخدم في حساب overall_score بشكل مباشر
                    # لكن الكود يستمر في استخدامها ضمن logic المطابقة لو وجدت
                    # بما أننا لا نحتاج المؤلف كـ output، فقد يتم تجاهل استخراجه هنا
                    # لكن للحفاظ على قوة المنطق المطابق، يمكن تركه.
                    current_authors = [normalize_title_for_comparison(a) for a in volume_info.get('authors', [])]
                    author_fuzz_score = 0
                    if current_authors:
                        for auth in current_authors:
                            author_fuzz_score = max(author_fuzz_score, fuzz.ratio(normalized_search_title, auth))
                    
                    overall_score = title_fuzz_score * 0.9 + author_fuzz_score * 0.1 # ما زلنا نستخدم المؤلف لتحسين درجة المطابقة

                    potential_matches.append({
                        'score': overall_score,
                        # 'author': ", ".join(volume_info.get('authors', [])) if volume_info.get('authors') else "Unknown", # تم حذف المؤلف من هنا
                        'cover_url': (volume_info.get('imageLinks', {}).get('extraLarge') or
                                      volume_info.get('imageLinks', {}).get('large') or
                                      volume_info.get('imageLinks', {}).get('medium') or
                                      volume_info.get('imageLinks', {}).get('small') or
                                      volume_info.get('imageLinks', {}).get('thumbnail') or
                                      volume_info.get('imageLinks', {}).get('smallThumbnail'))
                    })
                
                if potential_matches:
                    best_match = max(potential_matches, key=lambda x: x['score'])
                    best_match_score = best_match['score']

                    if best_match_score >= 90:
                        return best_match['cover_url'] # نرجع رابط الغلاف فقط
                    elif best_match_score >= 75: 
                        print(f"Warning: Medium match found for '{title}' (score: {best_match_score:.2f}). Cover might be inaccurate.")
                        return best_match['cover_url'] # نرجع رابط الغلاف
                    else:
                        return None # لا يوجد تطابق جيد، نرجع None
            else:
                return None # لم يتم العثور على الكتاب
        
        except requests.exceptions.Timeout:
            print(f"Request timed out for '{title}'. Attempt {attempt + 1}/{retries}. Retrying in {delay_between_retries}s...")
            time.sleep(delay_between_retries)
        except requests.exceptions.ConnectionError as e:
            print(f"Connection error for '{title}': {e}. Attempt {attempt + 1}/{retries}. Retrying in {delay_between_retries}s...")
            time.sleep(delay_between_retries)
        except requests.exceptions.RequestException as e:
            print(f"Request failed for '{title}': {e}. Attempt {attempt + 1}/{retries}. Retrying in {delay_between_retries}s...")
            time.sleep(delay_between_retries)
        except json.JSONDecodeError:
            print(f"JSON decoding failed for '{title}'. Attempt {attempt + 1}/{retries}. Retrying in {delay_between_retries}s...")
            time.sleep(delay_between_retries)
        except Exception as e:
            print(f"An unexpected error occurred for '{title}': {e}. Attempt {attempt + 1}/{retries}. Retrying in {delay_between_retries}s...")
            time.sleep(delay_between_retries)
            
    return None # فشل بعد كل المحاولات في جلب الغلاف

def download_cover_image(image_url, book_title, covers_dir="book_covers"):
    """
    يقوم بتحميل صورة الغلاف إلى مجلد محدد.
    يستخدم اسم الكتاب لإنشاء اسم ملف فريد.
    """
    if not image_url:
        return None

    os.makedirs(covers_dir, exist_ok=True)

    if not isinstance(book_title, str):
        book_title = str(book_title)

    # تنظيف اسم الكتاب لجعله مناسباً لاسم ملف
    safe_title = re.sub(r'[^\w\s-]', '', book_title).strip()
    safe_title = re.sub(r'\s+', '_', safe_title)
    
    if len(safe_title) > 150: # قص اسم الملف لتجنب مشاكل أنظمة التشغيل
        safe_title = safe_title[:150]

    _, ext = os.path.splitext(image_url.split('?')[0])
    if not ext or len(ext) > 5 or not ext.lower() in ['.jpg', '.jpeg', '.png', '.gif']: 
        ext = '.jpg'

    file_name = f"{safe_title}{ext}"
    file_path = os.path.join(covers_dir, file_name)

    # معالجة حالة وجود ملف بنفس الاسم بالفعل لتجنب الكتابة فوقه
    counter = 1
    while os.path.exists(file_path):
        file_name = f"{safe_title}_{counter}{ext}"
        file_path = os.path.join(covers_dir, file_name)
        counter += 1

    try:
        response = requests.get(image_url, stream=True, timeout=30) 
        response.raise_for_status()

        with open(file_path, 'wb') as out_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    out_file.write(chunk)
        return file_path
    except requests.exceptions.RequestException as e:
        print(f"Error downloading cover for '{book_title}' from '{image_url}': {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred during download for '{book_title}': {e}")
        return None

def enrich_books_with_covers(df, covers_output_dir="book_covers", delay_seconds=0.3): # تم تغيير اسم الدالة
    """
    تضيف بيانات الغلاف (مع تنزيل الغلاف) إلى DataFrame.
    """
    cover_urls = []
    cover_local_paths = []

    print("\n--- Enriching books with Cover metadata ---")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Fetching cover images"):
        title = row['title']
        
        # تم تغيير اسم الدالة المستدعاة هنا
        cover_api_url = get_book_cover_from_google_books(title) 
        
        local_cover_path = None
        if cover_api_url:
            local_cover_path = download_cover_image(cover_api_url, title, covers_output_dir)
        
        cover_urls.append(cover_api_url)
        cover_local_paths.append(local_cover_path)

        time.sleep(delay_seconds) 

    # لم نعد نضيف عمود 'author'
    df['cover_url'] = cover_urls
    df['cover_local_path'] = cover_local_paths
    
    return df

# الكود الرئيسي لتشغيل عملية إثراء البيانات فقط
if __name__ == "__main__":
    # مسارات الملفات (يرجى التأكد من ضبطها حسب جهازك)
    input_preprocessed_json_file = "D:\\Graduation Project\\project\\data\\preprocessed_books.json" 
    output_enriched_json_file = "D:\\Graduation Project\\project\\data\\enriched_books_only_covers.json" # تم تغيير اسم ملف الإخراج
    covers_directory = "D:\\Graduation Project\\project\\data\\book_covers" 

    # 1. تحميل البيانات المعالجة مسبقاً
    preprocessed_data_list = load_json_data(input_preprocessed_json_file)
    if preprocessed_data_list is None:
        print("Failed to load preprocessed data. Please ensure 'preprocessed_books.json' exists and is valid.")
    else:
        df_preprocessed = pd.DataFrame(preprocessed_data_list)
        print(f"Loaded {len(df_preprocessed)} books from preprocessed data.")

        # 2. إثراء البيانات بالأغلفة فقط
        # تم تغيير اسم الدالة المستدعاة هنا
        df_enriched = enrich_books_with_covers(df_preprocessed, covers_output_dir=covers_directory, delay_seconds=0.3)
        
        # 3. حفظ البيانات المثرية
        save_dataframe_to_json(df_enriched, output_enriched_json_file)
        
        print(f"\nTotal books processed for cover enrichment: {len(df_enriched)}")
        print("\n--- Sample of final enriched data (Covers only) ---")
        # عرض العناوين وروابط الأغلفة والمسارات المحلية فقط
        print(df_enriched[['title', 'cover_url', 'cover_local_path']].head())

        # عرض ملخص لنتائج جلب البيانات
        no_cover_url_found_count = df_enriched[df_enriched['cover_url'].isnull()].shape[0]
        no_local_cover_downloaded_count = df_enriched[df_enriched['cover_local_path'].isnull()].shape[0]
        
        print(f"\nSummary of Cover Enrichment:")
        print(f" - Books with no cover URL found (even if API call successful): {no_cover_url_found_count}")
        print(f" - Books where local cover download failed: {no_local_cover_downloaded_count}")
        print(f" - Books with successfully retrieved and downloaded cover: {len(df_enriched) - no_local_cover_downloaded_count}")