import json
import pandas as pd
import re
from tqdm import tqdm
from rake_nltk import Rake
from langdetect import detect, DetectorFactory
import nltk
from bs4 import BeautifulSoup

# إعدادات أساسية
nltk.download('stopwords')
nltk.download('punkt')
DetectorFactory.seed = 0  # لضمان نتائج ثابتة في اكتشاف اللغة

def load_json_data(file_path):
    """تحميل ملف JSON مع التعامل مع الأخطاء"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except Exception as e:
        print(f"Error loading the file: {e}")
        return None

def clean_text(text):
    """
    تنظيف النص من المسافات الزائدة، علامات الترقيم الغريبة،
    الروابط، وأي بقايا HTML، وتحويله إلى lowercase.
    """
    if not isinstance(text, str):
        return ""

    # إزالة ترويسات HTML
    text = BeautifulSoup(text, "html.parser").get_text()

    # إزالة الروابط (URLs)
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)

    # إزالة الأحرف غير ASCII (تبقى الأحرف الإنجليزية والأرقام وعلامات الترقيم الأساسية)
    # هذا يضمن أن النص يكون خالياً من أحرف غريبة قد تنتج عن الترميز أو اللغات الأخرى.
    text = re.sub(r"[^\x00-\x7F]+", " ", text)

    # إزالة علامات الترقيم والأحرف الخاصة باستثناء النقطة والفاصلة وعلامات الاستفهام والتعجب والشرطة الواصلة
    # وأيضاً الأرقام والأحرف الأبجدية الإنجليزية.
    # تم دمج وتنقيح التعبيرات المنتظمة هنا لتجنب التكرار.
    text = re.sub(r"[^a-zA-Z0-9\s.,!?\'\"-]", "", text)

    # توحيد المسافات: إزالة المسافات الزائدة (أكثر من مسافة واحدة) واستبدالها بمسافة واحدة
    text = re.sub(r'\s+', ' ', text).strip()

    # تحويل النص بالكامل إلى أحرف صغيرة (lowercase) للتوحيد
    text = text.lower()

    return text

def is_english(text):
    """
    التأكد أن النص باللغة الإنجليزية بدرجة ثقة معينة.
    تجنب النصوص القصيرة جداً التي قد تعطي نتائج خاطئة في اكتشاف اللغة.
    """
    if not isinstance(text, str) or len(text) < 20: # حد أدنى لطول النص لتجنب الأخطاء
        return False
    try:
        # langdetect يمكن أن يعطي نتائج بثقة مختلفة
        # يمكننا استخدام detect_langs للحصول على قائمة باللغات مع درجات الثقة
        detections = detect(text)
        return detections == 'en' # هنا نتحقق مباشرة من اكتشاف اللغة الإنجليزية
    except:
        return False

def extract_keywords(text, num_keywords=7):
    """
    استخراج كلمات مفتاحية من النص باستخدام RAKE.
    تم زيادة عدد الكلمات المستخرجة للحصول على تغطية أفضل.
    """
    rake = Rake()
    rake.extract_keywords_from_text(text)
    # الحصول على العبارات المرتبة (ranked phrases)
    ranked_phrases = rake.get_ranked_phrases()
    return ranked_phrases[:num_keywords]

def preprocess_books_data(json_data, min_length=100, max_length=6000):
    """
    معالجة البيانات وتحويلها إلى DataFrame مع تطبيق الفلاتر والتحسينات.
    تم ضبط max_length لتناسب أطول نص لديك.
    """
    titles = []
    urls = []
    contents = []
    keywords_list = []
    
    # معالجة كل كتاب
    for book in tqdm(json_data, desc="Processing books"):
        title = book.get('title', '')
        url = book.get('url', '')
        content = book.get('content', '')
        
        # تنظيف النص
        cleaned_content = clean_text(content)
        
        # التحقق من الشروط بعد التنظيف
        if (cleaned_content and 
            min_length <= len(cleaned_content) <= max_length and 
            is_english(cleaned_content) and 
            title.strip()):  # التأكد أن العنوان ليس فارغاً
            
            titles.append(title)
            urls.append(url)
            contents.append(cleaned_content)
            keywords_list.append(extract_keywords(cleaned_content))
    
    # إنشاء DataFrame
    df = pd.DataFrame({
        'title': titles,
        'url': urls,
        'content': contents,
        'keywords': keywords_list
    })
    
    # إزالة التكرار بناءً على العنوان والمحتوى
    df = df.drop_duplicates(subset=['title', 'content'])
    
    return df

def save_preprocessed_data(df, output_file):
    """حفظ البيانات كـ JSON"""
    data = df.to_dict(orient='records')
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
        print(f"Books saved in: {output_file}")
    except Exception as e:
        print(f"Error saving the file: {e}")

# مثال على الاستخدام
if __name__ == "__main__":
    # مسار ملف JSON (غيّره حسب مكان الملف عندك)
    input_file = "D:\\Graduation Project\\project\\data\\books.json"
    output_file = "D:\\Graduation Project\\project\\data\\preprocessed_books.json"
    
    # تحميل الداتا
    json_data = load_json_data(input_file)
    if json_data:
        # معالجة الداتا
        # تم ضبط max_length إلى 6000 بناءً على أطول نص لديك
        df = preprocess_books_data(json_data, min_length=100, max_length=6000)
        
        # حفظ الداتا
        save_preprocessed_data(df, output_file)
        print(f"Number of books after preprocessing: {len(df)}")
        print("\nSample of preprocessed data:")
        print(df.head())

        # عرض بعض تفاصيل البيانات المعالجة
        print("\nKeywords sample for a book:")
        if not df.empty:
            print(df['keywords'].iloc[0])
        else:
            print("No books processed to show keywords sample.")