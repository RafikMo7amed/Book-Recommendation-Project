FROM python:3.11-slim
WORKDIR /app

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# إنشاء مجلد الكاش مع إعطائه الصلاحيات اللازمة
RUN mkdir -p /app/cache && chmod -R 777 /app/cache

# نسخ ملفات التطبيق
COPY ./app.py /app/app.py
COPY ./api /app/api
COPY ./data /app/data

# تشغيل التطبيق
CMD ["python", "app.py"]