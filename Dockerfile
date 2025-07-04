FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
ENV HF_HOME /app/hf_cache
ENV TRANSFORMERS_CACHE /app/hf_cache

COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

RUN mkdir -p /app/hf_cache && chmod -R 777 /app/hf_cache

COPY ./app.py /app/app.py
COPY ./api /app/api
COPY ./data /app/data

CMD ["python", "app.py"]