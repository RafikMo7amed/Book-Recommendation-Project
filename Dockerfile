FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /workspace

ENV HF_HOME /workspace/hf_cache
ENV TRANSFORMERS_CACHE /workspace/hf_cache

COPY ./requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /workspace/requirements.txt

RUN mkdir -p /workspace/hf_cache && chmod -R 777 /workspace/hf_cache

COPY ./app.py /workspace/app.py
COPY ./api /workspace/api
COPY ./data /workspace/data

CMD ["python", "app.py"]