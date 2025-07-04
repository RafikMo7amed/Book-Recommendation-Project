FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
WORKDIR /workspace

ENV HF_HOME /workspace/hf_cache
ENV TRANSFORMERS_CACHE /workspace/hf_cache

COPY ./requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /workspace/requirements.txt

RUN mkdir -p /workspace/hf_cache && chmod -R 777 /workspace/hf_cache

COPY ./api /workspace/api
COPY ./data /workspace/data

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "7860"]