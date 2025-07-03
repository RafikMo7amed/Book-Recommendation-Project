FROM python:3.11-slim
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy both the api code and the necessary data
COPY ./api /app/api
COPY ./data /app/data

# Set the working directory to where the app lives
WORKDIR /app/api

# The command to run the API server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]