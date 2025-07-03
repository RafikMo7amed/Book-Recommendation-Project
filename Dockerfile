FROM python:3.11-slim
WORKDIR /app
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy the new top-level app.py
COPY ./app.py /app/app.py

# Copy both the api code and the necessary data
COPY ./api /app/api
COPY ./data /app/data

# The command to run the API server
CMD ["python", "app.py"]