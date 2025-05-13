FROM python:3.9-slim

WORKDIR /app

COPY model/requirements.txt ./requirements.txt

RUN apt-get update && apt-get install -y git \
    && pip install --upgrade pip \
    && pip install --default-timeout=100 --no-cache-dir -r requirements.txt

COPY . .

ENV DATA_PATH=./data/structured_data.csv
ENV MODEL_SAVE_PATH=./model/saved_models/model.pkl

CMD ["python", "model/model.py"]