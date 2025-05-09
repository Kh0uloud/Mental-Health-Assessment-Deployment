FROM python:slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --default-timeout=100 --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

ENV MLFLOW_TRACKING_URI=http://mlflow.example.com:8080
ENV MODEL_NAME=Mental_Health_assessment

CMD ["sh", "-c", "python app.py --mlflow_tracking_uri $MLFLOW_TRACKING_URI --model_name $MODEL_NAME"]