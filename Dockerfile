FROM python:3.11-slim

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir ".[mlops]"

CMD ["uvicorn", "tscv_vision.mlops:create_feature_service", "--factory", "--host", "0.0.0.0", "--port", "8000"]
