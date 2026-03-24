FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

RUN pip install --no-cache-dir pandas pyarrow scikit-learn==1.5.0

COPY ["starter.py", "./"]

ENTRYPOINT ["python", "starter.py"]
