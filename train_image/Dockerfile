FROM python:3.11-slim
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
COPY train.py /app/train.py
ENTRYPOINT ["python", "/app/train.py"]

