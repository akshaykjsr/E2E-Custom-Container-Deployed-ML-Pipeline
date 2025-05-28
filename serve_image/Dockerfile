FROM python:3.11-slim

# install runtime deps
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# copy your app
WORKDIR /app
COPY score.py .

# Vertex health + predict routes
ENV AIP_HEALTH_ROUTE=/healthz \
    AIP_PREDICT_ROUTE=/predict \
    PORT=8080

# exec gunicorn with 4 workers, 120s timeout, binding $PORT
ENTRYPOINT ["gunicorn","-w","4","--timeout","120","-b","0.0.0.0:8080","score:app"]

