# Wikipedia Infobox API only. Runs uvicorn; no frontend.
FROM python:3.12-slim

WORKDIR /app

COPY docker/requirements-api.txt requirements-api.txt
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
