FROM python:3.11-slim

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY langscout ./langscout
COPY models ./models

EXPOSE 8000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/health')" || exit 1
CMD ["uvicorn", "langscout.api:app", "--host", "0.0.0.0", "--port", "8000"]
