# CPU image for the recommendation API. Mount the downloaded data at /data
# (amazon/ folder) and the trained artifacts at /app/artifacts.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    SHELFRANK_HOME=/app \
    SHELFRANK_DATA=/data

WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[ui]"

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 7860
CMD ["uvicorn", "shelfrank.api:app", "--host", "0.0.0.0", "--port", "7860"]
