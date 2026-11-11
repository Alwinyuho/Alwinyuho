# CPU image for the demo (multilingual-e5-small + BM25, no reranker).
# Port 7860 matches Hugging Face Spaces.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    BHASHARAG_HOME=/app \
    BHASHARAG_PROFILE=cpu \
    BHASHARAG_LANGS=hi,ta

WORKDIR /app
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install ".[ml,ui]"
COPY reports ./reports

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 7860
CMD ["uvicorn", "bhasharag.api:app", "--host", "0.0.0.0", "--port", "7860"]
