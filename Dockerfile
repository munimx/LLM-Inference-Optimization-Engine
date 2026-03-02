FROM python:3.13-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src/ src/

RUN pip install --no-cache-dir .

# --- Runtime ---
FROM python:3.13-slim

WORKDIR /app
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY configs/ configs/

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "llm_inference_engine.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
