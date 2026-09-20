FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY glin/ glin/

RUN pip install --no-cache-dir .

EXPOSE 8000

ENTRYPOINT ["glin"]
CMD ["serve", "--mode", "http", "--host", "0.0.0.0", "--port", "8000"]
