FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding and cross-encoder weights into the Docker image during build
# so the live web app starts in 2 seconds!
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-base-en-v1.5'); \
    CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

COPY . .

# Create writable directories for live PDF uploads in the container
RUN mkdir -p pdfs parsed_docs extracted_figures chroma_db && \
    chmod -R 777 /app

ENV TRANSFORMERS_NO_TORCHVISION=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1
EXPOSE 7860

CMD ["streamlit", "run", "rag_app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.fileWatcherType=none"]