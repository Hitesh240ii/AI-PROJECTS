FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY requirements.txt .
# CPU-only torch keeps the image far smaller than the default CUDA build
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements.txt
# Bake the embedding model into the image so first start needs no download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app/main.py", "--server.address=0.0.0.0", "--server.port=8501"]
