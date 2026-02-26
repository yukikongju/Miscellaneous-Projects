FROM python:3.13-slim

# ENV PYTHONDONTWRITEBYTECODE=1 \
#     PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install --no-cache-dir uv
# RUN pip install streamlit==1.41.1

COPY pyproject.toml uv.lock ./
# RUN uv sync --frozen --no-dev
RUN uv sync

# COPY src ./src
COPY ./src .

# CMD ["uv", "run", "python", "src/utils.py"]
# CMD ["uv", "run", "streamlit", "run", "src/Pages.py"]

EXPOSE 8501

CMD ["sh", "-c", "uv run streamlit run Pages.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]

# ENTRYPOINT ["uv", "run", "streamlit", "run", "Pages.py", "--server.port=${PORT:-8501}", "--server.address=0.0.0.0"]
# ENTRYPOINT ["streamlit", "run", "Pages.py", "--server.port=8501", "--server.address=0.0.0.0"]

# CMD ["sh", "-c", "uv run streamlit run gui.py --server.address=0.0.0.0 --server.port=${PORT:-8501} --server.headless=true"]
