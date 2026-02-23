FROM python:3.13-slim

WORKDIR /app
EXPOSE 8051

RUN pip install --no-cache-dir uv

# Install dependencies from pyproject.toml + uv.lock.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY gui.py brief.py ./

# Cloud Run injects PORT at runtime.
CMD ["sh", "-c", "uv run streamlit run gui.py --server.address=0.0.0.0 --server.port=${PORT:-8051} --server.headless=true"]
