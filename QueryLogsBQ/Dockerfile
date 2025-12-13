FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY src/ .

# expose port for flask
EXPOSE 5000
CMD ["python", "main.py"]
