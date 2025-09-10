FROM python:3.13-slim

RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /dio-ai-business-card

COPY requirements.txt .

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

COPY . .

RUN mkdir .tmp

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]