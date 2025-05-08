FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY pdfusion_app.py . 
COPY pdfusion.png . 

RUN python -m pip install --upgrade pip wheel setuptools
COPY requirements.txt .
RUN pip install -r requirements.txt


RUN pip install --no-cache-dir \
    streamlit \
    requests \
    pillow

EXPOSE 8501

CMD ["streamlit", "run", "pdfusion_app.py", "--server.headless", "true"]
