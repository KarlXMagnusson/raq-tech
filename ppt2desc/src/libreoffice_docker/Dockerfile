FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libreoffice \
    fonts-dejavu \
    fonts-liberation \
    fonts-noto \
    fonts-noto-color-emoji \
    curl \
    fontconfig \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN fc-cache -f -v

RUN useradd --create-home libreoffice

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN chown -R libreoffice:libreoffice /app

USER libreoffice

EXPOSE 2002

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "2002"]
