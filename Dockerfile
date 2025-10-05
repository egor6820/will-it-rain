FROM python:3.11-slim

WORKDIR /app

# системні залежності для folium/pydeck (мінімум)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Port: використовуємо змінну PORT якщо є, інакше 8501
ENV PORT=8501

EXPOSE 8501

CMD ["bash", "-lc", "streamlit run app.py --server.port=${PORT} --server.address=[::] --server.enableCORS=false"]
