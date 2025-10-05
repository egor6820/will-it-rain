FROM python:3.11-slim

# Set workdir
WORKDIR /app

# Install system deps (optional, can extend if you need psycopg2 etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run Uvicorn
CMD ["uvicorn", "api:app", "--host", "::", "--port", "8000"]
