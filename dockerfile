# Use Python 3.11 slim for a smaller image size
FROM python:3.11-slim-bookworm

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_SERVER_HEADLESS=true

# Set the working directory
WORKDIR /app

# Install system dependencies required for RDKit and other libraries
# Added libexpat1 for RDKit dependency
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    libxrender1 \
    libxext6 \
    libsm6 \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# ---- Dependency layer (cached if requirements.txt unchanged) ----
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---- Source code layer ----
COPY . .

# Expose Streamlit default port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
