FROM python:3.11-slim

# -----------------------------
# Set working directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Copy files
# -----------------------------
COPY requirements.txt ./requirements.txt
COPY . ./

# -----------------------------
# Install dependencies
# -----------------------------
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Expose Streamlit port
# -----------------------------
EXPOSE 8501

# -----------------------------
# Environment variables
# -----------------------------
ENV PYTHONUNBUFFERED=1

# Streamlit config
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false
ENV STREAMLIT_FILE_WATCHER_TYPE=none

# -----------------------------
# Run Streamlit app
# -----------------------------
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
