# NBA Prediction System UI Dockerfile
# This builds a simple web interface for non-technical users

FROM python:3.13.1-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy UI code
COPY . .

# Create necessary directories
RUN mkdir -p /app/static /app/templates /app/logs

# Set environment variables
ENV FLASK_APP=app.py
ENV FLASK_ENV=production
ENV TZ=America/New_York

# Expose the web port
EXPOSE 80

# Run as non-root user
RUN useradd -m nbauser
USER nbauser

# Command to run when container starts
CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:app"]
