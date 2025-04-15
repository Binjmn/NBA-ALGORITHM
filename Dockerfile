# NBA Betting Prediction System Dockerfile
# This Dockerfile builds the core prediction system container

FROM python:3.13.1-slim

# Set timezone to EST as per project requirements
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Set working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories with appropriate permissions
RUN mkdir -p /app/data /app/logs /app/config /app/models \
    && chmod -R 777 /app/data /app/logs /app/config /app/models

# Set environment variable to indicate we're running in Docker
ENV RUNTIME_ENVIRONMENT=docker

# Create a non-root user to run the application
RUN useradd -m nbauser
USER nbauser

# Command to run when the container starts
# This will be the entry point that runs the scheduler
CMD ["python", "-m", "src.run_scheduler"]
