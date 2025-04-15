# NBA Betting Prediction System Dockerfile
# This Dockerfile builds the core prediction system container with auto-training capabilities

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
    postgresql-client \
    cron \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories with appropriate permissions
RUN mkdir -p /app/data /app/logs /app/config /app/models /app/data/model_weights \
    && chmod -R 777 /app/data /app/logs /app/config /app/models

# Set default environment variables that can be overridden at runtime
ENV POSTGRES_HOST=postgres \
    POSTGRES_PORT=5432 \
    POSTGRES_DB=nba_prediction \
    POSTGRES_USER=postgres \
    POSTGRES_PASSWORD=postgres \
    RUNTIME_ENVIRONMENT=docker \
    ODDS_API_KEY="" \
    BALLDONTLIE_API_KEY=""

# Add cron job for auto-training and performance tracking
COPY docker/crontab /etc/cron.d/nba-prediction-cron
RUN chmod 0644 /etc/cron.d/nba-prediction-cron \
    && crontab /etc/cron.d/nba-prediction-cron

# Copy startup script
COPY docker/startup.sh /app/docker/startup.sh
RUN chmod +x /app/docker/startup.sh

# For health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
  CMD curl -f http://localhost:5000/api/health || exit 1

# Create a non-root user to run the application
RUN useradd -m nbauser
USER nbauser

# Command to run when the container starts
# This will be the entry point that runs the scheduler
CMD ["/app/docker/startup.sh"]
