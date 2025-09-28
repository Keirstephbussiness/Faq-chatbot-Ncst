# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for blis
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libblas-dev \
    liblapack-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Set environment variables
ENV KNOWLEDGE_BASE_DIR=knowledge
ENV PORT=5000

# Expose port
EXPOSE 5000

# Run the app
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "app:app"]
