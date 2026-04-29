FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.ai/install.sh | sh

# Copy project
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Create output dirs
RUN mkdir -p reports mlruns

# Default: run full eval suite with llama3.2
CMD ["sh", "-c", "ollama serve & sleep 5 && ollama pull llama3.2 && python main.py --suite all --model llama3.2"]
