FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy dependency configuration
COPY pyproject.toml .

# Install uv and project dependencies globally in the container
RUN pip install --no-cache-dir uv && uv pip install --system -r pyproject.toml

# Copy the rest of the application code
COPY . .

# Expose the default Streamlit port
EXPOSE 8501

# Command to download the model, ingest the documentation, and launch Streamlit
CMD ["sh", "-c", "python download_model.py && python ingest.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]
