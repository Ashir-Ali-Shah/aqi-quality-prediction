# Use python slim image to keep size down
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies (needed for numpy/scipy sometimes)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose the port FastAPI runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "backend:app", "--host", "0.0.0.0", "--port", "8000"]