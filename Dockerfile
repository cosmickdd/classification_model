
# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project files (including model file)
COPY . .

# If your model file is not in the repo, add a COPY or download step here
# COPY mangrove_mobilenetv2.pth ./

# Expose the port Render expects
EXPOSE 10000

# Start the app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
