FROM python:3.11-slim

# Set environment variables for Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Create working directory
WORKDIR /app

# Install and configure 'uv'
RUN pip install --upgrade uv

# Set up virtual environment with uv
RUN uv venv create /app/env
ENV PATH="/app/env/bin:$PATH"

# Copy only the dependency file first for optimal caching
COPY requirements.txt /app/

# Install dependencies
RUN uv pip install -r requirements.txt

# Explicitly download spaCy model (if applicable)
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . /app

# Expose FastAPI default port
EXPOSE 8080

# Start the FastAPI app using uvicorn (with optimized settings)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]