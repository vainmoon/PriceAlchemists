FROM python:3.9

LABEL maintainer="saumnov.ext"

# Set working directory
WORKDIR /backend

# Copy requirements first to leverage Docker cache
COPY ./requirements.txt /backend/requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade -r /backend/requirements.txt

# Copy application code
COPY ./app /backend/app

# Copy model weights and data
COPY ./weights /backend/weights
COPY ./data /backend/data

# Expose port
EXPOSE 8050

# Run the application
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "8050"]