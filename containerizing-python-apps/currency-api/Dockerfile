# Use official Python image
FROM python:3.11-slim
# Set work directory
WORKDIR /app
# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Copy app code
COPY . .
# Expose port
EXPOSE 8000
# Run the app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
