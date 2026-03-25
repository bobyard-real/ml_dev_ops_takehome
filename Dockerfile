FROM python:3.12-slim

WORKDIR /app

# Install dependencies first (maximizes layer cache reuse)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download model assets at build time so startup is instant
RUN python -c "from inference.assets import ensure_model_assets; ensure_model_assets()"

EXPOSE 8000

CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
