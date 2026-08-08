FROM python:3.11-slim

WORKDIR /app

# Install system dependencies if needed (none are strictly required for our standard flask/numpy app)
# Copy python requirements
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy code components
COPY app.py models.py setup_models.py ./
COPY static/ ./static/
COPY templates/ ./templates/

# Pre-download models inside container to make it self-contained
RUN python setup_models.py

# Expose flask dev port
EXPOSE 5000

ENV FLASK_APP=app.py
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]
