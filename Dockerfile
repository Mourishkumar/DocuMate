# Use a slim Python base
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# system deps for some python packages (easyocr, opencv, etc.)
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Copy project
COPY . /app

# Use pip cache optimization
RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (uvicorn will use $PORT in render)
EXPOSE 9082

# Default command
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "9082"]
