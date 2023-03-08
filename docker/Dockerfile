# Get the Fast API image with Python version 3.7
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

# Create the directory for the container
WORKDIR /app
COPY requirements.txt ./requirements.txt
COPY ./src/api/main.py ./src/api/main.py
COPY ./src/models ./src/models
COPY ./setup.py ./setup.py

# Install the dependencies
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the serialized model and the vectors
COPY ./models/goal_regressor.joblib ./models/goal_regressor.joblib
COPY ./models/poisson.joblib ./models/poisson.joblib

# Run by specifying the host and port
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "80"]