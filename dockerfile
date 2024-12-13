FROM python:3.10

# Use the official TensorFlow image (CPU version)
FROM tensorflow/tensorflow:2.10.0

WORKDIR /app

COPY muse_docker.py .
COPY model/data/model ./model/data/model
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# 0.0.0.0 = localhost
# 8000 = the port of the container. To run this, sudo docker run -p whateverportoflocalhost:8000 <image_name>.
CMD ["uvicorn", "muse_docker:app", "--host", "0.0.0.0", "--port", "8000"]