FROM python:3.11-slim

WORKDIR /DoubleParkingViolation

RUN apt-get update && apt-get install -y libgl1

COPY requirements.txt /DoubleParkingViolation

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install "uvicorn[standard]"

COPY . /DoubleParkingViolation

EXPOSE 8080

# ENV GOOGLE_APPLICATION_CREDENTIALS="capstone-t5-4494681f8d0c.json"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port","8080"]
