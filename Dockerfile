FROM ubuntu:latest

RUN apt-get update -y
RUN apt-get install python3 python3-pip libgl1-mesa-glx libglib2.0-0 -y
COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt

ADD app /app

ENTRYPOINT ["python3", "/app/delugeqr.py"]
