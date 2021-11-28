FROM python:3.8

RUN mkdir /pump_databroker
COPY . /pump_databroker
WORKDIR /pump_databroker

RUN apt-get update -y 
RUN apt-get install -y python3-pip python3-dev 
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

RUN useradd app
USER app
ENTRYPOINT python3 -u pump_databroker.py