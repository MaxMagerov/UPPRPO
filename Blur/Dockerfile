FROM python:3.12

# Create app directory
WORKDIR /app

# Install app dependencies
COPY ./ ./

RUN apt-get update

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London
RUN apt-get install -y python3-tk portaudio19-dev

RUN pip install -r requirements.txt

RUN apt-get install ffmpeg libsm6 libxext6  -y

CMD [ "python", "Blur.py" ]
