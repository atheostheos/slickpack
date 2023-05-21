FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y ffmpeg libsm6 libxext6

COPY requirements.txt .
RUN pip install -r requirements.txt
