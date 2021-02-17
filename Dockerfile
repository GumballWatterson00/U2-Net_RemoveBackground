FROM python:3.8.5-slim

COPY . /deploy/
# COPY ./requirements.txt /deploy/
# COPY ./app.py /deploy/
# COPY ./u2net.pth/ /deploy/
# COPY ./__init__.py /deploy
# COPY ./detect.py/ /deploy/
# COPY libs/ /deploy/
# COPY ./predicted /deploy/
# COPY ./templates /deploy/
# COPY ./u2net /deploy/
# COPY ./upload /deploy
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
WORKDIR /deploy/
RUN pip3 install -r requirements.txt
EXPOSE 80
ENTRYPOINT [ "python", "app.py" ]