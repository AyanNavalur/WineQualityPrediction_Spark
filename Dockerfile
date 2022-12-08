FROM python:3.12.0a1-buster
RUN apt-get update &&\
    /usr/local/bin/python3 -m pip install --upgrade pip &&\
    /usr/local/bin/python3 -m pip install --upgrade setuptools

ENV PATH="/home/myuser/.local/bin:${PATH}."

WORKDIR /home/myuser
RUN pip3 install -r requirements.txt
ENTRYPOINT ["runuser", "-u", "myuser", "--", "python3", "-m", "flask", "run", "--host=0.0.0.0"]