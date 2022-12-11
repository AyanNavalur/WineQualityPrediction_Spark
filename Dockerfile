FROM openjdk:8

# RUN apt-get update && apt-get install -y --no-install-recommends \
#     python3.5 \
#     python3-pip \
#     && \
#     apt-get clean && \
#     rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y python3 python3-pip

RUN python3 --version

RUN apt-get update &&\
    python3 -m pip install --upgrade pip &&\
    python3 -m pip install --upgrade setuptools &&\
    adduser myuser

ENV PATH="/home/myuser/.local/bin:${PATH}"
# ENV PYSPARK_SUBMIT_ARGS="--master local[2] pyspark-shell"
WORKDIR /home/myuser
COPY --chown=myuser:myuser . /home/myuser
RUN pip3 install -r requirements.txt

# For getting hadoop binaries
RUN apt install -y git
RUN git clone https://github.com/cdarlint/winutils.git
ENV HADOOP_HOME=/home/myuser/winutils/hadoop-3.1.1
ENV PATH=$PATH:$HADOOP_HOME/bin

CMD ["python3", "trainingScript.py"]