FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-21-openjdk-arm64

RUN pip install --no-cache-dir \
    pyspark==4.0.0 \
    jupyter \
    notebook

WORKDIR /cours-spark

EXPOSE 8888 4040
