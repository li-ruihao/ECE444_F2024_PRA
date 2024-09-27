FROM ubuntu:latest
LABEL authors="Ruihao"
RUN apt-get update -y
RUN apt-get install -y python3-pip python3-venv python3-dev build-essential
RUN apt-get clean && rm -rf /var/lib/apt/lists/*
RUN python3 -m venv venv
ENV PATH="/venv/bin:$PATH"
WORKDIR /PRA2_web_app
COPY . /PRA2_web_app
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["hello.py"]