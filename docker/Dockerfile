FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .

RUN apt-get update -qq && apt-get install ffmpeg unzip -y

RUN pip install --no-cache-dir -r requirements.txt

CMD ["/bin/bash"]
