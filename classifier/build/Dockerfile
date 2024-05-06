FROM nvcr.io/nvidia/pytorch:24.01-py3

RUN apt update && apt install -y zip htop screen libgl1-mesa-glx

COPY requirements.txt .
RUN python -m pip install --upgrade pip
RUN pip install --no-cache -r requirements.txt

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH

CMD ["python", "-m", "celery", "-A", "worker", "worker", "-l", "info", "-Q", "classify_animals", "-E", "--pool=solo"]
