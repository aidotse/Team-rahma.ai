FROM fastdotai/fastai:2.1.3


ENV TZ=Europe/Kiev
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN set -x && \
    apt update && \
    apt install -y --no-install-recommends \
        git\
        wget\
        build-essential \
        apt-utils \
        ca-certificates \
        curl \
        software-properties-common \
        libopencv-dev \ 
        python3-dev \
        python3-pip \ 
        python3-setuptools \
        cmake \
        swig \
        wget \
        unzip \
        screen \
        nano

COPY requirements.txt /tmp/ 
RUN pip3 install pip --upgrade
RUN pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip3 install -r /tmp/requirements.txt
RUN apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /main/

# publish port
EXPOSE 2020

# Example Entry point
#ENTRYPOINT ["/bin/bash","-c", "/usr/bin/python3 -m server.py"]
CMD [./scripts/start_notebook_in_docker.sh]
