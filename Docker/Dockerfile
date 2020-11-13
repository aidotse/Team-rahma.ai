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

# CellProfiler >>>
RUN apt update
RUN apt -y upgrade
RUN apt install -y make gcc build-essential libgtk-3-dev
RUN apt-get install -y \
    python3.8-dev \
    python3.8-venv \
    python3-pip \
    openjdk-11-jdk-headless \
    default-libmysqlclient-dev \
    libnotify-dev \
    libsdl2-dev \
    libwebkit2gtk-4.0-dev

ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH=$PATH:/home/ubuntu/.local/bin

RUN python3.8 -m pip uninstall enum34
RUN python3.8 -m pip install wheel
RUN python3.8 -m  pip install \
  -f https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-18.04  \
  wxPython==4.1.0
COPY /CellProfiler /tmp/
RUN python3.8 -m pip install /tmp/CellProfiler/.
# <<< CellProfiler 

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
