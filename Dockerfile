FROM nvidia/cuda:10.2-base-ubuntu18.04

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# set noninteractive installation
RUN echo export DEBIAN_FRONTEND=noninteractive
#install tzdata package
RUN apt-get update && apt-get install -y \
    tzdata \
 && rm -rf /var/lib/apt/lists/* # (2) switch to (1)
# set your timezone
RUN ln -fs /usr/share/zoneinfo/Asia/Taipei /etc/localtime # (1) switch to (2)
# RUN dpkg-reconfigure -f noninteractive tzdata

RUN apt-get update && apt-get install -qqy \
    x11-apps \
    locales \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    openssh-server \
    vim	\
    ffmpeg \
    xvfb \
    python3-tk \
    python3-pip \
    python-opengl \
 && rm -rf /var/lib/apt/lists/*

ENV NOTVISIBLE "in users profile"
RUN echo "export VISIBLE=now" >> /etc/profile

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade Pillow
RUN pip3 install pipreqs flake8 flake8-unused-arguments numpy torch matplotlib gym==0.17.0

# Locale
RUN locale-gen en_US.UTF-8
RUN locale-gen zh_TW.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

CMD /bin/sh -c 'service ssh restart && bash'
