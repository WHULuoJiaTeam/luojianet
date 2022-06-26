FROM ubuntu:18.04

MAINTAINER MindSpore Authors

# Set env
ENV PYTHON_ROOT_PATH /usr/local/python-3.7.5
ENV PATH ${PYTHON_ROOT_PATH}/bin:/usr/local/bin:/root/.local/bin:$PATH

# Install base tools
RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt install -y \
    vim \
    wget \
    curl \
    xz-utils \
    net-tools \
    openssh-client \
    git \
    ntpdate \
    tzdata \
    tcl \
    sudo \
    bash-completion

# Install compile tools
RUN DEBIAN_FRONTEND=noninteractive apt install -y \
    gcc \
    g++ \
    zlibc \
    make \
    libgmp-dev \
    patch \
    autoconf \
    libtool \
    automake \
    flex \
    libjpeg8-dev

# Install the rest dependent tools
RUN DEBIAN_FRONTEND=noninteractive apt install -y \
    libnuma-dev                

# Set bash
RUN echo "dash dash/sh boolean false" | debconf-set-selections
RUN DEBIAN_FRONTEND=noninteractive dpkg-reconfigure dash

# Install python (v3.7.5)
RUN apt install -y libffi-dev libssl-dev zlib1g-dev libbz2-dev libncurses5-dev \
    libgdbm-dev libgdbm-compat-dev liblzma-dev libreadline-dev libsqlite3-dev \
    && cd /tmp \
    && wget https://github.com/python/cpython/archive/v3.7.5.tar.gz \
    && tar -xvf v3.7.5.tar.gz \
    && cd /tmp/cpython-3.7.5 \
    && mkdir -p ${PYTHON_ROOT_PATH} \
    && ./configure --prefix=${PYTHON_ROOT_PATH} --enable-shared \
    && make -j4 \
    && make install -j4 \
    && rm -f /usr/local/bin/python \
    && rm -f /usr/local/bin/pip \
    && rm -f /usr/local/lib/libpython3.7m.so.1.0 \
    && ln -s ${PYTHON_ROOT_PATH}/bin/python3.7 /usr/local/bin/python \
    && ln -s ${PYTHON_ROOT_PATH}/bin/pip3.7 /usr/local/bin/pip \
    && ln -s ${PYTHON_ROOT_PATH}/lib/libpython3.7m.so.1.0 /usr/local/lib/libpython3.7m.so.1.0 \
    && ldconfig \
    && rm -rf /tmp/cpython-3.7.5 \
    && rm -f /tmp/v3.7.5.tar.gz

# Set pip source
RUN mkdir -pv /root/.pip \
    && echo "[global]" > /root/.pip/pip.conf \
    && echo "trusted-host=mirrors.aliyun.com" >> /root/.pip/pip.conf \
    && echo "index-url=http://mirrors.aliyun.com/pypi/simple/" >> /root/.pip/pip.conf

# Install MindSpore cpu whl package
RUN pip install --no-cache-dir https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.5.0-rc1/MindSpore/cpu/x86_64/mindspore-1.5.0rc1-cp37-cp37m-linux_x86_64.whl

# Install MindInsight cpu whl package
RUN pip install --no-cache-dir https://ms-release.obs.cn-north-4.myhuaweicloud.com/1.5.0-rc1/MindInsight/any/mindinsight-1.5.0rc1-py3-none-any.whl
