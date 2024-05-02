FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

ENV JULIA_VER=1.10.2


RUN apt update -y && apt upgrade -y && \
    apt-get install -y wget build-essential wget unzip checkinstall libncursesw5-dev  libssl-dev  libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev && \
    cd /usr/src && \
    wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz && \
    tar xzf Python-3.10.12.tgz && \
    cd Python-3.10.12 && \
    ./configure --enable-optimizations && \
    make altinstall

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VER}-linux-x86_64.tar.gz
RUN tar -xf julia-${JULIA_VER}-linux-x86_64.tar.gz
RUN rm julia-${JULIA_VER}-linux-x86_64.tar.gz
ENV PATH=$PATH:julia-${JULIA_VER}/bin

RUN julia -e 'using Pkg; Pkg.add("IJulia");'

RUN apt-get update && apt-get install -y netcat
RUN pip3.10 install ipykernel
RUN python3.10 -m ipykernel install --user --name=python3.10.12-kernel


COPY send_gpu_metrics.sh send_gpu_metrics.sh

USER root

CMD ["jupyter", "notebook", "--allow-root", "--ip='*'", "--no-browser", "--notebook-dir=notebooks"]
