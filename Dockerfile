FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

ENV JULIA_VER=1.10.2


RUN apt-get update -y && apt-get upgrade -y && \
    apt-get install -y wget build-essential wget unzip checkinstall libncursesw5-dev  libssl-dev  \
    libsqlite3-dev tk-dev libgdbm-dev libc6-dev libbz2-dev libffi-dev zlib1g-dev netcat && \
    cd /usr/src && \
    wget https://www.python.org/ftp/python/3.10.12/Python-3.10.12.tgz && \
    tar xzf Python-3.10.12.tgz && \
    cd Python-3.10.12 && \
    ./configure --enable-optimizations && \
    make altinstall

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VER}-linux-x86_64.tar.gz && \
    tar -xf julia-${JULIA_VER}-linux-x86_64.tar.gz && \
    rm julia-${JULIA_VER}-linux-x86_64.tar.gz

ENV PATH=$PATH:julia-${JULIA_VER}/bin

RUN julia -e 'using Pkg; Pkg.add("IJulia");'

RUN pip3.10 install ipykernel && \
    python3.10 -m ipykernel install --user --name=python3.10.12-kernel

COPY send_gpu_metrics.sh send_gpu_metrics.sh
COPY requirements_3_11.txt requirements_3_11.txt
COPY requirements_3_10.txt requirements_3_10.txt

RUN apt-get install -y libcairo2-dev pkg-config python3-dev && \
    pip3 install pycairo

RUN pip3 install -r requirements_3_11.txt
RUN pip3.10 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
    pip3.10 install pytorch-ignite
RUN pip3.10 install -r requirements_3_10.txt

COPY Env /tf/Env
RUN julia -e 'using Pkg; Pkg.activate("/tf/Env"); Pkg.instantiate()'

USER root

CMD ["jupyter", "notebook", "--allow-root", "--ip='*'", "--no-browser", "--notebook-dir=notebooks"]
