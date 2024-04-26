FROM tensorflow/tensorflow:2.14.0-gpu-jupyter

ENV JULIA_VER=1.10.2

RUN apt-get update && apt-get install wget unzip

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VER}-linux-x86_64.tar.gz
RUN tar -xf julia-${JULIA_VER}-linux-x86_64.tar.gz
RUN rm julia-${JULIA_VER}-linux-x86_64.tar.gz
ENV PATH=$PATH:julia-${JULIA_VER}/bin

RUN julia -e 'using Pkg; Pkg.add("IJulia");'

USER root

CMD ["jupyter", "notebook", "--allow-root", "--ip='*'", "--no-browser", "--notebook-dir=notebooks"]
