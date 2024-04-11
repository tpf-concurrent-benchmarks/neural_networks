FROM jupyter/base-notebook:x86_64-python-3.11.6

ENV JULIA_VER=1.10.2

RUN wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VER}-linux-x86_64.tar.gz
RUN tar -xf julia-${JULIA_VER}-linux-x86_64.tar.gz
RUN rm julia-${JULIA_VER}-linux-x86_64.tar.gz
ENV PATH=$PATH:julia-${JULIA_VER}/bin

RUN julia -e 'using Pkg; Pkg.add("IJulia");'

USER root
RUN apt-get update && apt-get install -y unzip

CMD ["jupyter", "notebook", "--allow-root", "--ip='*'", "--no-browser", "--notebook-dir=notebooks"]
