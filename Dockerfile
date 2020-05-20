FROM nvidia/cuda

ENV DEBIAN_FRONTEND=noninteractive 

RUN apt-get update && apt-get install -y --no-install-recommends \
  locales \
  python-dev \
  python3 \
  python3-dev \
  python3-pip \
  python3-setuptools \
  python3-wheel \
  texlive-full \
  && locale-gen en_US.UTF-8 \
  && useradd -m user

RUN pip3 install --upgrade pip \
  && pip3 install --upgrade \
  matplotlib \
  numpy \
  scikit-learn \
  tensorflow \
  torch \
  torchvision

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

USER user 

COPY --chown=user:user dgd/numpy/ user/datasets/dgd/numpy/
COPY --chown=user:user nslkdd/numpy/ user/datasets/nslkdd/numpy/
COPY --chown=user:user phishing/numpy/ user/datasets/phishing/numpy/
COPY --chown=user:user unswnb15/numpy/ user/datasets/unswnb15/numpy/

WORKDIR user/

ENTRYPOINT ["bash"]
