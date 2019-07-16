FROM ubuntu:latest

RUN apt-get update && apt-get install -y --no-install-recommends \
  locales \
  python3 \
  python3-pip \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists* \
  && locale-gen en_US.UTF-8 \
  && useradd -m user

RUN pip3 install \
  matplotlib \
  numpy \
  scikit-learn 

ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

USER user 

COPY --chown=user:user nslkdd/numpy/ user/datasets
COPY --chown=user:user unswnb15/numpy/ user/datasets
COPY --chown=user:user dgd/numpy/ user/datasets

WORKDIR user/

ENTRYPOINT ["bash"]
