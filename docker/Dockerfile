FROM continuumio/miniconda3
MAINTAINER aluck@cordeo.ru

EXPOSE 8888

RUN apt update && apt -y dist-upgrade
RUN apt install -y unzip

RUN mkdir /d2l-en && cd /d2l-en && curl https://www.d2l.ai/d2l-en-1.0.zip -o d2l-en.zip && unzip d2l-en.zip && rm d2l-en.zip
RUN /opt/conda/bin/conda update -n base -c defaults conda
RUN /opt/conda/bin/conda env update -f /d2l-en/environment.yml
RUN /opt/conda/bin/conda install jupyter -y --quiet

ADD --chown=0 run_conda.sh /root/

CMD /root/run_conda.sh
