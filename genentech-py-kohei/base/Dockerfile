FROM ubuntu:wily
MAINTAINER Kohei Ozaki <i@ho.lc>

ADD sources.list /etc/apt/sources.list
ADD install.sh /root/install.sh
ADD zshrc /root/.zshrc

RUN bash /root/install.sh

ENV PATH $PATH:/root/anaconda2/bin
WORKDIR /opt
