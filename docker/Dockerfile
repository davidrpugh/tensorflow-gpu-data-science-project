FROM nvidia/cuda:10.1-devel-ubuntu16.04

LABEL maintainer="pughdr <david.pugh@kaust.edu.sa>"

SHELL [ "/bin/bash", "-c" ]

RUN apt-get update --fix-missing && \
    apt-get install -y wget bzip2 curl git gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG username=al-khawarizmi
ARG uid=1000
ARG gid=100
ENV USER $username
ENV UID $uid
ENV GID $gid
ENV HOME /home/$USER

RUN adduser --disabled-password \
    --gecos "Non-root user" \
    --uid $UID \
    --gid $GID \
    --home $HOME \
    $USER

USER $USER

RUN mkdir $HOME/app
WORKDIR $HOME/app

COPY --chown=$UID:$GID environment.yml requirements.txt postBuild ./
COPY --chown=$UID:$GID ./bin/create-conda-environment.sh ./bin/
COPY --chown=$UID:$GID ./docker/entrypoint.sh ./docker/
RUN chmod u+x ./postBuild ./bin/create-conda-environment.sh ./docker/entrypoint.sh

ENV MINICONDA_VERSION 4.7.12
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-$MINICONDA_VERSION-Linux-x86_64.sh -O ~/miniconda.sh && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b && \
    rm ~/miniconda.sh && \
    echo ". $HOME/miniconda3/etc/profile.d/conda.sh" >> ~/.profile

RUN ~/miniconda3/bin/conda update --name base --channel defaults conda && \
    ~/miniconda3/bin/conda init bash
RUN ./bin/create-conda-environment.sh
RUN ~/miniconda3/bin/conda clean --all --yes

# use an entrypoint script to insure conda environment is properly activated at runtime
ENTRYPOINT [ "./docker/entrypoint.sh" ]

# default command will be to launch JupyterLab server for development
CMD [ "jupyter", "lab", "--no-browser", "--ip", "0.0.0.0" ]
