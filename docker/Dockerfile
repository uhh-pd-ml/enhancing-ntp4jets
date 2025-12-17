FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN apt-get update && apt-get install -y \
    vim \
    neovim \
    wget \
    curl \
    git \
    zsh \
    libxcb-xinerama0 \
    lsb-release \
    bat \
    ripgrep

RUN curl -sL https://raw.githubusercontent.com/wimpysworld/deb-get/main/deb-get | bash -s install deb-get
RUN deb-get install \
    git-delta

# Install GitHub cli
RUN mkdir -p -m 755 /etc/apt/keyrings && wget -qO- https://cli.github.com/packages/githubcli-archive-keyring.gpg | tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
    && chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
    && apt update \
    && apt install gh -y

# allow pip install of "sklearn", which should be replaced by "scikit-learn"
# (if this is not used, the github CI pipeline fails from time to time since
# some packages still have "sklearn" in their dependencies)
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

RUN conda init bash

COPY requirements.txt .
RUN pip install -r requirements.txt

RUN conda install -y jupyter
RUN git clone https://github.com/minyoungg/vqtorch && cd vqtorch && pip install -e .

SHELL ["/bin/bash", "--login", "-c"]
