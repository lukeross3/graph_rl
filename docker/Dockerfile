# Base on python3.6 slim image
FROM python:3.6-slim

# Add linux packages
RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   vim \
                   curl \
                   ca-certificates \
                   mpich \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

# Install python requirements
ADD requirements.txt /requirements.txt
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r /requirements.txt

# Clone git repo
RUN mkdir /proj && \
    cd /proj && \
    mkdir ~/.ssh/ && \
    git clone https://github.com/lukeross3/graph_rl.git

# Set the working directory to the graph_rl project
WORKDIR /proj/graph_rl/

# Run bash so `docker run -it` opens to bash instead of python cli
CMD ["/bin/bash"]