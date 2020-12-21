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
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

# Clone git repo
RUN mkdir /proj && \
    cd /proj && \
    git clone git@github.com:lukeross3/graph_rl.git