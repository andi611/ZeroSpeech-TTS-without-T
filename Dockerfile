FROM python:3.6

RUN apt -y update && apt -y upgrade && apt -y install python3-pip python3-dev git wget

ENV CODE_DIR /root/code

COPY . $CODE_DIR/ZeroSpeech
COPY ~/english $CODE_DIR/ZeroSpeech
WORKDIR $CODE_DIR/ZeroSpeech

# Clean up pycache and pyc files
RUN rm -rf __pycache__ && \
    find . -name "*.pyc" -delete && \
    python3 -m pip install --upgrade setuptools && \
    python3 -m pip install numpy scipy torch torchvision librosa soundfile h5py tqdm tensorboardX

