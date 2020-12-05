FROM pytorch/pytorch:latest
RUN apt-get update -y

WORKDIR /code

COPY trlfpi/ ./trlfpi
COPY environment.yml .
COPY setup.py .

RUN conda env create -f environment.yml
SHELL ["conda", "run", "-n", "TRLFPI", "/bin/bash", "-c"]
RUN pip install -e .

ENTRYPOINT ["conda", "run", "-n", "TRLFPI", "trlfpi"]
