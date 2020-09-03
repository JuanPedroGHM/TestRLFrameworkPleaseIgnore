FROM pytorch/pytorch:latest

WORKDIR /code

RUN apt-get update -y
RUN pip install gym \
                matplotlib \
                tensorboard \
		scikit-learn	
