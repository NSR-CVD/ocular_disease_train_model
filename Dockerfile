FROM python:3.10.9

WORKDIR /usr/src/app

COPY . .

RUN pwd

RUN sudo pip install --upgrade pip

RUN sudo pip install -r requirement.txt 

COPY . .

# Run the application:
CMD ["python", "train_model.py"]