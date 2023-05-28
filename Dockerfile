# Specifies base image and tag
FROM python:3.7
WORKDIR /

# Installs additional packages
#RUN pip install pkg1 pkg2 pkg3

# Downloads training data
#RUN curl https://example-url/path-to-data/data-filename --output /root/data-filename

# Copies the trainer code to the docker image.
#COPY your-path-to/model.py /root/model.py
#COPY your-path-to/task.py /root/task.py

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "app.py"]