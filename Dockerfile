# Specifies base image and tag

FROM python:3.9

# Sets up the entry point to invoke the trainer.
ENTRYPOINT ["python", "app.py"]
