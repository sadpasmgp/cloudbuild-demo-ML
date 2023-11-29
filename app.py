###
Cloudbuild-demo:

Submitting a Vertex AI training custom Job via github trigger


The steps performed include:

Formalize model experiment in a script
Run model traning using local script on Vertex AI Training
Check out ML experiment parameters and metrics in Vertex AI Experiments


Dataset :
The Glass Identification dataset is a dataset from USA Forensic Science Service with 6 types of glass defined in terms of their oxide content (i.e. Na, Fe, K, etc). The goal is classifying the types of glass based on oxide features.
###


# 1. Setting up env
PROJECT_ID = "sandbox-dev-dbg"  # @param {type:"string"}

# Set the project id
! gcloud config set project {PROJECT_ID}

REGION = "us-central1"  # @param {type: "string"}

import random
import string
import os

# TO Avoid duplication of naming convention, add a randomized 8 characters
# Generate a uuid of length 8
def generate_uuid():
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=8))


UUID = generate_uuid()


#Create CLoud storage bucket
BUCKET_URI = f"gs://your-bucket-name-{PROJECT_ID}-unique"  # @param {type:"string"}

! gsutil mb -l $REGION -p $PROJECT_ID $BUCKET_URI

#Set relevant service account
SERVICE_ACCOUNT = "[your-service-account]"  # @param {type:"string"}

IS_COLAB = False

if (
    SERVICE_ACCOUNT == ""
    or SERVICE_ACCOUNT is None
    or SERVICE_ACCOUNT == "[your-service-account]"
):
    # Get your service account from gcloud
    if not IS_COLAB:
        shell_output = !gcloud auth list 2>/dev/null
        SERVICE_ACCOUNT = shell_output[2].replace("*", "").strip()

    if IS_COLAB:
        shell_output = ! gcloud projects describe  $PROJECT_ID
        project_number = shell_output[-1].split(":")[1].strip().replace("'", "")
        SERVICE_ACCOUNT = f"{project_number}-compute@developer.gserviceaccount.com"

    print("Service Account:", SERVICE_ACCOUNT)

"""
Set service account access for Vertex AI Training :
Run the following commands to grant your service account access to read and update metadata in Vertex AI ML Metadata while the custom training job is running
"""

! gsutil iam ch serviceAccount:{SERVICE_ACCOUNT}:roles/storage.objectCreator $BUCKET_URI

! gsutil iam ch serviceAccount:{SERVICE_ACCOUNT}:roles/storage.objectViewer $BUCKET_URI

#Set folder
import os

TUTORIAL_DIR = os.path.join(
    os.getcwd(), "custom_training_autologging_local_script_tutorial"
)
os.makedirs(TUTORIAL_DIR, exist_ok=True)

#Get Data
SOURCE_DATA_URL = "gs://cloud-samples-data/vertex-ai/dataset-management/datasets/uci_glass_preprocessed/glass.csv"
DESTINATION_DATA_URL = f"{BUCKET_URI}/data/glass.csv"

! gsutil cp $SOURCE_DATA_URL $DESTINATION_DATA_URL

#Import libraries
import os

from google.cloud import aiplatform as vertex_ai

# Define Constants

# Training
EXPERIMENT_NAME = f"glass-classification-{UUID}"
TRAIN_SCRIPT_PATH = os.path.join(TUTORIAL_DIR, "task.py")
JOB_DISPLAY_NAME = f"sklearn-autologged-custom-job-{UUID}"
PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI = (
    f"{REGION.split('-')[0]}-docker.pkg.dev/vertex-ai/training/tf-cpu.2-11:latest"
)
MODEL_FILE_URI = f"{BUCKET_URI}/models/model.joblib"
DESTINATION_DATA_PATH = DESTINATION_DATA_URL.replace("gs://", "/gcs/")
MODEL_FILE_PATH = MODEL_FILE_URI.replace("gs://", "/gcs/")
REPLICA_COUNT = 1
TRAIN_MACHINE_TYPE = "n1-standard-4"
TRAINING_JOBS_URI = f"{BUCKET_URI}/jobs"

# Initialize the Vertex AI SDK for Python for your project.
vertex_ai.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET_URI)

vertex_ai.init(
    project=PROJECT_ID,
    location=REGION,
    staging_bucket=BUCKET_URI,
    experiment=EXPERIMENT_NAME,
)

# Train a scikit-learn model with a prebuilt container
# Then, you train a custom model using a prebuilt container for scikit-learn models

task_script = f"""
#!/usr/bin/env python3

'''
A simple module to train a classifier on the glass dataset.
'''

# Libraries
import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib


# Variables
DATA_PATH = '{DESTINATION_DATA_PATH}'
MODEL_PATH = '{MODEL_FILE_PATH}'
TEST_SIZE = 0.2
SEED = 8

# Helpers
def read_data(path):
    df = pd.read_csv(path)
    return df


def split_data(df):
    y = df.pop('glass_type')
    X = df
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=SEED)
    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=5)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy


def save_model(model, path):
    p = Path(path)
    if not p.parent.exists():
      p.parent.mkdir(parents=True)
    joblib.dump(model, path)


def main():

    # Read data
    df = read_data(DATA_PATH)

    # Split data
    X_train, X_test, y_train, y_test = split_data(df)

    # Train model
    model = train_model(X_train, y_train)

    # Evaluate model
    accuracy = evaluate_model(model, X_test, y_test)
    print('Model accuracy:', accuracy)

    # Save model
    save_model(model, MODEL_PATH)



if __name__ == '__main__':

    # Run main
    main()
"""

with open(TRAIN_SCRIPT_PATH, "w") as train_file:
    train_file.write(task_script)
train_file.close()

"""
Define custom training job:

Define a custom job with the prebuilt container image for training code packaged as Python script. In this case, you set enable_autolog=True to automatically track parameters and metrics after the training job completes.

"""
job = vertex_ai.CustomJob.from_local_script(
    project=PROJECT_ID,
    staging_bucket=TRAINING_JOBS_URI,
    display_name=JOB_DISPLAY_NAME,
    script_path=TRAIN_SCRIPT_PATH,
    container_uri=PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI,
    requirements=["pandas", "scikit-learn"],
    replica_count=REPLICA_COUNT,
    machine_type=TRAIN_MACHINE_TYPE,
    enable_autolog=True,
)

"""
Run custom training job

Next, you run the training job using the method run.
"""

job.run(experiment=EXPERIMENT_NAME, service_account=SERVICE_ACCOUNT)

"""
Get your autologged experiment:
After you train your model, you can get parameters and metrics of the autologged experiment.
"""

experiment_df = vertex_ai.get_experiment_df(experiment=EXPERIMENT_NAME)
experiment_df.T

experiment_run = experiment_df.run_name.iloc[0]

with vertex_ai.start_run(experiment_run, resume=True) as run:
    # get the latest logged custom job
    logged_job = run.get_logged_custom_jobs()[-1]

print(logged_job.job_spec)
