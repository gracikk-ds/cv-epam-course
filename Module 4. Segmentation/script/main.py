import os
import click
from pathlib import Path
from datetime import datetime
from google.cloud import storage
from google.cloud import aiplatform
from distutils.core import run_setup


PROJECT_ID = "or2-msq-mu-data-df3r5"
BUCKET_NAME = "my_backet"


SERVICE_ACCOUNT = "myvertex@or2-msq-mu-data-df3r5.iam.gserviceaccount.com"
TB_RESOURCE_NAME = "projects/or2-msq-mu-data-df3r5/locations/us-central1/tensorboards/3057677457645767296"
os.environ["GCLOUD_PROJECT"] = PROJECT_ID


@click.group()
def main():
    """
    Entry point for Vertex AI related CLI
    """
    pass


@main.command()
@click.option(
    "--dataset_folder",
    default="gs://my_backet/Barcodes/dataset",
    help="GCS path to dataset.",
)
@click.option("--max_epochs", default=1)
@click.option(
    "--model_description", default="Barcodes Segmentation model", required=True
)
@click.option(
    "--experiment_name",
    default="barcodes segmentation experiment",
    help="A name of the experiment.",
)
@click.option(
    "--model_type",
    default="FPN",
    help="Type of the decoder model",
)
def train_barcode_segmentation(
    dataset_folder, max_epochs, model_description, experiment_name, model_type
):
    # >>
    # Zero step: define variable names
    # <<
    # setup variables
    print("start setup")
    APP_NAME = "barcode_segmentation"
    python_package_gcs_uri = f"{APP_NAME}/train_barcode_segmentation_package"
    PYTHON_PACKAGE_APPLICATION_DIR = "jobs/vertex_train_barcode_segmentation"

    # job variables
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    JOB_NAME = f"{APP_NAME}_train_barcode_segnet_{TIMESTAMP}"
    PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI = (
        "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-7:latest"
    )
    python_module_name = "trainer.task"

    # >>
    # The first step: initialize the Vertex SDK for Python
    # <<
    print("start Vertex SDK")
    aiplatform.init(
        project=PROJECT_ID, location="us-central1", staging_bucket=BUCKET_NAME
    )
    print("done!")

    # >>
    # The second step: create setup archive of jobs/vertex_train_segmentation dir
    # and upload it to the google storage
    # <<
    # Change dir from ./scripts to ./jobs/vertex_train_barcodes folder
    os.chdir(Path(__file__).parent.resolve() / ".." / PYTHON_PACKAGE_APPLICATION_DIR)
    print(f"Creating distribution of {PYTHON_PACKAGE_APPLICATION_DIR}")

    # checking whether setup archive already exist or not
    for arch in Path("dist").glob("*.gz"):
        # if yes -> remove it
        arch.unlink()

    # create new setup archive
    run_setup("setup.py", script_args=["sdist"])

    # create absolute path to setup archive
    dist = list(Path("dist").glob("*.gz"))[0].absolute()
    print(str(dist))

    # Instantiates a client
    storage_client = storage.Client()
    # instantiates a bucket object owned by this client.
    bucket = storage_client.bucket(BUCKET_NAME)
    # upload setup archive to google storage
    blob = bucket.blob("/".join([python_package_gcs_uri, dist.name]))
    blob.upload_from_filename(dist)
    print(
        f"Distribution of {PYTHON_PACKAGE_APPLICATION_DIR} is uploaded to {BUCKET_NAME}"
    )

    # >>
    # The third step: configure the Custom Job resource
    # <<
    # configure the Custom Job resource
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=f"{JOB_NAME}",
        model_description=model_description,
        python_package_gcs_uri=f"gs://{BUCKET_NAME}/{python_package_gcs_uri}/{dist.name}",
        python_module_name=python_module_name,
        container_uri=PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI,
        labels={
            "base_output_dir": f"gs://{BUCKET_NAME}/model-output/{TIMESTAMP}",
            "dataset": dataset_folder,
            "experiment": experiment_name,
        },
    )

    training_args = [
        "--dataset_folder",
        dataset_folder,
        "--max_epochs",
        str(max_epochs),
        "--model_type",
        str(model_type),
    ]

    # >>
    # The fourth step: submit the Custom Job to Vertex Training service
    # <<
    job.run(
        base_output_dir=f"gs://{BUCKET_NAME}/model-output/{TIMESTAMP}",
        replica_count=1,
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        service_account=SERVICE_ACCOUNT,
        tensorboard=TB_RESOURCE_NAME,
        args=training_args,
        sync=True,
        enable_web_access=True,
    )


@main.command()
@click.option(
    "--dataset_folder",
    default="my_backet/Bars/dataset",
    help="GCS path to dataset.",
)
@click.option("--max_epochs", default=75)
@click.option("--model_description", default="Bars Segmentation model", required=True)
@click.option(
    "--experiment_name",
    default="bars segmentation experiment",
    help="A name of the experiment.",
)
@click.option(
    "--model_type",
    default="FPN",
    help="Type of the decoder model",
)
def train_bars_segmentation(
    dataset_folder, max_epochs, model_description, experiment_name, model_type
):
    # >>
    # Zero step: define variable names
    # <<
    # setup variables
    print("start setup")
    APP_NAME = "bars_segmentation"
    python_package_gcs_uri = f"{APP_NAME}/train_bars_segmentation_package"
    PYTHON_PACKAGE_APPLICATION_DIR = "jobs/vertex_train_bars_segmentation"

    # job variables
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    JOB_NAME = f"{APP_NAME}_train_bars_segnet_{TIMESTAMP}"
    PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI = (
        "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-9:latest"
    )
    python_module_name = "trainer.task"

    # >>
    # The first step: initialize the Vertex SDK for Python
    # <<
    print("start Vertex SDK")
    aiplatform.init(
        project=PROJECT_ID, location="us-central1", staging_bucket=BUCKET_NAME
    )
    print("done!")

    # >>
    # The second step: create setup archive of jobs/vertex_train_segmentation dir
    # and upload it to the google storage
    # <<
    # Change dir from ./scripts to ./jobs/vertex_train_barcodes folder
    os.chdir(Path(__file__).parent.resolve() / ".." / PYTHON_PACKAGE_APPLICATION_DIR)
    print(f"Creating distribution of {PYTHON_PACKAGE_APPLICATION_DIR}")

    # checking whether setup archive already exist or not
    for arch in Path("dist").glob("*.gz"):
        # if yes -> remove it
        arch.unlink()

    # create new setup archive
    run_setup("setup.py", script_args=["sdist"])

    # create absolute path to setup archive
    dist = list(Path("dist").glob("*.gz"))[0].absolute()
    print(str(dist))

    # Instantiates a client
    storage_client = storage.Client()
    # instantiates a bucket object owned by this client.
    bucket = storage_client.bucket(BUCKET_NAME)
    # upload setup archive to google storage
    blob = bucket.blob("/".join([python_package_gcs_uri, dist.name]))
    blob.upload_from_filename(dist)
    print(
        f"Distribution of {PYTHON_PACKAGE_APPLICATION_DIR} is uploaded to {BUCKET_NAME}"
    )

    # >>
    # The third step: configure the Custom Job resource
    # <<
    # configure the Custom Job resource
    job = aiplatform.CustomPythonPackageTrainingJob(
        display_name=f"{JOB_NAME}",
        model_description=model_description,
        python_package_gcs_uri=f"gs://{BUCKET_NAME}/{python_package_gcs_uri}/{dist.name}",
        python_module_name=python_module_name,
        container_uri=PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI,
        labels={
            "base_output_dir": f"gs://{BUCKET_NAME}/model-output/{TIMESTAMP}",
            "dataset": dataset_folder,
            "experiment": experiment_name,
        },
    )

    training_args = [
        "--dataset_folder",
        dataset_folder,
        "--max_epochs",
        str(max_epochs),
        "--model_type",
        str(model_type),
    ]

    # >>
    # The fourth step: submit the Custom Job to Vertex Training service
    # <<
    job.run(
        base_output_dir=f"gs://{BUCKET_NAME}/model-output/{TIMESTAMP}",
        replica_count=1,
        machine_type="n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        service_account=SERVICE_ACCOUNT,
        tensorboard=TB_RESOURCE_NAME,
        args=training_args,
        sync=True,
        enable_web_access=True,
    )


if __name__ == "__main__":
    main()
