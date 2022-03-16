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
    default="gs://my_backet/Barcodes/dataset_yolo",
    help="GCS path to dataset.",
)
@click.option("--max_epochs", default=100)
@click.option("--model", default="yolov5s")
@click.option("--model_description", default="Barcodes detection model", required=True)
@click.option(
    "--experiment_name",
    default="barcodes detection experiment",
    help="Name of the experiment.",
)
def train_barcode_detection(
    dataset_folder,
    max_epochs,
    model,
    model_description,
    experiment_name,
):
    # >>
    # Zero step: define variable names
    # <<
    # setup variables
    print("start setup")
    APP_NAME = "barcode_detection"
    python_package_gcs_uri = f"{APP_NAME}/train_barcode_detection_package"
    PYTHON_PACKAGE_APPLICATION_DIR = "jobs/vertex_train_yolov5"

    # job variables
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    JOB_NAME = f"{APP_NAME}_train_barcode_detection_{TIMESTAMP}"
    PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI = (
        "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-10:latest"
    )
    python_module_name = "trainer.task"

    # >>
    # The first step: initialize the Vertex SDK for Python
    # <<
    print("start Vertex SDK")
    aiplatform.init(
        # your Google Cloud Project ID or number
        # environment default used is not set
        project=PROJECT_ID,
        # the Vertex AI region you will use
        # defaults to us-central1
        location="us-central1",
        # Google Cloud Storage bucket in same region as location
        # used to stage artifacts
        staging_bucket=BUCKET_NAME,
    )
    print("done!")

    # >>
    # The second step: create setup archive of jobs/vertex_train_yolov5 dir
    # and upload it to the google storage
    # <<
    # Change dir from ./scripts to ./jobs/vertex_train_yolov5 folder
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
        "--model",
        model,
    ]

    # >>
    # The fourth step: submit the Custom Job to Vertex Training service
    # <<
    model = job.run(
        base_output_dir=f"gs://{BUCKET_NAME}/model-output/{TIMESTAMP}",
        replica_count=1,
        machine_type="n1-standard-8",
        accelerator_type="NVIDIA_TESLA_T4",
        accelerator_count=1,
        service_account=SERVICE_ACCOUNT,
        tensorboard=TB_RESOURCE_NAME,
        args=training_args,
        sync=True,
        enable_web_access=True,
    )

    print(f"Model = {model}")


@main.command()
@click.option(
    "--dataset_folder",
    default="gs://my_backet/Barcodes/Barcodes_v1.1_merged",
    help="GCS path to dataset.",
)
@click.option("--max_epochs", default=100)
@click.option("--train_ann_file", default="train/train.json")
@click.option("--val_ann_file", default="val/val.json")
@click.option("--test_ann_file", default="test/test.json")
@click.option("--train_img_prefix", help="train images prefix", default="train/Images")
@click.option("--val_img_prefix", help="val images prefix", default="val/Images")
@click.option("--test_img_prefix", help="test images prefix", default="test/Images")
@click.option(
    "--load_from",
    default="gs://my_backet/models/detectors/yolox/yolox_s_8x8_300e_coco.pth",
    required=False,
)
@click.option("--model_description", default="yolox detector", required=True)
@click.option(
    "--experiment_name",
    default="yolox detector experiment",
    help="A name of the experiment.",
)
def train_barcode_yolox(
    dataset_folder,
    max_epochs,
    train_ann_file,
    val_ann_file,
    test_ann_file,
    train_img_prefix,
    val_img_prefix,
    test_img_prefix,
    load_from,
    model_description,
    experiment_name,
):
    # >>
    # Zero step: define variable names
    # <<
    # setup variables
    APP_NAME = "barcode"
    python_package_gcs_uri = f"{APP_NAME}/train_yolox_package"
    PYTHON_PACKAGE_APPLICATION_DIR = "jobs/vertex_train_yolox"

    # job variables
    TIMESTAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    JOB_NAME = f"{APP_NAME}_train_yolox_detector_{TIMESTAMP}"
    PRE_BUILT_TRAINING_CONTAINER_IMAGE_URI = (
        "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.1-9:latest"
    )
    python_module_name = "trainer.task"

    # >>
    # The first step: initialize the Vertex SDK for Python
    # <<
    aiplatform.init(
        # your Google Cloud Project ID or number
        # environment default used is not set
        project=PROJECT_ID,
        # the Vertex AI region you will use
        # defaults to us-central1
        location="us-central1",
        # Google Cloud Storage bucket in same region as location
        # used to stage artifacts
        staging_bucket=BUCKET_NAME,
        # the name of the experiment
        # experiment=experiment_name,
        # description of the experiment above
        # experiment_description=experiment_description,
    )

    # >>
    # The second step: create setup archive of jobs/vertex_train_yolox dir
    # and upload it to the google storage
    # <<
    # Change dir from ./scripts to ./jobs/vertex_train_yolox folder
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
        "--train_ann_file",
        train_ann_file,
        "--val_ann_file",
        val_ann_file,
        "--test_ann_file",
        test_ann_file,
        "--train_img_prefix",
        train_img_prefix,
        "--val_img_prefix",
        val_img_prefix,
        "--test_img_prefix",
        test_img_prefix,
    ]
    if load_from is not None:
        training_args.extend(["--load_from", load_from])

    # >>
    # The fourth step: submit the Custom Job to Vertex Training service
    # <<
    job.run(
        base_output_dir=f"gs://{BUCKET_NAME}/model-output/{TIMESTAMP}",
        replica_count=1,
        machine_type="n1-standard-8",
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
