from kfp import dsl
from kfp.v2.dsl import component
from google_cloud_pipeline_components.v1.custom_job import CustomTrainingJobOp

# Common dependencies for all Python-based components
COMMON_PACKAGES = [
    # Ensure AI Platform SDK supports route overrides
    "google-cloud-aiplatform>=1.94.0,<2.0.0",
    "kfp==2.13.0",
    "protobuf<5,>=4.21.1",
    "tabulate>=0.8.6,<1",
    "requests-toolbelt>=0.8.0,<2",
    "PyYAML>=5.3,<7",
    "kubernetes>=8.0.0,<31",
    "kfp-server-api>=2.1.0,<2.5.0",
    "kfp-pipeline-spec==0.6.0",
    "click>=8.0.0,<9",
]

# --- 1) Component: upload a trained model via Vertex AI Python SDK ---
@component(
    base_image="python:3.11-slim",
    packages_to_install=COMMON_PACKAGES,
)
def upload_model(
    project: str,
    location: str,
    model_display_name: str,
    model_dir: str,
    serve_image: str,
) -> str:
    from google.cloud import aiplatform

    # Initialize AI Platform SDK
    aiplatform.init(project=project, location=location)

    # Upload the model and explicitly set health + predict routes
    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=model_dir,
        serving_container_image_uri=serve_image,
        # Use explicit route parameters for ICU
        serving_container_predict_route="/predict",
        serving_container_health_route="/healthz",
    )
    return model.resource_name

# --- 2) Component: create an endpoint via Vertex AI Python SDK ---
@component(
    base_image="python:3.11-slim",
    packages_to_install=COMMON_PACKAGES,
)
def create_endpoint(
    project: str,
    location: str,
    endpoint_display_name: str,
) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_display_name)
    return endpoint.resource_name

# --- 3) Component: deploy a model to an endpoint via Vertex AI Python SDK ---
@component(
    base_image="python:3.11-slim",
    packages_to_install=COMMON_PACKAGES,
)
def deploy_model(
    project: str,
    location: str,
    endpoint_name: str,
    model_name: str,
    deployed_model_display_name: str,
    machine_type: str,
) -> None:
    from google.cloud import aiplatform

    aiplatform.init(project=project, location=location)
    endpoint = aiplatform.Endpoint(endpoint_name)
    model = aiplatform.Model(model_name)

    # Deploy the model with a dedicated machine
    endpoint.deploy(
        model=model,
        deployed_model_display_name=deployed_model_display_name,
        machine_type=machine_type,
        traffic_split={"0": 100},
    )

# --- 4) Pipeline definition ---
@dsl.pipeline(
    name="custom-container-train-upload-deploy",
    description="Train a model in a custom container, upload it, create an endpoint, and deploy.",
)
def pipeline(
    project_id: str,
    region: str,
    train_image_uri: str,
    serve_image_uri: str,
    train_data_gcs: str,
    artifact_bucket: str,
    model_display_name: str = "my-model",
    endpoint_display_name: str = "my-endpoint",
    machine_type: str = "n2-standard-4",
    train_split: float = 0.8,
    learning_rate: float = 0.05,
    n_estimators: int = 500,
    random_state: int = 42,
):
    # 1) Custom training job
    worker_pool_specs = [{
        "machine_spec": {"machine_type": machine_type},
        "replica_count": 1,
        "container_spec": {
            "image_uri": train_image_uri,
            "args": [
                "--data-path", train_data_gcs,
                "--model-dir", f"{artifact_bucket}/model",
            ],
        },
    }]

    train_task = CustomTrainingJobOp(
        display_name="train-step",
        project=project_id,
        location=region,
        worker_pool_specs=worker_pool_specs,
        base_output_directory=f"{artifact_bucket}/trainer-output",
    )
    model_artifact_uri = f"{artifact_bucket}/trainer-output/model"

    # 2) Upload the model with explicit routes
    upload_task = upload_model(
        project=project_id,
        location=region,
        model_display_name=model_display_name,
        model_dir=model_artifact_uri,
        serve_image=serve_image_uri,
    )

    # 3) Create endpoint
    endpoint_task = create_endpoint(
        project=project_id,
        location=region,
        endpoint_display_name=endpoint_display_name,
    )

    # 4) Deploy the model
    deploy_task = deploy_model(
        project=project_id,
        location=region,
        endpoint_name=endpoint_task.output,
        model_name=upload_task.output,
        deployed_model_display_name=f"{model_display_name}-deployed",
        machine_type=machine_type,
    )

    # Define execution order
    upload_task.after(train_task)
    endpoint_task.after(upload_task)
    deploy_task.after(endpoint_task)

