from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

#----------------------------------------------------------------------

search_algorithm = None # Esto activa la busqueda bayesiana que es default

#----------------------------------------------------------------------

aiplatform.init(
    project="proyecto-1-461620",
    location="us-central1",
    staging_bucket="gs://proyecto_1_ml_central"
)

job = aiplatform.HyperparameterTuningJob(
    display_name="xgboost-hpt-job",
    custom_job=aiplatform.CustomJob(
        display_name="xgboost-custom-job",
        worker_pool_specs=[
            {
                "machine_spec": {
                    "machine_type": "n1-standard-4"
                },
                "replica_count": 1,
                "container_spec": {
                    "image_uri": "us-central1-docker.pkg.dev/proyecto-1-461620/my-kfp-repo/xgboost-hpt-img:latest",
                    "args": [
                        "--year=2025",
                        "--month=6",
                        "--n_estimators=5",
                        "--max_depth=5"
                    ]
                }
            }
        ],
    ),
    metric_spec={"f1_score": "maximize"},
    parameter_spec={
        "n_estimators": hpt.IntegerParameterSpec(min=2, max=10, scale="linear"),
        "max_depth": hpt.IntegerParameterSpec(min=2, max=10, scale="linear"),
    },
    max_trial_count = 10,
    parallel_trial_count = 3,
    search_algorithm = search_algorithm 
)

job.run(
    service_account="vertex-ai-pipeline-sa@proyecto-1-461620.iam.gserviceaccount.com"
)
