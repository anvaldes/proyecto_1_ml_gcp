from google.cloud import aiplatform

aiplatform.init(
    project="proyecto-1-461620",
    location="us-central1"
)

job = aiplatform.PipelineJob(
    display_name="xgboost-training-2025-06",
    template_path="pipeline_xgboost.json",
    parameter_values={
        "year": 2025,
        "month": 6,
        "model_output_path": "gs://proyecto_1_ml_central/models"
    },
    enable_caching=False
)

# 👉 El parámetro `service_account` va aquí
job.run(service_account="vertex-ai-pipeline-sa@proyecto-1-461620.iam.gserviceaccount.com")
