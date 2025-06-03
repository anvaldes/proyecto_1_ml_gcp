from kfp import dsl
from kfp.dsl import component
import google.cloud.aiplatform as aip

#----------------------------------------------------------------------

@component(
    base_image="us-central1-docker.pkg.dev/proyecto-1-461620/my-kfp-repo/xgboost-pipeline-img:latest"
)
def run_hyperparameter_tuning(year: int, month: int) -> dict:
    from google.cloud import aiplatform
    from google.cloud.aiplatform import hyperparameter_tuning as hpt
    import json
    import gcsfs

    aiplatform.init(
        project="proyecto-1-461620",
        location="us-central1",
        staging_bucket="gs://proyecto_1_ml_central"
    )

    job = aiplatform.HyperparameterTuningJob(
        display_name="xgb-hpt-job",
        custom_job=aiplatform.CustomJob(
            display_name="xgb-custom-job",
            worker_pool_specs=[
                {
                    "machine_spec": {"machine_type": "n1-standard-4"},
                    "replica_count": 1,
                    "container_spec": {
                        "image_uri": "us-central1-docker.pkg.dev/proyecto-1-461620/my-kfp-repo/xgboost-hpt-img:latest",
                        "args": [
                            f"--year={year}",
                            f"--month={month}",
                            "--n_estimators=10",  # placeholder
                            "--max_depth=7"       # placeholder
                        ]
                    }
                }
            ],
        ),
        metric_spec={"f1_score": "maximize"},
        parameter_spec={
            "n_estimators": hpt.IntegerParameterSpec(min=10, max=10, scale="linear"),
            "max_depth": hpt.IntegerParameterSpec(min=7, max=7, scale="linear"),
        },
        max_trial_count = 1,
        parallel_trial_count = 1,
        search_algorithm="random"
    )

    job.run(service_account="vertex-ai-pipeline-sa@proyecto-1-461620.iam.gserviceaccount.com")

    best_trial = job.trials[0]
    
    best_params = best_trial.parameters
    
    n_estimators_opt = best_params[1].value
    max_depth_opt = best_params[0].value
    
    best_params_dict = {
        'n_estimators': n_estimators_opt,
        'max_depth': max_depth_opt

    }

    return best_params_dict


#----------------------------------------------------------------------

@component(
    base_image="us-central1-docker.pkg.dev/proyecto-1-461620/my-kfp-repo/xgboost-pipeline-img:latest"
)
def train_xgboost_model(
    year: int,
    month: int,
    dataset: str,
    model_output_path: str,
    params: dict

) -> str:
    import pandas as pd
    import xgboost as xgb
    import joblib
    import gcsfs

    path_X = f"gs://proyecto_1_ml_central/datasets/{year:04d}_{month:02d}/X_{dataset}.csv"
    path_y = f"gs://proyecto_1_ml_central/datasets/{year:04d}_{month:02d}/y_{dataset}.csv"

    print('Path X:', path_X)
    print('Path y:', path_y)

    print('Params:', params)

    X = pd.read_csv(path_X)
    y = pd.read_csv(path_y)

    model = xgb.XGBClassifier(
        n_estimators = params['n_estimators'], 
        max_depth = params['max_depth'], 
        learning_rate = 0.2, 
        random_state = 0)

    model = model.fit(X, y)

    model_filename = "model.joblib"
    joblib.dump(model, model_filename)

    fs = gcsfs.GCSFileSystem()
    full_model_path = f"{model_output_path}/{year:04d}_{month:02d}/{model_filename}"
    with fs.open(full_model_path, 'wb') as f:
        with open(model_filename, 'rb') as local_f:
            f.write(local_f.read())

    return full_model_path

#----------------------------------------------------------------------

@component(
    base_image="us-central1-docker.pkg.dev/proyecto-1-461620/my-kfp-repo/xgboost-pipeline-img:latest"
)
def register_model_in_vertex_ai(
    year: int,
    month: int,
    model_output_path: str,
    model_display_name: str
) -> str:
    from google.cloud import aiplatform

    print('INICIO')

    aiplatform.init(
        project="proyecto-1-461620",
        location="us-central1",
        staging_bucket="gs://proyecto_1_ml_central"
    )
    
    print('1'*70)

    artifact_uri = f"{model_output_path}/{year:04d}_{month:02d}"  # Ej: gs://proyecto_1_ml/models/2025_06

    print('2'*70)

    model = aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest"
    )

    print('3'*70)

    return model.resource_name

#----------------------------------------------------------------------

@dsl.pipeline(
    name="xgboost-training-pipeline",
    description="Pipeline que entrena un modelo XGBoost desde un dataset en GCS"
)
def pipeline(
    year: int,
    month: int,
    model_output_path: str
):  

    hpt_step = run_hyperparameter_tuning(year=year, month=month)

    train_step = train_xgboost_model(
        year = year,
        month = month,
        dataset = 'train',
        model_output_path = model_output_path,
        params = hpt_step.output
    )

    train_step.after(hpt_step)

    register_step = register_model_in_vertex_ai(
        year = year,
        month = month,
        model_output_path = model_output_path,
        model_display_name = f"xgboost-model-{year}-{month}"
    )

    register_step.after(train_step)

#----------------------------------------------------------------------