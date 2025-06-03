from google.cloud import aiplatform

# Inicializa Vertex AI
aiplatform.init(
    project="proyecto-1-461620",
    location="us-central1"
)

# Nombre del Hyperparameter Tuning Job (display_name usado al crear el job)
TARGET_DISPLAY_NAME = "xgb-hpt-job"  # 🔁 Reemplaza si usaste otro nombre

# Lista los jobs y filtra por nombre
jobs = aiplatform.HyperparameterTuningJob.list(
    filter=f'display_name="{TARGET_DISPLAY_NAME}"',
    order_by="create_time desc"
)

if not jobs:
    raise ValueError(f"No se encontró ningún HPT Job con nombre: {TARGET_DISPLAY_NAME}")

# Toma el más reciente
job = jobs[0]

print('JOB:', job)
print('-'*70)

# Ordena los trials por métrica (asume 'f1_score')
sorted_trials = sorted(
    job.trials,
    key=lambda t: t.final_measurement.metrics[0].value,
    reverse=True
)


best_trial = sorted_trials[0]

best_params = best_trial.parameters
max_depth_opt = best_params[0].value
n_estimators_opt = best_params[1].value

print('Max Depth:', max_depth_opt)
print('N estimators:', n_estimators_opt)