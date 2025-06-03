import argparse
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score
import json
import gcsfs  # NUEVO
import os
import hypertune

# Argumentos desde Vertex AI
parser = argparse.ArgumentParser()
parser.add_argument("--year", type=int, required=True)
parser.add_argument("--month", type=int, required=True)
parser.add_argument("--n_estimators", type=int, required=True)
parser.add_argument("--max_depth", type=int, required=True)
args = parser.parse_args()

print('Args:', args)

# Rutas a datasets en GCS
X_train_path = f"gs://proyecto_1_ml_central/datasets/{args.year:04d}_{args.month:02d}/X_train.csv"
y_train_path = f"gs://proyecto_1_ml_central/datasets/{args.year:04d}_{args.month:02d}/y_train.csv"

X_val_path = f"gs://proyecto_1_ml_central/datasets/{args.year:04d}_{args.month:02d}/X_val.csv"
y_val_path = f"gs://proyecto_1_ml_central/datasets/{args.year:04d}_{args.month:02d}/y_val.csv"

print("Leyendo datasets...")
X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path)

X_val = pd.read_csv(X_val_path)
y_val = pd.read_csv(y_val_path)

# Entrenar modelo
print("Entrenando modelo...")
model = xgb.XGBClassifier(
    n_estimators=args.n_estimators,
    max_depth=args.max_depth,
    random_state=42
)
model = model.fit(X_train, y_train)

y_val_pred_prob = model.predict_proba(X_val)[:, 1]
y_val_pred = (y_val_pred_prob >= 0.25)

f1 = f1_score(y_val, y_val_pred, average = 'macro')
print('F1 Score:', round(f1*100, 2))

#----------------------------------------------------------------------

print("Reporting metric with hypertune...")
hpt = hypertune.HyperTune()
hpt.report_hyperparameter_tuning_metric(
    hyperparameter_metric_tag = 'f1_score',
    metric_value = float(f1)
)

#----------------------------------------------------------------------

print("Completado.")