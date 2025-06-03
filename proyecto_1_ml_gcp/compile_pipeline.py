from kfp import compiler
from pipeline_xgboost import pipeline

compiler.Compiler().compile(
    pipeline_func=pipeline,
    package_path="pipeline_xgboost.json"
)