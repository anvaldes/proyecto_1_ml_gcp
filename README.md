# 🤖 XGBoost ML Pipeline on Google Cloud (Vertex AI)

This repository contains a full **machine learning pipeline** built for **Google Cloud Vertex AI**, using **custom Docker containers** and orchestrated with **Kubeflow Pipelines (KFP)**.  
The pipeline performs **hyperparameter tuning**, **training**, and **model registration** using datasets stored in **Google Cloud Storage (GCS)**.

---

## 🚀 Features

- Uses `Vertex AI HyperparameterTuningJob` to find optimal XGBoost parameters  
- Trains a model with the best hyperparameters  
- Stores trained models in GCS  
- Registers models in **Vertex AI Model Registry**  
- Modular pipeline defined in `KFP` with reusable container components  
- Custom Docker images for HPT and training logic  
- Compatible with GCP services and Vertex AI Pipelines

---

## ☁️ Vertex AI Deployment

This project uses Google Cloud's **Vertex AI Pipelines** with custom components.

### 1. Build Docker Images

Build and push your Docker images to Artifact Registry:

```bash
# From the root of proyecto_1_hpt_img/
docker build -t us-central1-docker.pkg.dev/YOUR_PROJECT/my-kfp-repo/xgboost-hpt-img:latest .
docker push us-central1-docker.pkg.dev/YOUR_PROJECT/my-kfp-repo/xgboost-hpt-img:latest
```

### 2. Compile pipeline

```bash
python compile_pipeline.py
```

### 3. Launch Pipeline

```bash
python run_pipeline.py
```

---
