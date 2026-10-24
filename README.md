<div align="center">

# 🌾 Smart Agricultural Advisor

**Machine-learning web app that recommends the best crop to grow and estimates the expected yield from soil and weather data.**

[![Live Demo](https://img.shields.io/badge/Live_Demo-Streamlit-2e9e5b?style=for-the-badge&logo=streamlit&logoColor=white)](https://smart-farm-ml.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.12-3776AB?style=for-the-badge&logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-Random_Forest-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Regressor-1f7a45?style=for-the-badge)

[**Open the app →**](https://smart-farm-ml.streamlit.app)

</div>

<!-- Add a screenshot: upload it to docs/screenshot.png, then remove these comment markers
<p align="center"><img src="docs/screenshot.png" alt="App screenshot" width="85%"></p>
-->

---

## Overview

Choosing the right crop depends on soil nutrients, acidity, temperature, humidity and rainfall. This project turns those readings into a clear recommendation in one click.

The app combines two independent models:

| Task | Model | Output |
|------|-------|--------|
| Crop recommendation | Random Forest Classifier | Best crop, match %, and the top 3 candidates |
| Yield estimation | XGBoost Regressor | Expected yield in tonnes per hectare |

## Features

- Recommends one of **22 crops** with a confidence score and the top 3 alternatives
- Estimates the **expected yield (t/ha)** and shows where it sits within the training range
- Seven simple inputs: N, P, K, pH, temperature, humidity, rainfall, plus optional fertilizer amount
- Warns when your inputs fall outside the range the yield model was trained on
- Responsive interface that works on desktop and mobile
- Robust loading: if the saved models don't match the server's library versions, the app retrains them automatically from the CSV files

## Model performance

Evaluated on a held-out 20% test split (`random_state=42`).

| Model | Metric | Result |
|-------|--------|--------|
| Random Forest (100 trees) | Accuracy | **99.55%** |
| XGBoost (200 estimators, depth 6) | R² / MAE / RMSE | **0.988** / 0.138 / 0.212 |

## Datasets

| Dataset | File | Rows | Target |
|---------|------|------|--------|
| Crop recommendation | `Crop_recommendation.csv` | 2,200 | `label` (22 crops) |
| Yield prediction | `Crop Yiled with Soil and Weather.csv` | 2,596 | `yield` (t/ha) |

Crop recommendation features: `N`, `P`, `K` (kg/ha), `temperature` (°C), `humidity` (%), `ph`, `rainfall` (mm).
Yield features: `Fertilizer` (kg/ha), `temp` (°C), `N`, `P`, `K`.

The two datasets are independent (the recommendation data has no yield values, and the yield data has different features), so each model is trained separately and the app combines their outputs.

## Project structure

```
Smart_Agricultural_Advisor/
├── app.py                              # Streamlit interface
├── model_utils.py                      # Model loading / automatic retraining
├── train.py                            # Train both models and save them
├── Smart_Agricultural_Advisor.ipynb    # EDA, training and evaluation notebook
├── Crop_recommendation.csv             # Crop recommendation dataset
├── Crop Yiled with Soil and Weather.csv# Yield dataset
├── rf_model.joblib                     # Trained Random Forest
├── xgb_model.joblib                    # Trained XGBoost
├── label_encoder.joblib                # Crop label encoder
├── fert_mean.joblib                    # Default fertilizer value
├── requirements.txt
└── .streamlit/config.toml              # Theme
```

## Getting started

```bash
# 1. Clone
git clone https://github.com/YasminYaser/Smart_Agricultural_Advisor.git
cd Smart_Agricultural_Advisor

# 2. Install
pip install -r requirements.txt

# 3. Run
streamlit run app.py
```

### Retrain the models (optional)

```bash
python train.py
```

This trains both models from the CSV files and saves them to `models/`.

## Deployment

The app is deployed on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Push the repository to GitHub.
2. Create a new app on Streamlit Cloud and select the repo, branch `main`, and file `app.py`.
3. Under **Advanced settings**, choose Python 3.12, then deploy.

## Limitations

- The yield model was trained on a **separate, narrow dataset** (temperature 24–40 °C, N 59–80, P 18–25, K 15–22, fertilizer 50–80 kg/ha). Outside these ranges the estimate is only a rough guide, and the app says so.
- The yield estimate is **not specific to the recommended crop**.
- Recommendations are decision support, not a replacement for local agronomic advice.

## Tech stack

Python · Streamlit · scikit-learn · XGBoost · pandas · NumPy · joblib
