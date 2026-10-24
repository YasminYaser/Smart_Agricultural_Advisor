# 🌾 Smart Agricultural Advisor

ML-powered agricultural advisor that recommends the best crop (Random Forest) and estimates expected yield (XGBoost) based on soil and weather data.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Retrain the models (optional)
```bash
python train.py
```
The app also retrains automatically if the saved models can't be loaded on the server.

## Deploy (Streamlit Community Cloud)
1. Upload all files to a GitHub repo.
2. Go to https://share.streamlit.io → **New app** → choose the repo, branch `main`, main file `app.py`.
3. Open **Advanced settings** and pick Python 3.12 (more compatible with the ML libraries), then **Deploy**.
