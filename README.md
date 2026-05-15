# Student Dropout & Academic Success Prediction

A machine learning project that predicts whether a student is likely to **drop out** or be **enrolled/successful**, using a trained **CatBoost** model. The project includes a **Streamlit** web app for interactive predictions, probability visualization, and exploratory data analysis

## Features

- ✅ Trained CatBoost classification model
- ✅ Prediction UI built with **Streamlit**
- ✅ Confidence/probability distribution visualization (Plotly)
- ✅ Dataset inspection and EDA (distributions, correlations, statistics)
- ✅ Saves and loads preprocessing artifacts (scaler + label encoder)

---

## Project Structure

- `app.py` — Streamlit application (UI + inference + visualizations)
- `train.py` — Training pipeline (model comparison + final CatBoost training)
- `students_dropout_academic_success.csv` — Dataset
- `requirements.txt` — Python dependencies
- `Models/` — Saved model artifacts (`best_model.pkl`, `scaler.pkl`, `label_encoder.pkl`)
- `catboost_info/` — CatBoost training logs (generated during training)

---

## Installation

1. Create/activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

Run the training script:

```bash
python train.py
```

This will:
- clean and impute missing values
- encode the target labels
- split data (80/20, stratified)
- scale features (StandardScaler)
- balance classes using SMOTE
- compare multiple models
- train the final CatBoost model
- save artifacts to `models/` (or `Models/` depending on your run/output)

> Note: Ensure the saved model paths match what `app.py` loads.

---

## Running the Streamlit App

Start the web application:

```bash
streamlit run app.py
```

Then open the provided local URL in your browser.

### App Pages
- **Home**: overview and model info
- **Predictions**: enter numeric features via sliders and get:
  - predicted status
  - confidence score
  - probability bar chart across classes
  - recommendations based on prediction
- **Data Analysis**: feature distributions, correlation heatmap, stats, and feature info
- **About Model**: training and preprocessing summary

---

## Dependencies

Key libraries:
- `streamlit`
- `catboost`
- `scikit-learn`
- `imbalanced-learn` (SMOTE)
- `plotly`

---

## Notes / Assumptions

- The app expects the dataset file `students_dropout_academic_success.csv` to be present.
- The app expects saved preprocessing artifacts:
  - scaler
  - label encoder
  - trained CatBoost model

---

## License

Add a license of your choice (e.g., MIT).
