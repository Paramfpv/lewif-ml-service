import joblib
import numpy as np
import pandas as pd
import gradio as gr
from huggingface_hub import hf_hub_download

_bundle = joblib.load(
    hf_hub_download(repo_id="ParamFpv/lewif-crp-model", filename="crp_model.joblib")
)
_model = _bundle["model"]

FEATURES = [
    "age", "female", "bmi", "ever_smoked", "sleep_hours",
    "trouble_sleeping", "vigorous_work", "vigorous_recreation",
    "sedentary_minutes", "ever_drinks",
]


def predict_crp(age, female, bmi, ever_smoked, sleep_hours,
                trouble_sleeping, vigorous_work, vigorous_recreation,
                sedentary_minutes, ever_drinks):
    row = pd.DataFrame([[age, female, bmi, ever_smoked, sleep_hours,
                         trouble_sleeping, vigorous_work, vigorous_recreation,
                         sedentary_minutes, ever_drinks]], columns=FEATURES)
    crp = float(np.expm1(_model.predict(row)[0]))
    return round(max(0.1, crp), 2)


demo = gr.Interface(
    fn=predict_crp,
    inputs=[
        gr.Number(label="Age (years)"),
        gr.Radio([0, 1], label="Sex (0=male, 1=female)"),
        gr.Number(label="BMI (kg/m²)"),
        gr.Radio([0, 1], label="Ever smoked (0=no, 1=yes)"),
        gr.Number(label="Sleep hours per night"),
        gr.Radio([0, 1], label="Trouble sleeping (0=no, 1=yes)"),
        gr.Radio([0, 1], label="Vigorous work activity (0=no, 1=yes)"),
        gr.Radio([0, 1], label="Vigorous recreational activity (0=no, 1=yes)"),
        gr.Number(label="Sedentary minutes per day"),
        gr.Radio([0, 1], label="Ever drinks alcohol (0=no, 1=yes)"),
    ],
    outputs=gr.Number(label="Predicted CRP (mg/L)"),
    title="LEWIF CRP Predictor",
    description=(
        "Predicts C-Reactive Protein (CRP) from lifestyle variables using "
        "XGBoost trained on NHANES 2015-2023 data (26,411 rows). "
        "MAE: 1.89 mg/L | R²: 0.179"
    ),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
