---
title: LEWIF ML Service
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: gradio
app_file: app.py
pinned: false
---

# LEWIF ML Service

FastAPI microservice that predicts C-Reactive Protein (CRP) from lifestyle variables using an XGBoost model trained on NHANES data.

Used by the [LEWIF](https://levvif.vercel.app) biological age calculator when CRP is missing from a user's lab report.

## Endpoint

```
POST /predict/crp
```

**Request body:**

| Field | Type | Description |
|---|---|---|
| age | float | Age in years |
| female | int | 1 = female, 0 = male |
| bmi | float | BMI (kg/m²) |
| ever_smoked | int | 1 = ever smoked |
| sleep_hours | float | Hours of sleep per night |
| trouble_sleeping | int | 1 = yes |
| vigorous_work | int | 1 = vigorous work activity |
| vigorous_recreation | int | 1 = vigorous recreational activity |
| sedentary_minutes | float | Sedentary minutes per day |
| ever_drinks | int | 1 = drinks alcohol |

**Response:**

```json
{
  "crp_mg_per_l": 1.84,
  "model_mae": 1.89,
  "model_r2": 0.179
}
```

## Model

- Algorithm: XGBoost regression
- Training data: NHANES 2015-2023 (26,411 rows, 3 survey cycles)
- Target: log-transformed CRP (mg/L), inverse-transformed at inference
- MAE: 1.89 mg/L | Median AE: 0.88 mg/L | R2: 0.179
- 19% better than the population flat default (2.0 mg/L)

## References

- Levine, M.E. et al. (2018). PhenoAge. Aging, 10(4), 573-591.
- NHANES - CDC National Health and Nutrition Examination Survey
