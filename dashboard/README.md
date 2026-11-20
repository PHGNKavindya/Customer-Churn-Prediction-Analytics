# AWS Customer Churn Prediction & Analytics — Streamlit Dashboard

This dashboard adds a simple user interface to the existing churn analysis project. It includes:
- Business overview KPIs and charts
- Churn exploration charts
- Individual customer churn prediction
- Data-quality summary and dataset preview
- Athena SQL result viewer and business recommendations

## Quick start (Windows)

1. Extract this ZIP.
2. Open the extracted folder in VS Code.
3. Open Terminal → New Terminal.
4. Create and activate a virtual environment if you do not already have one:

   ```powershell
   py -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

5. Install dependencies:

   ```powershell
   pip install -r requirements.txt
   ```

6. Start the dashboard:

   ```powershell
   streamlit run app.py
   ```

7. Your browser should open the local dashboard. If not, open the local URL shown in the terminal (normally `http://localhost:8501`).

## Dataset and model behaviour

- The bundled raw CSV is in `data/raw/IT_customer_churn.csv`.
- If `data/processed/customer_churn_cleaned.csv` exists, the dashboard uses it in preference to the raw CSV.
- On the first use of the prediction page, the app trains a Gradient Boosting pipeline from the bundled dataset and saves it to `models/churn_pipeline.joblib`.
- If you have an already trained pipeline that uses the same feature names and preprocessing, put it at `models/churn_pipeline.joblib` in the supported format: either a scikit-learn Pipeline or a dictionary containing `pipeline` and optional `threshold`.
- The default classification threshold is 0.322712, based on the prior project threshold-tuning result. Retrain and evaluate on your own saved pipeline before reporting performance.
- The app excludes a customer ID field from model features if present.
- The app does not upload data to AWS by itself. The existing AWS S3, Glue, Athena, Lambda and CloudWatch resources remain separate. AWS hosting is optional and should be considered only after local testing.

## Project folders

```text
AWS_Customer_Churn_Streamlit_Dashboard/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   ├── raw/
│   ├── processed/
│   └── athena_results/
├── models/
└── outputs/
```

## Notes for the report

Describe the dashboard as a local Streamlit prototype unless you actually deploy it to a cloud hosting service. Do not claim it is publicly hosted or integrated with AWS unless that integration is implemented and tested.
