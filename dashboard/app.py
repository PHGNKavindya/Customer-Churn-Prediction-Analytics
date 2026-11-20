import os
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import joblib

ROOT = Path(__file__).parent
RAW_FILE = ROOT / "data" / "raw" / "IT_customer_churn.csv"
CLEAN_FILE = ROOT / "data" / "processed" / "customer_churn_cleaned.csv"
MODEL_FILE = ROOT / "models" / "churn_pipeline.joblib"
THRESHOLD = 0.322712

st.set_page_config(page_title="Customer Churn Intelligence", page_icon="📊", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
[data-testid="stMetric"] {background: rgba(100,116,139,.08); border: 1px solid rgba(100,116,139,.15); padding: 14px 16px; border-radius: 12px;}
.hero {padding: 1.2rem 1.4rem; border-radius: 16px; background: linear-gradient(120deg,#102a43,#176b87); color:white; margin-bottom: 1rem;}
.hero h1 {margin:0; font-size:2rem;}
.hero p {margin:.35rem 0 0 0; color:#e2f3f7;}
.small-note {color: #64748b; font-size: .88rem;}
</style>
<div class="hero">
  <h1>Customer Churn Intelligence</h1>
  <p>Explore customer retention patterns, understand churn drivers, and estimate churn risk.</p>
</div>
""", unsafe_allow_html=True)

EXPECTED = [
    "gender","SeniorCitizen","Partner","Dependents","tenure","PhoneService",
    "MultipleLines","InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod","MonthlyCharges","TotalCharges","Churn"
]
TARGET = "Churn"

@st.cache_data
def load_data():
    candidate = CLEAN_FILE if CLEAN_FILE.exists() else RAW_FILE
    if not candidate.exists():
        raise FileNotFoundError(f"Dataset not found: {candidate}")
    df = pd.read_csv(candidate)
    # Normalise common CSV variants without silently inventing data.
    df.columns = [str(c).strip() for c in df.columns]
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce")
        if "tenure" in df.columns:
            zero_tenure = pd.to_numeric(df["tenure"], errors="coerce").fillna(0).eq(0)
            df.loc[zero_tenure & df["TotalCharges"].isna(), "TotalCharges"] = 0
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(df["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    if TARGET in df.columns:
        df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df.drop_duplicates().reset_index(drop=True)
    return df

@st.cache_resource
def train_or_load_model(df):
    if MODEL_FILE.exists():
        try:
            obj = joblib.load(MODEL_FILE)
            if isinstance(obj, dict) and "pipeline" in obj:
                return obj["pipeline"], obj.get("threshold", THRESHOLD), "Loaded saved model"
            return obj, THRESHOLD, "Loaded saved model"
        except Exception:
            pass
    features = [
    c for c in df.columns
    if c not in [TARGET, "_ChurnLabel"]
    and c.lower() not in ("customerid", "customer_id")
    ]
    X = df[features].copy()
    y = df[TARGET].astype(str).str.strip().map({"No": 0, "Yes": 1})
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid].astype(int)
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                            ("onehot", OneHotEncoder(handle_unknown="ignore"))])
    prep = ColumnTransformer([("num", numeric, num_cols), ("cat", categorical, cat_cols)])
    model = Pipeline([("prep", prep), ("model", GradientBoostingClassifier(random_state=42))])
    # Keep app startup practical; use a fixed split to report honest held-out metrics.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, proba)
    }
    try:
        joblib.dump({"pipeline": model, "threshold": THRESHOLD}, MODEL_FILE)
    except Exception:
        pass
    return model, THRESHOLD, ("Trained on first launch", metrics)

try:
    df = load_data()
except Exception as e:
    st.error(f"Unable to load the dataset. Check the CSV file in data/raw/. Details: {e}")
    st.stop()

if TARGET not in df.columns:
    st.error("The dataset must include a 'Churn' target column.")
    st.stop()

churn_norm = df[TARGET].astype(str).str.strip().str.lower()
churn_yes = churn_norm.isin(["yes", "1", "true"])
churn_no = churn_norm.isin(["no", "0", "false"])
df["_ChurnLabel"] = np.where(churn_yes, "Yes", np.where(churn_no, "No", df[TARGET].astype(str)))
valid_target = df["_ChurnLabel"].isin(["Yes", "No"])
analysis_df = df.loc[valid_target].copy()
analysis_df["_ChurnLabel"] = analysis_df["_ChurnLabel"].astype(str)

with st.sidebar:
    st.header("Navigation")
    page = st.radio("Choose a page", ["Overview", "Explore churn", "Predict churn", "Data quality", "SQL insights"])
    st.divider()
    st.caption("Project: AWS Customer Churn Prediction & Analytics")
    st.caption(f"Dataset: {len(df):,} rows × {len(df.columns):,} columns")

if page == "Overview":
    st.subheader("Business overview")
    total = len(analysis_df)
    churned = int((analysis_df["_ChurnLabel"] == "Yes").sum())
    rate = churned / total if total else 0
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Customers", f"{total:,}")
    c2.metric("Churned customers", f"{churned:,}")
    c3.metric("Churn rate", f"{rate:.2%}")
    c4.metric("Dataset columns", len(df.columns))
    left,right = st.columns(2)
    with left:
        counts = analysis_df["_ChurnLabel"].value_counts().rename_axis("Churn").reset_index(name="Customers")
        fig = px.pie(counts, names="Churn", values="Customers", hole=.58, title="Customer churn distribution",
                     color="Churn", color_discrete_map={"No":"#23856D","Yes":"#E76F51"})
        fig.update_layout(legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        if "Contract" in df.columns:
            contract = analysis_df.groupby("Contract", dropna=False).agg(
                Customers=("_ChurnLabel","size"),
                Churned=("_ChurnLabel", lambda s: (s=="Yes").sum())
            ).reset_index()
            contract["Churn rate"] = contract["Churned"] / contract["Customers"] * 100
            fig = px.bar(contract, x="Contract", y="Churn rate", text=contract["Churn rate"].map(lambda x:f"{x:.1f}%"),
                         title="Churn rate by contract", color="Contract")
            fig.update_layout(showlegend=False, yaxis_title="Churn rate (%)")
            st.plotly_chart(fig, use_container_width=True)
    st.markdown("### Key takeaways")
    st.markdown("""
    - **Contract type matters:** month-to-month customers show substantially higher churn than customers on longer contracts.
    - **Early tenure is a risk period:** recently joined customers churn more often.
    - **Payment method can signal risk:** electronic-check customers show higher churn in the SQL analysis.
    - Use these patterns to prioritise onboarding, proactive support and retention offers; treat patterns as associations, not proof of causation.
    """)

elif page == "Explore churn":
    st.subheader("Explore churn patterns")
    c1,c2 = st.columns(2)
    with c1:
        if "tenure" in df.columns:
            fig = px.box(analysis_df, x="_ChurnLabel", y="tenure", color="_ChurnLabel",
                         title="Tenure by churn status", labels={"_ChurnLabel":"Churn","tenure":"Tenure (months)"})
            st.plotly_chart(fig, use_container_width=True)
        if "MonthlyCharges" in df.columns:
            fig = px.box(analysis_df, x="_ChurnLabel", y="MonthlyCharges", color="_ChurnLabel",
                         title="Monthly charges by churn status", labels={"_ChurnLabel":"Churn","MonthlyCharges":"Monthly charges"})
            st.plotly_chart(fig, use_container_width=True)
    with c2:
        if "PaymentMethod" in df.columns:
            p = analysis_df.groupby("PaymentMethod").agg(
                Customers=("_ChurnLabel","size"),
                Churned=("_ChurnLabel",lambda s:(s=="Yes").sum())
            ).reset_index()
            p["Churn rate (%)"] = p["Churned"]/p["Customers"]*100
            fig = px.bar(p.sort_values("Churn rate (%)", ascending=False), x="PaymentMethod", y="Churn rate (%)",
                         title="Churn rate by payment method", text=p["Churn rate (%)"].map(lambda x:f"{x:.1f}%"))
            fig.update_layout(xaxis_tickangle=-25)
            st.plotly_chart(fig, use_container_width=True)
        if "InternetService" in df.columns:
            p = analysis_df.groupby("InternetService").agg(
                Customers=("_ChurnLabel","size"),
                Churned=("_ChurnLabel",lambda s:(s=="Yes").sum())
            ).reset_index()
            p["Churn rate (%)"] = p["Churned"]/p["Customers"]*100
            fig = px.bar(p, x="InternetService", y="Churn rate (%)", title="Churn rate by internet service",
                         text=p["Churn rate (%)"].map(lambda x:f"{x:.1f}%"))
            st.plotly_chart(fig, use_container_width=True)
    st.caption("Charts describe associations in the dataset; they do not establish causation.")

elif page == "Predict churn":
    st.subheader("Estimate an individual customer's churn risk")
    st.info("The app uses your trained Gradient Boosting pipeline if supplied. If no saved model is present, it trains a local model on first launch.")
    model, threshold, model_status = train_or_load_model(df)
    if isinstance(model_status, tuple):
        st.caption(f"{model_status[0]} · threshold = {threshold:.3f}")
        with st.expander("First-launch model evaluation"):
            st.json({k: round(float(v),4) for k,v in model_status[1].items()})
    else:
        st.caption(f"{model_status} · threshold = {threshold:.3f}")
    feature_cols = [c for c in df.columns if c not in [TARGET, "_ChurnLabel"] and c.lower() not in ("customerid","customer_id")]
    # Create a form using dataset-derived choices and safe numeric ranges.
    with st.form("prediction_form"):
        st.write("Enter the customer's information")
        values = {}
        cols = st.columns(2)
        for i, col in enumerate(feature_cols):
            with cols[i % 2]:
                series = df[col]
                if pd.api.types.is_numeric_dtype(series):
                    default = float(pd.to_numeric(series, errors="coerce").median())
                    minv = float(pd.to_numeric(series, errors="coerce").min())
                    maxv = float(pd.to_numeric(series, errors="coerce").max())
                    if minv == maxv:
                        values[col] = st.number_input(col, value=default)
                    elif col.lower() == "tenure":
                        values[col] = st.number_input(col, min_value=max(0.0,minv), max_value=maxv, value=min(max(default,minv),maxv), step=1.0)
                    else:
                        values[col] = st.number_input(col, min_value=minv, max_value=maxv, value=min(max(default,minv),maxv))
                else:
                    opts = sorted(series.dropna().astype(str).unique().tolist())
                    if not opts:
                        opts = ["Unknown"]
                    values[col] = st.selectbox(col, opts, index=0)
        submitted = st.form_submit_button("Predict churn risk", type="primary")
    if submitted:
        input_df = pd.DataFrame([{c: values[c] for c in feature_cols}])
        try:
            probability = float(model.predict_proba(input_df)[0,1])
            prediction = "High churn risk" if probability >= threshold else "Lower churn risk"
            a,b = st.columns(2)
            a.metric("Estimated churn probability", f"{probability:.1%}")
            b.metric("Model classification", prediction)
            st.progress(min(max(probability,0.0),1.0))
            if probability >= threshold:
                st.warning("Suggested action: consider proactive outreach, onboarding support, or a suitable retention offer.")
            else:
                st.success("Suggested action: continue normal customer engagement and monitor changes.")
            st.caption("This is a model estimate, not a guarantee about an individual customer's future behaviour.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

elif page == "Data quality":
    st.subheader("Data quality checks")
    missing_by_col = df.isna().sum()
    duplicates = int(df.duplicated().sum())
    c1,c2,c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Exact duplicate rows", f"{duplicates:,}")
    quality = pd.DataFrame({
        "Column": df.columns,
        "Data type": [str(df[c].dtype) for c in df.columns],
        "Missing values": [int(missing_by_col[c]) for c in df.columns],
        "Unique values": [int(df[c].nunique(dropna=False)) for c in df.columns]
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)
    st.markdown("### Preview")
    st.dataframe(df.head(20), use_container_width=True)
    st.download_button("Download data used by dashboard", df.to_csv(index=False).encode("utf-8"),
                       file_name="customer_churn_dashboard_data.csv", mime="text/csv")

elif page == "SQL insights":
    st.subheader("SQL analytics results")
    results_dir = ROOT / "data" / "athena_results"
    files = sorted(results_dir.glob("*.csv"))
    if not files:
        st.warning("No Athena result CSVs were bundled. Add your query-result CSV files to data/athena_results/.")
    else:
        labels = {f.name: f.name.replace("athena_query_","Query ").replace("_"," ").replace(".csv","") for f in files}
        selected = st.selectbox("Select an Athena query result", [f.name for f in files], format_func=lambda x: labels[x])
        result = pd.read_csv(results_dir / selected)
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button("Download selected result", result.to_csv(index=False).encode("utf-8"),
                           file_name=selected, mime="text/csv")
    st.markdown("### Business recommendations")
    st.markdown("""
    1. Prioritise month-to-month customers for retention outreach and explain the value of longer contracts.
    2. Offer early-tenure onboarding check-ins because churn is concentrated among newer customers.
    3. Review the customer experience and available alternatives for electronic-check users.
    4. Investigate high monthly charges and consider targeted plan reviews before renewal.
    """)

st.divider()
st.caption("Academic project prototype · Predictions are estimates · Do not use customer data containing personal identifiers without appropriate authorisation.")
