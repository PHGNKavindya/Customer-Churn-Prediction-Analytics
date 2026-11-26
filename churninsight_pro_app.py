from pathlib import Path
import os
import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path(__file__).resolve().parent
THRESHOLD = 0.322712

# Find the user's existing project data without copying it into this app package.
# Works when this file is placed in the project root or one folder below it.
SEARCH_BASES = [ROOT, ROOT.parent]
SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".tox"
}

def find_project_file(filenames):
    wanted = {name.lower() for name in filenames}
    for base in SEARCH_BASES:
        if not base.exists():
            continue
        for current, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for filename in files:
                if filename.lower() in wanted:
                    return Path(current) / filename
    return None

CLEAN_FILE = find_project_file([
    "customer_churn_cleaned.csv",
    "cleaned_customer_churn.csv",
    "cleaned_dataset.csv",
    "cleaned_data.csv",
    "IT_customer_churn_cleaned.csv",
])
RAW_FILE = find_project_file([
    "IT_customer_churn.csv",
    "telco_customer_churn.csv",
    "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "customer_churn.csv",
])

ATHENA_FILES = {
    "Overall churn summary": "athena_query_01_overall_churn.csv",
    "Churn by contract": "athena_query_02_contract_churn.csv",
    "Churn by payment method": "athena_query_03_payment_method_churn.csv",
    "Churn by internet service": "athena_query_04_internet_service_churn.csv",
    "Average charges and tenure": "athena_query_05_charges_by_churn.csv",
    "Churn by tenure group": "athena_query_06_tenure_churn.csv",
}

st.set_page_config(
    page_title="ChurnInsight Pro",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------
# Theme and reusable interface elements
# ---------------------------------------------------------------------
st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
      :root {
        --bg: #0a1020;
        --panel: #111b30;
        --panel-2: #16233b;
        --line: rgba(166,190,224,.13);
        --text: #edf3ff;
        --muted: #91a2bd;
        --cyan: #35d6c2;
        --blue: #6e8cff;
        --orange: #ffad70;
      }
      html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
      }
      .stApp {
        background:
          radial-gradient(ellipse at 78% 0%, rgba(45,102,160,.16), transparent 32%),
          radial-gradient(ellipse at 0% 35%, rgba(27,179,163,.08), transparent 28%),
          var(--bg);
        color: var(--text);
      }
      [data-testid="stHeader"] { background: rgba(10,16,32,.88); }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #111b30 0%, #0d1628 100%);
        border-right: 1px solid var(--line);
      }
      [data-testid="stSidebar"] .block-container { padding-top: 1.45rem; }
      .block-container { padding-top: 1.45rem; padding-bottom: 2.8rem; max-width: 1500px; }
      h1,h2,h3,h4 { color: var(--text) !important; font-family: 'Manrope', sans-serif !important; }
      p, label, span, div { }
      [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(24,39,65,.96), rgba(17,27,48,.96));
        border: 1px solid var(--line);
        padding: 17px 18px;
        border-radius: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,.12);
      }
      [data-testid="stMetricLabel"] { color: #9eafc8 !important; font-size: .83rem !important; }
      [data-testid="stMetricValue"] { color: #f2f7ff !important; font-family: 'Manrope',sans-serif; }
      [data-testid="stMetricDelta"] { color: var(--cyan) !important; }
      div[data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--line) !important;
        background: rgba(17,27,48,.70);
        border-radius: 16px;
      }
      .brand {
        display:flex; align-items:center; gap:11px; margin: 0 0 1.3rem 0;
      }
      .brand-mark {
        height:42px; width:42px; border-radius:13px;
        display:flex; align-items:center; justify-content:center;
        color:#061722; font-size:23px; font-weight:900;
        background: linear-gradient(140deg,#54ead6,#6498ff);
        box-shadow: 0 7px 24px rgba(53,214,194,.20);
      }
      .brand-name { color:#f1f6ff; font:800 17px 'Manrope',sans-serif; letter-spacing:-.5px; }
      .brand-sub { color:#8ea2bf; font-size:10px; text-transform:uppercase; letter-spacing:1.7px; margin-top:1px; }
      .eyebrow {
        color: var(--cyan); text-transform:uppercase; letter-spacing:2px;
        font-size:10px; font-weight:700; margin-bottom:7px;
      }
      .hero {
        position:relative; overflow:hidden; border:1px solid rgba(122,167,218,.2);
        border-radius:22px; padding:27px 30px; margin-bottom:21px;
        background: linear-gradient(112deg,rgba(22,48,77,.98),rgba(19,38,65,.94) 57%,rgba(20,64,75,.88));
      }
      .hero:after {
        content:""; position:absolute; width:270px; height:270px; right:-52px; top:-150px;
        border:1px solid rgba(83,225,206,.25); border-radius:50%;
        box-shadow:0 0 0 35px rgba(83,225,206,.035),0 0 0 75px rgba(83,225,206,.025);
      }
      .hero h1 { margin:0; font-size:30px; letter-spacing:-1.2px; }
      .hero p { color:#b3c4da; margin:7px 0 0 0; max-width:700px; font-size:14px; }
      .hero-tag {
        display:inline-block; border:1px solid rgba(91,224,205,.27);
        color:#91f0df; background:rgba(53,214,194,.08); border-radius:30px;
        padding:5px 10px; font-size:10px; letter-spacing:1px; margin-bottom:10px;
      }
      .section-title { color:#eef4ff; font:700 17px 'Manrope',sans-serif; margin:6px 0 12px 0; }
      .section-note { color:#8ea0ba; font-size:12px; margin-top:-6px; margin-bottom:13px; }
      .insight-card {
        background:linear-gradient(145deg,#172642,#111b30);
        border:1px solid var(--line); border-radius:15px; padding:16px 17px; min-height:112px;
      }
      .insight-label { color:#8fa3bf; font-size:11px; text-transform:uppercase; letter-spacing:1.3px; }
      .insight-value { color:#f3f7ff; font:700 23px 'Manrope',sans-serif; margin-top:7px; }
      .insight-sub { color:#8fa3bf; font-size:11px; margin-top:4px; }
      .callout {
        border-left:3px solid var(--cyan); padding:11px 14px; margin:9px 0;
        background:rgba(53,214,194,.055); border-radius:0 10px 10px 0;
        color:#c7d4e8; font-size:13px;
      }
      .risk-high { color:#ffb18c; font-weight:700; }
      .risk-low { color:#62e2c8; font-weight:700; }
      .tiny-muted { color:#8396b2; font-size:11px; }
      .stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {
        border-radius:10px; border:1px solid rgba(77,217,198,.25);
        background:linear-gradient(110deg,#1aa995,#287e99); color:#061a23;
        font-weight:700; min-height:42px;
      }
      .stButton>button:hover, .stDownloadButton>button:hover {
        border-color:#5cebd7; color:#fff; background:#176b78;
      }
      [data-testid="stRadio"] > label { color:#9aacc5 !important; font-size:11px; }
      [data-testid="stRadio"] div[role="radiogroup"] { gap:7px; }
      [data-testid="stRadio"] div[role="radiogroup"] label {
        background:rgba(255,255,255,.025); border:1px solid transparent;
        border-radius:10px; padding:8px 10px;
      }
      [data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) {
        background:rgba(53,214,194,.11); border-color:rgba(53,214,194,.27);
      }
      input, textarea, [data-baseweb="select"] > div {
        background-color:#121f35 !important; color:#eef4ff !important;
        border-color:rgba(166,190,224,.18) !important;
      }
      [data-testid="stDataFrame"] { border:1px solid var(--line); border-radius:12px; overflow:hidden; }
      hr { border-color:var(--line) !important; }
      footer {visibility:hidden;}
      #MainMenu {visibility:hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

PALETTE = ["#35d6c2", "#6e8cff", "#ffad70", "#be8cff", "#5db8e8", "#ec7c9c"]
PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans", color="#b9c8dc", size=11),
    margin=dict(l=16, r=14, t=48, b=16),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#b9c8dc", size=10)),
)

def style_fig(fig, height=320):
    fig.update_layout(**PLOT_LAYOUT, height=height)
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="rgba(180,200,225,.12)")
    fig.update_yaxes(gridcolor="rgba(180,200,225,.09)", zeroline=False)
    return fig

def hero(kicker, title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-tag">{kicker}</div>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def section(title, note=None):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if note:
        st.markdown(f'<div class="section-note">{note}</div>', unsafe_allow_html=True)

def insight(label, value, sub):
    st.markdown(
        f'<div class="insight-card"><div class="insight-label">{label}</div>'
        f'<div class="insight-value">{value}</div><div class="insight-sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# Data and model
# ---------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_data():
    source = CLEAN_FILE if CLEAN_FILE.exists() else RAW_FILE
    if not source.exists():
        raise FileNotFoundError(
            "Could not find your existing customer churn CSV. Keep this app in your project folder, "
            "or one folder below it. The app searches the current folder and its parent for "
            "customer_churn_cleaned.csv, cleaned_dataset.csv, IT_customer_churn.csv, or customer_churn.csv."
        )
    df = pd.read_csv(source)
    df.columns = [str(c).strip() for c in df.columns]
    if "TotalCharges" in df:
        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True), errors="coerce"
        )
        if "tenure" in df:
            tenure = pd.to_numeric(df["tenure"], errors="coerce").fillna(0)
            df.loc[(tenure == 0) & df["TotalCharges"].isna(), "TotalCharges"] = 0
    for col in ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates().reset_index(drop=True)
    if "Churn" not in df:
        raise ValueError("The dataset must contain the target column 'Churn'.")
    churn = df["Churn"].astype(str).str.strip().str.lower()
    df["Churn"] = np.where(churn.isin(["yes", "1", "true"]), "Yes",
                    np.where(churn.isin(["no", "0", "false"]), "No", np.nan))
    df = df.dropna(subset=["Churn"]).reset_index(drop=True)
    df["_ChurnFlag"] = (df["Churn"] == "Yes").astype(int)
    return df

@st.cache_resource(show_spinner="Preparing the churn prediction model...")
def get_model(_data_token):
    # Use the same modelling family and project threshold for the interactive demo.
    data = load_data()
    target = data["Churn"].map({"No": 0, "Yes": 1}).astype(int)
    features = [
        c for c in data.columns
        if c not in ["Churn", "_ChurnFlag", "_ChurnLabel"]
        and c.lower() not in ("customerid", "customer_id", "id")
    ]
    X = data[features].copy()
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num_cols = [c for c in X.columns if c not in cat_cols]

    numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipe, num_cols),
        ("categorical", categorical_pipe, cat_cols),
    ])
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(random_state=42)),
    ])
    X_train, _, y_train, _ = train_test_split(
        X, target, test_size=0.20, random_state=42, stratify=target
    )
    model.fit(X_train, y_train)
    return model, features

def load_athena_result(filename):
    path = find_project_file([filename])
    if path is None:
        return None
    return pd.read_csv(path)

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load the project dataset: {exc}")
    st.stop()

total = len(df)
churned = int(df["_ChurnFlag"].sum())
retained = total - churned
churn_rate = churned / total if total else 0.0
avg_monthly = float(df["MonthlyCharges"].mean()) if "MonthlyCharges" in df else 0.0
avg_tenure = float(df["tenure"].mean()) if "tenure" in df else 0.0

# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div class="brand">
          <div class="brand-mark">◈</div>
          <div><div class="brand-name">ChurnInsight</div><div class="brand-sub">Analytics workspace</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="eyebrow">Workspace</div>', unsafe_allow_html=True)
    page = st.radio(
        "NAVIGATION",
        ["Command Center", "Segment Explorer", "Churn Risk Studio", "Data Health", "Athena Insights"],
        label_visibility="collapsed",
    )
    st.divider()
    st.markdown('<div class="eyebrow">Dataset status</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="callout">● Dataset loaded<br><span class="tiny-muted">{total:,} records · {len(df.columns)-1} source fields</span></div>',
        unsafe_allow_html=True,
    )
    st.caption("Academic prototype · Local Streamlit app")

# ---------------------------------------------------------------------
# 1. Command Center
# ---------------------------------------------------------------------
if page == "Command Center":
    hero(
        "CUSTOMER RETENTION INTELLIGENCE",
        "Know your customers. Anticipate churn.",
        "A unified view of customer retention signals, segment behaviour and churn-risk insights.",
    )

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Customer base", f"{total:,}", help="Customers in the cleaned analysis dataset.")
    k2.metric("Customers churned", f"{churned:,}")
    k3.metric("Overall churn rate", f"{churn_rate:.2%}")
    k4.metric("Average monthly charge", f"{avg_monthly:.2f}")

    st.write("")
    left, right = st.columns([1.05, 1.35], gap="large")
    with left:
        section("Retention composition", "Share of retained and churned customers.")
        pie = go.Figure(data=[go.Pie(
            labels=["Retained", "Churned"],
            values=[retained, churned],
            hole=.72,
            marker=dict(colors=["#35d6c2", "#ffad70"], line=dict(color="#111b30", width=4)),
            textinfo="percent",
            textfont=dict(size=12, color="#f5f8ff"),
            hovertemplate="%{label}<br>%{value:,} customers · %{percent}<extra></extra>",
        )])
        pie.add_annotation(
            text=f"<b>{churn_rate:.1%}</b><br><span style='font-size:10px;color:#91a2bd'>CHURN RATE</span>",
            x=.5, y=.5, showarrow=False, font=dict(size=22, color="#f1f6ff"),
        )
        style_fig(pie, 320)
        pie.update_layout(showlegend=True, legend=dict(orientation="h", y=-.08, x=.14))
        st.plotly_chart(pie, use_container_width=True, config={"displayModeBar": False})

    with right:
        section("Churn exposure by contract", "Contract groups compared by observed churn rate.")
        if "Contract" in df:
            contract = (
                df.groupby("Contract", dropna=False)
                .agg(customers=("_ChurnFlag", "size"), churned=("_ChurnFlag", "sum"))
                .reset_index()
            )
            contract["churn_rate"] = contract["churned"] / contract["customers"] * 100
            contract = contract.sort_values("churn_rate")
            fig = go.Figure(go.Bar(
                x=contract["churn_rate"], y=contract["Contract"], orientation="h",
                marker=dict(color=["#35d6c2", "#6e8cff", "#ffad70"][:len(contract)],
                            line=dict(width=0)),
                text=[f"{v:.1f}%" for v in contract["churn_rate"]],
                textposition="outside",
                customdata=contract[["customers", "churned"]],
                hovertemplate="%{y}<br>Churn rate: %{x:.2f}%<br>Customers: %{customdata[0]:,}<br>Churned: %{customdata[1]:,}<extra></extra>",
            ))
            fig.update_layout(xaxis_title="Observed churn rate (%)", yaxis_title="", showlegend=False)
            style_fig(fig, 320)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    section("Signals worth watching", "Patterns from the project's EDA and SQL analysis.")
    s1, s2, s3 = st.columns(3, gap="medium")
    with s1:
        insight("Highest contract churn", "Month-to-month", "Observed churn rate: 42.64%")
    with s2:
        insight("Payment-method signal", "Electronic check", "Observed churn rate: 45.15%")
    with s3:
        insight("Tenure gap", "19.55 months", "Average tenure difference between retained and churned customers")

    st.markdown(
        '<div class="callout">Use these patterns to prioritise investigation and retention experiments. '
        'They show associations in this dataset and do not prove causation.</div>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------
# 2. Segment Explorer
# ---------------------------------------------------------------------
elif page == "Segment Explorer":
    hero(
        "SEGMENT INTELLIGENCE",
        "Explore where churn is concentrated.",
        "Filter customer groups, compare observed churn rates and investigate differences across service and billing segments.",
    )
    dimensions = [c for c in ["Contract", "InternetService", "PaymentMethod", "Partner", "Dependents"] if c in df]
    f1, f2 = st.columns([1, 1])
    with f1:
        dimension = st.selectbox("Analyse segment by", dimensions, index=0)
    with f2:
        min_customers = st.slider("Minimum segment size", min_value=0, max_value=1000, value=0, step=50)

    grouped = (
        df.groupby(dimension, dropna=False)
        .agg(customers=("_ChurnFlag", "size"), churned=("_ChurnFlag", "sum"),
             avg_tenure=("tenure", "mean"), avg_monthly=("MonthlyCharges", "mean"))
        .reset_index()
    )
    grouped["churn_rate"] = grouped["churned"] / grouped["customers"] * 100
    grouped = grouped[grouped["customers"] >= min_customers].sort_values("churn_rate", ascending=False)

    if not grouped.empty:
        top = grouped.iloc[0]
        a, b, c = st.columns(3)
        a.metric("Segments shown", len(grouped))
        b.metric("Highest-rate segment", str(top[dimension]))
        c.metric("Highest observed churn", f"{top['churn_rate']:.2f}%")

    left, right = st.columns([1.2, 1], gap="large")
    with left:
        section(f"Churn rate by {dimension}", "Higher bars indicate a higher observed churn rate.")
        bar = go.Figure(go.Bar(
            x=grouped["churn_rate"], y=grouped[dimension].astype(str), orientation="h",
            marker=dict(color=grouped["churn_rate"], colorscale=[[0, "#35d6c2"], [.55, "#6e8cff"], [1, "#ffad70"]]),
            text=[f"{v:.1f}%" for v in grouped["churn_rate"]], textposition="outside",
            customdata=grouped[["customers", "churned"]],
            hovertemplate="%{y}<br>Churn: %{x:.2f}%<br>Customers: %{customdata[0]:,}<br>Churned: %{customdata[1]:,}<extra></extra>",
        ))
        bar.update_layout(xaxis_title="Observed churn rate (%)", yaxis_title="", showlegend=False)
        style_fig(bar, 390)
        st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})
    with right:
        section("Segment detail", "Counts, churn and average customer characteristics.")
        view = grouped.rename(columns={
            dimension: "Segment", "customers": "Customers", "churned": "Churned",
            "churn_rate": "Churn rate (%)", "avg_tenure": "Avg. tenure (months)",
            "avg_monthly": "Avg. monthly charge",
        })
        st.dataframe(view.round(2), use_container_width=True, hide_index=True, height=385)

    st.download_button(
        "Download segment analysis",
        grouped.to_csv(index=False).encode("utf-8"),
        file_name=f"churn_by_{dimension.lower()}.csv",
        mime="text/csv",
        use_container_width=False,
    )

    section("Customer tenure profile", "Compare customer tenure distributions for retained and churned groups.")
    if "tenure" in df:
        tenure_fig = px.histogram(
            df, x="tenure", color="Churn", nbins=24, barmode="overlay", opacity=.75,
            color_discrete_map={"No": "#6e8cff", "Yes": "#ffad70"},
        )
        tenure_fig.update_layout(xaxis_title="Tenure (months)", yaxis_title="Customers", legend_title="")
        style_fig(tenure_fig, 310)
        st.plotly_chart(tenure_fig, use_container_width=True, config={"displayModeBar": False})

# ---------------------------------------------------------------------
# 3. Churn Risk Studio
# ---------------------------------------------------------------------
elif page == "Churn Risk Studio":
    hero(
        "PREDICTIVE WORKSPACE",
        "Estimate an individual customer's churn risk.",
        "Enter a customer profile and use the Gradient Boosting pipeline to estimate churn probability.",
    )
    st.markdown(
        '<div class="callout">This is an academic demonstration model. Predictions are estimates, not guarantees, '
        'and should not be used as the sole basis for customer decisions.</div>',
        unsafe_allow_html=True,
    )
    model, feature_cols = get_model(str(CLEAN_FILE if CLEAN_FILE.exists() else RAW_FILE))
    st.markdown('<div class="section-title">Customer profile</div>', unsafe_allow_html=True)
    with st.form("churn_profile", border=True):
        left, mid, right = st.columns(3, gap="large")
        col_groups = [feature_cols[0::3], feature_cols[1::3], feature_cols[2::3]]
        input_values = {}
        for container, cols in zip([left, mid, right], col_groups):
            with container:
                for col in cols:
                    series = df[col]
                    if pd.api.types.is_numeric_dtype(series):
                        clean = pd.to_numeric(series, errors="coerce").dropna()
                        minv = float(clean.min()) if len(clean) else 0.0
                        maxv = float(clean.max()) if len(clean) else 100.0
                        default = float(clean.median()) if len(clean) else 0.0
                        if minv == maxv:
                            maxv = minv + 1.0
                        is_integer = col in ["tenure", "SeniorCitizen"]
                        if is_integer:
                            input_values[col] = st.number_input(
                                col, min_value=int(minv), max_value=int(maxv),
                                value=int(round(default)), step=1,
                            )
                        else:
                            input_values[col] = st.number_input(
                                col, min_value=minv, max_value=maxv,
                                value=min(max(default, minv), maxv), step=0.5,
                            )
                    else:
                        options = sorted(series.dropna().astype(str).unique().tolist())
                        if not options:
                            options = ["Unknown"]
                        preferred = "Month-to-month" if col == "Contract" and "Month-to-month" in options else options[0]
                        input_values[col] = st.selectbox(
                            col, options, index=options.index(preferred)
                        )
        submitted = st.form_submit_button("Calculate churn risk", use_container_width=True)

    if submitted:
        profile = pd.DataFrame([{c: input_values[c] for c in feature_cols}])
        try:
            probability = float(model.predict_proba(profile)[0, 1])
            predicted = probability >= THRESHOLD
            st.write("")
            p1, p2 = st.columns([1, 1], gap="large")
            with p1:
                section("Estimated probability")
                st.metric("Churn probability", f"{probability:.1%}")
                st.progress(min(max(probability, 0.0), 1.0))
            with p2:
                section("Model assessment")
                if predicted:
                    st.markdown('<div class="insight-card"><div class="insight-label">CLASSIFICATION</div><div class="insight-value risk-high">Higher risk</div><div class="insight-sub">Probability is at or above the configured decision threshold.</div></div>', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="insight-card"><div class="insight-label">CLASSIFICATION</div><div class="insight-value risk-low">Lower risk</div><div class="insight-sub">Probability is below the configured decision threshold.</div></div>', unsafe_allow_html=True)
            st.caption(f"Decision threshold used in this demo: {THRESHOLD:.4f}.")
            if predicted:
                st.warning("Possible next step: review the customer's service experience and consider proactive retention outreach.")
            else:
                st.success("Possible next step: maintain regular engagement and monitor changes in customer behaviour.")
        except Exception as exc:
            st.error(f"Prediction could not be generated: {exc}")

# ---------------------------------------------------------------------
# 4. Data Health
# ---------------------------------------------------------------------
elif page == "Data Health":
    hero(
        "DATA QUALITY MONITOR",
        "Trust the data behind the decision.",
        "Review dataset completeness, duplicates, column types and a sample of the records used by this dashboard.",
    )
    missing = df.isna().sum()
    duplicates = int(df.drop(columns=["_ChurnFlag"], errors="ignore").duplicated().sum())
    records, fields = st.columns(2)
    records.metric("Validated records", f"{len(df):,}")
    fields.metric("Dataset fields", len([c for c in df.columns if c != "_ChurnFlag"]))

    c1, c2, c3 = st.columns(3)
    c1.metric("Duplicate rows", f"{duplicates:,}")
    c2.metric("Missing cells", f"{int(missing.drop(labels=['_ChurnFlag'], errors='ignore').sum()):,}")
    c3.metric("Complete columns", f"{int((missing == 0).sum())}/{len(df.columns)}")

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        section("Missing values by field")
        quality = pd.DataFrame({
            "Field": [c for c in df.columns if c != "_ChurnFlag"],
            "Missing": [int(df[c].isna().sum()) for c in df.columns if c != "_ChurnFlag"],
        }).sort_values("Missing", ascending=True)
        fig = go.Figure(go.Bar(
            x=quality["Missing"], y=quality["Field"], orientation="h",
            marker_color="#35d6c2",
        ))
        fig.update_layout(xaxis_title="Missing cells", yaxis_title="")
        style_fig(fig, 390)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        section("Field-level audit")
        audit = pd.DataFrame({
            "Field": [c for c in df.columns if c != "_ChurnFlag"],
            "Data type": [str(df[c].dtype) for c in df.columns if c != "_ChurnFlag"],
            "Missing": [int(df[c].isna().sum()) for c in df.columns if c != "_ChurnFlag"],
            "Unique": [int(df[c].nunique(dropna=False)) for c in df.columns if c != "_ChurnFlag"],
        })
        st.dataframe(audit, use_container_width=True, hide_index=True, height=390)

    section("Dataset preview", "The first records from the cleaned customer dataset.")
    st.dataframe(df.drop(columns=["_ChurnFlag"], errors="ignore").head(30), use_container_width=True, hide_index=True)
    st.download_button(
        "Download cleaned dataset",
        df.drop(columns=["_ChurnFlag"], errors="ignore").to_csv(index=False).encode("utf-8"),
        file_name="customer_churn_cleaned.csv",
        mime="text/csv",
    )

# ---------------------------------------------------------------------
# 5. Athena Insights
# ---------------------------------------------------------------------
elif page == "Athena Insights":
    hero(
        "SERVERLESS SQL ANALYTICS",
        "Explore the results of Athena queries.",
        "Review exported query outputs from the AWS analysis and connect the results to business interpretation.",
    )
    queries = ATHENA_FILES
    chosen = st.selectbox("Select an Athena analysis", list(queries.keys()))
    result = load_athena_result(queries[chosen])
    if result is None:
        st.warning("This exported result file is missing. Add the CSV to data/athena_results/.")
    else:
        section(chosen, "Exported Amazon Athena query result · CSV snapshot")
        if "churn_rate_percent" in result.columns:
            metric_row = st.columns(min(3, len(result)))
            if len(result) == 1 and "churn_rate_percent" in result:
                metric_row[0].metric("Overall churn rate", f"{float(result.iloc[0]['churn_rate_percent']):.2f}%")
                metric_row[1].metric("Customers", f"{int(result.iloc[0]['total_customers']):,}")
                metric_row[2].metric("Churned customers", f"{int(result.iloc[0]['churned_customers']):,}")
        if "churn_rate_percent" in result.columns and len(result) > 1:
            cat_col = result.columns[0]
            fig = go.Figure(go.Bar(
                x=result[cat_col].astype(str), y=result["churn_rate_percent"],
                marker=dict(color=PALETTE[:len(result)]),
                text=[f"{x:.1f}%" for x in result["churn_rate_percent"]],
                textposition="outside",
                customdata=result[["total_customers", "churned_customers"]],
                hovertemplate="%{x}<br>Churn rate: %{y:.2f}%<br>Customers: %{customdata[0]:,}<br>Churned: %{customdata[1]:,}<extra></extra>",
            ))
            fig.update_layout(xaxis_title="", yaxis_title="Churn rate (%)")
            style_fig(fig, 320)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        elif "average_monthly_charges" in result.columns:
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=result["churn"], y=result["average_monthly_charges"],
                name="Average monthly charges", marker_color="#35d6c2",
            ))
            fig.update_layout(xaxis_title="Churn status", yaxis_title="Average monthly charges")
            style_fig(fig, 300)
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button(
            "Download query result",
            result.to_csv(index=False).encode("utf-8"),
            file_name=queries[chosen],
            mime="text/csv",
        )

    st.divider()
    section("Business actions to consider")
    a, b, c = st.columns(3)
    with a:
        insight("Contract strategy", "Encourage longer terms", "Test suitable offers for month-to-month customers.")
    with b:
        insight("Early lifecycle", "Improve onboarding", "Prioritise support during the first year.")
    with c:
        insight("Billing experience", "Review e-check journey", "Investigate friction for electronic-check users.")

    st.markdown(
        '<div class="callout">Connection note: this version reads exported Athena result CSVs stored with the app. '
        'It does not submit live SQL queries to AWS Athena.</div>',
        unsafe_allow_html=True,
    )

st.divider()
st.markdown(
    '<div class="tiny-muted">CHURNINSIGHT PRO · Customer churn analytics prototype · '
    'Predictions are estimates and should be interpreted alongside business context.</div>',
    unsafe_allow_html=True,
)
