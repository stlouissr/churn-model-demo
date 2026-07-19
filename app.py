# Streamlit Community Cloud batch churn-scoring app for the Healthy Meals XGBoost model.
#
# Users upload a CSV of customer records (thousands of rows are fine). The app reproduces
# the notebook's exact feature-engineering + one-hot-encoding pipeline, scores every row with
# the trained XGBoost model, and returns a downloadable CSV of churn probabilities.
#
# REQUIRED FILES (commit these next to app.py and requirements.txt in your GitHub repo):
#   - churn_xgb_healthy_meals.pkl       (trained XGBClassifier)
#   - churn_encoder_healthy_meals.pkl   (fitted OneHotEncoder)
# Download both from the Snowflake stage @subscription_data.project_data.churn_model_stage.

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------------------------------------------------------------------
# Artifact locations.
# ----------------------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "churn_xgb_healthy_meals.pkl"
ENCODER_PATH = APP_DIR / "churn_encoder_healthy_meals.pkl"

# ----------------------------------------------------------------------------------------
# The columns a user must provide in the uploaded CSV. Everything else the model needs,
# aka the 9 engineered ratio/trend features and the one-hot columns, is derived by this app,
# exactly as in the training notebook, so the user never enters those by hand.
# ----------------------------------------------------------------------------------------
REQUIRED_CATEGORICAL = ["INCOME_LEVEL", "EDUCATION", "DEVICE_TYPE"]
REQUIRED_NUMERIC = [
    "SUBSCRIPTION_AMOUNT", "AGE", "TECH_COMFORT_SCORE",
    "TOTAL_NUM_SESSIONS", "GROSS_TOTAL_SESSION_LENGTH", "ACTIVE_DAYS",
    "ACTIVE_QUARTERS", "ACTIVE_MONTHS", "MAX_DAILY_SESSIONS",
    "AVG_DAILY_SESSIONS", "STD_DAILY_SESSIONS", "MAX_DAILY_LENGTH",
    "STD_DAILY_LENGTH", "H1_SESSIONS", "H2_SESSIONS", "Q4_SESSIONS",
    "Q4_LENGTH", "Q4_ACTIVE_DAYS", "RECENCY_DAYS",
    "DAYS_SINCE_FIRST_ACTIVE", "NO_ACTIVITY_2022",
]
REQUIRED_COLUMNS = REQUIRED_CATEGORICAL + REQUIRED_NUMERIC


# ----------------------------------------------------------------------------------------
# Load model + encoder once and cache them across reruns.
# ----------------------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading model and encoder...")
def load_artifacts():
    if not MODEL_PATH.exists() or not ENCODER_PATH.exists():
        missing = [p.name for p in (MODEL_PATH, ENCODER_PATH) if not p.exists()]
        raise FileNotFoundError(
            "Missing artifact file(s): " + ", ".join(missing)
            + ". Commit them to the repo next to app.py."
        )
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


# ----------------------------------------------------------------------------------------
# Feature engineering identical to the assignment notebook's logic.
# ----------------------------------------------------------------------------------------
def safe_div(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(b > 0, np.divide(a, b, out=np.zeros_like(a), where=b > 0), 0.0)


def add_engineered_features(df):
    df["SESSIONS_PER_ACTIVE_DAY"]   = safe_div(df["TOTAL_NUM_SESSIONS"], df["ACTIVE_DAYS"])
    df["MINUTES_PER_SESSION"]       = safe_div(df["GROSS_TOTAL_SESSION_LENGTH"], df["TOTAL_NUM_SESSIONS"])
    df["MINUTES_PER_ACTIVE_DAY"]    = safe_div(df["GROSS_TOTAL_SESSION_LENGTH"], df["ACTIVE_DAYS"])
    df["AVG_SESSIONS_PER_QUARTER"]  = safe_div(df["TOTAL_NUM_SESSIONS"], df["ACTIVE_QUARTERS"])
    df["SESSIONS_PER_ACTIVE_MONTH"] = safe_div(df["TOTAL_NUM_SESSIONS"], df["ACTIVE_MONTHS"])
    df["H2_TO_H1_RATIO"]            = safe_div(df["H2_SESSIONS"], df["H1_SESSIONS"] + 1.0)  # +1 smoothing
    df["H2_MINUS_H1"]               = df["H2_SESSIONS"] - df["H1_SESSIONS"]
    df["Q4_SESSION_SHARE"]          = safe_div(df["Q4_SESSIONS"], df["TOTAL_NUM_SESSIONS"])
    df["ACTIVE_DAY_RATE"]           = df["ACTIVE_DAYS"] / 365.0
    return df


def build_feature_matrix(raw, model, encoder):
    """Turn a raw customer CSV into the exact feature matrix the model expects."""
    work = raw.copy()

    # Categorical --> clean strings (encoder handles unseen values via handle_unknown='ignore').
    for c in REQUIRED_CATEGORICAL:
        work[c] = work[c].astype(str).str.strip()

    # Numeric --> coerce; note how many cells failed to parse so we can warn the user.
    for c in REQUIRED_NUMERIC:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    n_bad = int(work[REQUIRED_NUMERIC].isna().sum().sum())
    if n_bad:
        # Mirror the notebook's "missing activity --> 0" convention so scoring can proceed.
        work[REQUIRED_NUMERIC] = work[REQUIRED_NUMERIC].fillna(0.0)

    # Derive the 9 engineered features exactly as in training.
    work = add_engineered_features(work)

    # One-hot encode with the FITTED encoder (transform, never fit_transform).
    cat_in = list(getattr(encoder, "feature_names_in_", REQUIRED_CATEGORICAL))
    encoded = encoder.transform(work[cat_in])
    encoded_df = pd.DataFrame(
        encoded, columns=encoder.get_feature_names_out(cat_in), index=work.index
    )

    combined = pd.concat([work, encoded_df], axis=1)

    # Align to the model's stored training feature order (the source of truth).
    feature_names = getattr(model.get_booster(), "feature_names", None)
    if not feature_names:
        raise RuntimeError("Model has no stored feature names; cannot align columns.")

    # Any name absent here would only be a one-hot column for a category not present in this
    # batch; reindex fills those with 0.0, which is the correct encoding.
    X = combined.reindex(columns=feature_names, fill_value=0.0).astype(float)
    return X, n_bad


def make_template():
    """A tiny example CSV so users know the expected columns/format."""
    example = {
        "CUSTOMER_ID": [100001, 100002],
        "INCOME_LEVEL": ["Medium", "High"],
        "EDUCATION": ["Bachelor", "Graduate"],
        "DEVICE_TYPE": ["Multi-device", "Mobile-only"],
        "SUBSCRIPTION_AMOUNT": [149.0, 99.0],
        "AGE": [41, 33],
        "TECH_COMFORT_SCORE": [3, 4],
        "TOTAL_NUM_SESSIONS": [220, 95],
        "GROSS_TOTAL_SESSION_LENGTH": [5400.0, 1800.0],
        "ACTIVE_DAYS": [180, 70],
        "ACTIVE_QUARTERS": [4, 3],
        "ACTIVE_MONTHS": [11, 6],
        "MAX_DAILY_SESSIONS": [6, 4],
        "AVG_DAILY_SESSIONS": [1.22, 1.36],
        "STD_DAILY_SESSIONS": [0.9, 1.1],
        "MAX_DAILY_LENGTH": [95.0, 60.0],
        "STD_DAILY_LENGTH": [18.0, 12.0],
        "H1_SESSIONS": [130, 40],
        "H2_SESSIONS": [90, 55],
        "Q4_SESSIONS": [40, 30],
        "Q4_LENGTH": [960.0, 700.0],
        "Q4_ACTIVE_DAYS": [35, 25],
        "RECENCY_DAYS": [12, 3],
        "DAYS_SINCE_FIRST_ACTIVE": [330, 200],
        "NO_ACTIVITY_2022": [0, 0],
    }
    return pd.DataFrame(example).to_csv(index=False).encode("utf-8")


# ----------------------------------------------------------------------------------------
# User interface
# ----------------------------------------------------------------------------------------
st.set_page_config(page_title="Healthy Meals Churn Scorer", page_icon=None, layout="wide")
st.title("Healthy Meals - Batch Churn Scorer")
st.write(
    "Upload a CSV of customer records to score each one's probability of churn. "
    "The app rebuilds the engineered features and one-hot encoding automatically, "
    "so you only need to provide the raw columns listed below."
)

with st.sidebar:
    st.header("How to use")
    st.markdown(
        "1. Upload a CSV with the **required columns**.\n"
        "2. (Optional) Adjust the churn threshold.\n"
        "3. Review the summary and **download** the scored file."
    )
    threshold = st.slider(
        "Churn flag threshold", min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        help="Rows with churn probability >= this value are flagged as predicted churners.",
    )
    st.download_button(
        "Download CSV template", data=make_template(),
        file_name="churn_input_template.csv", mime="text/csv",
    )
    with st.expander("Required columns"):
        st.markdown("**Categorical**")
        st.write(REQUIRED_CATEGORICAL)
        st.markdown("**Numeric**")
        st.write(REQUIRED_NUMERIC)
        st.caption(
            "An optional CUSTOMER_ID (or any other) column is passed through untouched to the "
            "output. Engineered features are computed for you; don't include them."
        )

# Load artifacts up front so config problems surface immediately.
try:
    model, encoder = load_artifacts()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load model artifacts: {e}")
    st.stop()

uploaded = st.file_uploader("Upload customer CSV", type=["csv"])

if uploaded is None:
    st.info("Awaiting a CSV upload. Use the sidebar template if you need the column layout.")
    st.stop()

# Read the file.
try:
    raw = pd.read_csv(uploaded)
except Exception as e:  # noqa: BLE001
    st.error(f"Could not read the CSV: {e}")
    st.stop()

if raw.empty:
    st.warning("The uploaded file has no rows.")
    st.stop()

# Normalize headers to UPPERCASE so 'age' / 'Age' / 'AGE' all work.
raw.columns = [str(c).strip().upper() for c in raw.columns]

# Validate that every required column is present.
missing_cols = [c for c in REQUIRED_COLUMNS if c not in raw.columns]
if missing_cols:
    st.error(
        "Your file is missing these required columns:\n\n"
        + ", ".join(missing_cols)
        + "\n\nDownload the template in the sidebar for the exact layout."
    )
    st.stop()

# Score.
try:
    X, n_bad = build_feature_matrix(raw, model, encoder)
    churn_proba = model.predict_proba(X)[:, 1]
except Exception as e:  # noqa: BLE001
    st.error(f"Scoring failed: {e}")
    st.stop()

results = raw.copy()
results["CHURN_PROBABILITY"] = np.round(churn_proba, 6)
results["RENEWAL_PROBABILITY"] = np.round(1.0 - churn_proba, 6)
results["CHURN_PREDICTION"] = (churn_proba >= threshold).astype(int)

# Summary metrics.
st.subheader("Summary")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Customers scored", f"{len(results):,}")
c2.metric("Predicted churners", f"{int(results['CHURN_PREDICTION'].sum()):,}")
c3.metric("Mean churn probability", f"{churn_proba.mean():.3f}")
c4.metric("Flag threshold", f"{threshold:.2f}")

if n_bad:
    st.warning(
        f"{n_bad} numeric cell(s) could not be parsed and were treated as 0 "
        "(matching the notebook's missing-activity convention)."
    )

# Distribution of churn probabilities.
st.subheader("Churn probability distribution")
bins = np.linspace(0, 1, 11)
hist = pd.cut(churn_proba, bins=bins, include_lowest=True).value_counts().sort_index()
hist.index = [f"{b.left:.1f}-{b.right:.1f}" for b in hist.index]
st.bar_chart(hist)

# Preview + download.
st.subheader("Scored records")
st.dataframe(
    results.sort_values("CHURN_PROBABILITY", ascending=False).head(1000),
    use_container_width=True,
)
st.caption("Preview shows the 1,000 highest-risk rows; the download contains all rows.")

st.download_button(
    "Download scored CSV",
    data=results.to_csv(index=False).encode("utf-8"),
    file_name="churn_scored.csv",
    mime="text/csv",
)
