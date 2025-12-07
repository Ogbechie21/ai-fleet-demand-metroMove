import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# ---------- 1. LOAD & PREPARE DATA (same logic as your notebook) ----------

@st.cache_data
def load_data():
    df = pd.read_csv("data/fleet_dummy_5000.csv")

    # Create binary label: high-risk trip (Delayed = 1, others = 0)
    df["status_lower"] = df["status"].str.lower()
    df["high_risk"] = (df["status_lower"] == "delayed").astype(int)

    # Time features
    df["pickup_time"] = pd.to_datetime(df["pickup_time"])
    df["pickup_hour"] = df["pickup_time"].dt.hour
    df["pickup_dayofweek"] = df["pickup_time"].dt.dayofweek

    # Features we decided to use (without profit_margin to avoid leakage)
    features = [
        "distance_km",
        "fuel_cost",
        "driver_pay",
        "toll_cost",
        "load_value",
        "violation_count",
        "speeding_incidents",
        "gps_start_lat",
        "gps_start_lon",
        "gps_end_lat",
        "gps_end_lon",
        "pickup_hour",
        "pickup_dayofweek",
    ]

    X = df[features]
    y_class = df["high_risk"]
    y_reg = df["maintenance_cost"]

    return df, X, y_class, y_reg, features


@st.cache_resource
def train_models():
    df, X, y_class, y_reg, features = load_data()

    # --- Train/test split (just to validate, but final models train on full data) ---
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y_class, test_size=0.2, random_state=42, stratify=y_class
    )
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X, y_reg, test_size=0.2, random_state=42
    )

    # --- Classification model: Balanced Logistic Regression with best params ---
    clf_logreg_bal = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                C=0.01,
                solver="liblinear"
            ))
        ]
    )
    clf_logreg_bal.fit(X_train_c, y_train_c)

    # --- Regression model: Random Forest Regressor with tuned-ish params ---
    reg_rf = Pipeline(
        steps=[
            ("model", RandomForestRegressor(
                n_estimators=50,
                max_depth=10,
                random_state=42
            ))
        ]
    )
    reg_rf.fit(X_train_r, y_train_r)

    # Return models & feature names
    return clf_logreg_bal, reg_rf, features


clf_logreg_bal, reg_rf, FEATURES = train_models()
_, full_X, _, _, _ = load_data()

# ---------- 2. STREAMLIT UI ----------

st.title("🚍 MetroMove Fleet Risk & Maintenance Prediction")
st.write(
    "This app uses machine learning to predict **trip risk** (delayed vs normal) "
    "and estimate **maintenance cost** based on trip and vehicle features."
)

st.sidebar.header("Input Trip / Vehicle Features")

# Use dataset statistics to suggest sensible default ranges
X_desc = full_X.describe()

def num_input_from_stats(label, col_name, step=1.0):
    col_stats = X_desc[col_name]
    min_val = float(col_stats["min"])
    max_val = float(col_stats["max"])
    mean_val = float(col_stats["mean"])
    # Streamlit number_input
    return st.sidebar.number_input(
        label,
        min_value=min_val,
        max_value=max_val,
        value=mean_val,
        step=step,
        format="%.3f"
    )

distance_km = num_input_from_stats("Distance (km)", "distance_km", step=1.0)
fuel_cost = num_input_from_stats("Fuel cost", "fuel_cost", step=1.0)
driver_pay = num_input_from_stats("Driver pay", "driver_pay", step=1.0)
toll_cost = num_input_from_stats("Toll cost", "toll_cost", step=1.0)
load_value = num_input_from_stats("Load value", "load_value", step=10.0)
violation_count = num_input_from_stats("Violation count", "violation_count", step=1.0)
speeding_incidents = num_input_from_stats("Speeding incidents", "speeding_incidents", step=1.0)
gps_start_lat = num_input_from_stats("Start GPS latitude", "gps_start_lat", step=0.01)
gps_start_lon = num_input_from_stats("Start GPS longitude", "gps_start_lon", step=0.01)
gps_end_lat = num_input_from_stats("End GPS latitude", "gps_end_lat", step=0.01)
gps_end_lon = num_input_from_stats("End GPS longitude", "gps_end_lon", step=0.01)

pickup_hour = st.sidebar.slider("Pickup hour (0–23)", 0, 23, 8)
pickup_dayofweek = st.sidebar.slider("Day of week (0=Mon, 6=Sun)", 0, 6, 2)

# Build one-row DataFrame for prediction
input_data = pd.DataFrame(
    [[
        distance_km,
        fuel_cost,
        driver_pay,
        toll_cost,
        load_value,
        violation_count,
        speeding_incidents,
        gps_start_lat,
        gps_start_lon,
        gps_end_lat,
        gps_end_lon,
        pickup_hour,
        pickup_dayofweek
    ]],
    columns=FEATURES
)

st.subheader("Preview of Input Data")
st.write(input_data)

if st.button("🔍 Predict Trip Risk & Maintenance Cost"):
    # Classification
    risk_pred = clf_logreg_bal.predict(input_data)[0]
    risk_prob = clf_logreg_bal.predict_proba(input_data)[0, 1]

    # Regression
    cost_pred = reg_rf.predict(input_data)[0]

    st.subheader("Prediction Results")

    if risk_pred == 1:
        st.error(f"🚨 High-Risk Trip Detected (Delayed probability: {risk_prob:.2f})")
    else:
        st.success(f"✅ Trip Likely Normal (Delayed probability: {risk_prob:.2f})")

    st.info(f"Estimated maintenance cost: **{cost_pred:.2f}** (approximate)")