
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from pathlib import Path

st.set_page_config(page_title="Nuclear Structure Diagnostic AI", layout="centered")

BASE = Path(__file__).resolve().parent

@st.cache_resource
def load_models():
    inv = tf.keras.models.load_model(BASE/"inverse_model.keras", compile=False)
    prm = tf.keras.models.load_model(BASE/"param_model.keras", compile=False)
    fwd = tf.keras.models.load_model(BASE/"forward_model.keras", compile=False)
    return inv, prm, fwd

inverse_model, param_model, forward_model = load_models()

st.title("Diagnostic AI for Nuclear Shape (IBM-guided)")
st.caption("Input either (Element + A/N) → predict R ratios, or (E2/E4/E6) → compute R ratios. Then diagnose structure.")

# ---- Decision regions (simple, transparent, can be adjusted)
def diagnose_from_R(r42, r62):
    # Robust simple rules (common in nuclear structure):
    # Vibrational ~2.0, gamma-soft ~2.5, rotational ~3.33 for R4/2
    if r42 < 2.0:
        return "spherical / near-vibrational"
    elif r42 < 2.6:
        return "transitional / gamma-soft"
    else:
        return "deformed / rotational-like"

def safe_float(x, default=np.nan):
    try:
        return float(x)
    except:
        return default

mode = st.radio(
    "Choose input mode",
    ["(1) Element + A (or N) → Predict R4/2 & R6/2", "(2) E2/E4/E6 → Compute R4/2 & R6/2"],
    index=0
)

st.divider()

# ---------------- Mode 1 ----------------
if mode.startswith("(1)"):
    col1, col2 = st.columns(2)
    with col1:
        element = st.selectbox("Element", ["Xe","Ba","Ce","Nd"])
        use = st.radio("Use", ["A (mass number)", "N (neutron number)"], index=0)
    with col2:
        val = st.number_input("Value", min_value=0, max_value=300, value=132, step=1)
        nb = st.number_input("Nb (boson number)", min_value=0, max_value=40, value=8, step=1)

    # Optional: E2_1 can help; if unknown, set 0 and let model still work (but better give E2 when available)
    e2 = st.number_input("E2_1 (keV) [optional but recommended]", min_value=0.0, max_value=5000.0, value=500.0, step=1.0)

    # Note: inverse_model expects [Nb, R4_2, R6_2] in training,
    # but for deployment we will produce R ratios via a small direct head:
    # We will use a lightweight fallback: forward_model + estimated params from (Nb, R4, R6) not possible.
    # Instead, we load a small lookup table if exists; else we approximate using forward_model with median params.
    # Better solution: use your direct prediction model if you exported it.
    # For now: We provide "Predicted R" using a minimal regressor embedded via coefficients (not available).
    # So we will ask user to provide R4/2, R6/2 OR use Mode(2).
    st.warning("For most accurate use, provide E2/E4/E6 in Mode (2). If you want pure (Element,A/N) prediction, export the direct-prediction model too.")

    r42_in = st.number_input("R4/2 (if you already have it)", min_value=0.5, max_value=4.5, value=2.5, step=0.01)
    r62_in = st.number_input("R6/2 (if you already have it)", min_value=0.5, max_value=8.0, value=4.0, step=0.01)

    X = np.array([[nb, r42_in, r62_in]], dtype=np.float32)
    pred = inverse_model.predict(X, verbose=0)[0]
    r42_pred, r62_pred = float(pred[0]), float(pred[1])

    params = param_model.predict(X, verbose=0)[0]
    eps, kappa, chi = map(float, params)

    st.subheader("Outputs")
    st.write(f"**Predicted R4/2:** {r42_pred:.4f}")
    st.write(f"**Predicted R6/2:** {r62_pred:.4f}")
    st.write(f"**Diagnosis:** {diagnose_from_R(r42_pred, r62_pred)}")

    with st.expander("IBM parameters (inverse-estimated)"):
        st.write(f"ε (eps): {eps:.4f}")
        st.write(f"κ (kappa): {kappa:.4f}")
        st.write(f"χ (chi): {chi:.4f}")

# ---------------- Mode 2 ----------------
else:
    col1, col2, col3 = st.columns(3)
    with col1:
        e2 = st.number_input("E2_1 (keV)", min_value=0.0, max_value=5000.0, value=500.0, step=1.0)
    with col2:
        e4 = st.number_input("E4_1 (keV)", min_value=0.0, max_value=8000.0, value=1200.0, step=1.0)
    with col3:
        e6 = st.number_input("E6_1 (keV)", min_value=0.0, max_value=12000.0, value=2000.0, step=1.0)

    nb = st.number_input("Nb (boson number)", min_value=0, max_value=40, value=8, step=1)

    if e2 <= 0:
        st.error("E2 must be > 0 to compute ratios.")
        st.stop()

    r42 = e4 / e2
    r62 = e6 / e2

    st.subheader("Computed ratios")
    st.write(f"**R4/2 = E4/E2:** {r42:.4f}")
    st.write(f"**R6/2 = E6/E2:** {r62:.4f}")

    X = np.array([[nb, r42, r62]], dtype=np.float32)
    pred = inverse_model.predict(X, verbose=0)[0]
    r42_pred, r62_pred = float(pred[0]), float(pred[1])

    params = param_model.predict(X, verbose=0)[0]
    eps, kappa, chi = map(float, params)

    st.subheader("Diagnosis")
    st.write(f"**Diagnosis:** {diagnose_from_R(r42, r62)}")

    with st.expander("IBM parameters (inverse-estimated)"):
        st.write(f"ε (eps): {eps:.4f}")
        st.write(f"κ (kappa): {kappa:.4f}")
        st.write(f"χ (chi): {chi:.4f}")

    with st.expander("Model check (optional)"):
        st.caption("Here we show the model's internal predicted ratios from the inverse model head (should be close).")
        st.write(f"Inverse-model head R4/2_pred: {r42_pred:.4f}")
        st.write(f"Inverse-model head R6/2_pred: {r62_pred:.4f}")
