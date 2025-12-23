import numpy as np
import streamlit as st
import tensorflow as tf
from pathlib import Path

# ✅ IMPORTANT: allow Lambda deserialization in Keras 3 (Streamlit Cloud)
import keras
keras.config.enable_unsafe_deserialization()

st.set_page_config(page_title="Nuclear Structure Diagnostic AI", layout="centered")
BASE = Path(__file__).resolve().parent

# ---- If you used a custom Layer in training, define it هنا بنفس الاسم ----
CHI_MAX = 2.645751  # عدّلها إذا كنت تستخدم قيمة مختلفة

class IBMParameterLayer(tf.keras.layers.Layer):
    def call(self, raw):
        # raw shape: (batch,3)
        eps   = 2.0  * tf.keras.activations.sigmoid(raw[:, 0])  # (0,2)
        kappa = 0.25 * tf.keras.activations.sigmoid(raw[:, 1])  # (0,0.25)
        chi   = CHI_MAX * tf.keras.activations.tanh(raw[:, 2])  # (-CHI_MAX, +CHI_MAX)
        return tf.stack([eps, kappa, chi], axis=1)

CUSTOM_OBJECTS = {
    "IBMParameterLayer": IBMParameterLayer,
}

@st.cache_resource
def load_models():
    # ✅ safe_mode=False fixes Lambda layer loading in Keras 3
    inv = tf.keras.models.load_model(
        BASE / "inverse_model.keras",
        compile=False,
        safe_mode=False,
        custom_objects=CUSTOM_OBJECTS
    )
    prm = tf.keras.models.load_model(
        BASE / "param_model.keras",
        compile=False,
        safe_mode=False,
        custom_objects=CUSTOM_OBJECTS
    )
    fwd = tf.keras.models.load_model(
        BASE / "forward_model.keras",
        compile=False,
        safe_mode=False,
        custom_objects=CUSTOM_OBJECTS
    )
    return inv, prm, fwd

inverse_model, param_model, forward_model = load_models()

st.title("Diagnostic AI for Nuclear Shape (IBM-guided)")

def diagnose_from_R(r42, r62):
    if r42 < 2.0:
        return "spherical / near-vibrational"
    elif r42 < 2.6:
        return "transitional / gamma-soft"
    else:
        return "deformed / rotational-like"

mode = st.radio(
    "Choose input mode",
    ["(2) E2/E4/E6 → Compute R4/2 & R6/2 ثم تشخيص + IBM params",
     "(1) (Nb, R4/2, R6/2) → IBM inversion (إذا R عندك جاهزة)"],
    index=0
)

st.divider()

if mode.startswith("(2)"):
    c1, c2, c3 = st.columns(3)
    with c1:
        e2 = st.number_input("E2_1 (keV)", min_value=0.0, max_value=5000.0, value=500.0, step=1.0)
    with c2:
        e4 = st.number_input("E4_1 (keV)", min_value=0.0, max_value=8000.0, value=1200.0, step=1.0)
    with c3:
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

    params = param_model.predict(X, verbose=0)[0]
    eps, kappa, chi = map(float, params)

    st.subheader("Diagnosis")
    st.write(f"**Diagnosis:** {diagnose_from_R(r42, r62)}")

    with st.expander("IBM parameters (inverse-estimated)"):
        st.write(f"ε (eps): {eps:.6f}")
        st.write(f"κ (kappa): {kappa:.6f}")
        st.write(f"χ (chi): {chi:.6f}")

else:
    st.caption("هذا المدخل يستخدم inversion مباشرة: X = [Nb, R4/2, R6/2].")
    nb  = st.number_input("Nb", min_value=0, max_value=40, value=8, step=1)
    r42 = st.number_input("R4/2", min_value=0.5, max_value=4.5, value=2.5, step=0.01)
    r62 = st.number_input("R6/2", min_value=0.5, max_value=8.0, value=4.0, step=0.01)

    X = np.array([[nb, r42, r62]], dtype=np.float32)
    params = param_model.predict(X, verbose=0)[0]
    eps, kappa, chi = map(float, params)

    st.subheader("Diagnosis")
    st.write(f"**Diagnosis:** {diagnose_from_R(r42, r62)}")

    with st.expander("IBM parameters (inverse-estimated)"):
        st.write(f"ε (eps): {eps:.6f}")
        st.write(f"κ (kappa): {kappa:.6f}")
        st.write(f"χ (chi): {chi:.6f}")
