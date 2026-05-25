import streamlit as st
import numpy as np
import pickle

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Startup Profit Prediction",
    page_icon="📈",
    layout="centered"
)

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    with open("mulit-linearmodel.sav", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# ---------------- TITLE ----------------
st.title("📈 Startup Profit Prediction App")

st.write(
    "Enter the startup details below to predict the expected profit."
)

st.markdown("---")

# ---------------- INPUTS ----------------
st.subheader("Enter Startup Details")

rd_spend = st.number_input(
    "R&D Spend",
    min_value=0.0,
    step=1000.0,
    format="%.2f"
)

administration = st.number_input(
    "Administration",
    min_value=0.0,
    step=1000.0,
    format="%.2f"
)

marketing_spend = st.number_input(
    "Marketing Spend",
    min_value=0.0,
    step=1000.0,
    format="%.2f"
)

state = st.selectbox(
    "Select State",
    ("California", "Florida", "New York")
)

# ---------------- STATE ENCODING ----------------
state_florida = 0
state_newyork = 0

if state == "Florida":
    state_florida = 1

elif state == "New York":
    state_newyork = 1

# ---------------- PREDICTION ----------------
if st.button("Predict Profit"):

    # Input array
    features = np.array([[
        rd_spend,
        administration,
        marketing_spend,
        state_florida,
        state_newyork
    ]])

    # Predict
    prediction = float(model.predict(features)[0])

    # Output
    st.success(f"Predicted Profit: ${prediction:,.2f}")

    st.balloons()

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Machine Learning Model using Multiple Linear Regression")