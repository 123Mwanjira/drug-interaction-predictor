# --- IMPORTS ---
import streamlit as st
import sys
import os
import numpy as np

# --- PATH FIX ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- PROJECT IMPORTS ---
from src.data.sample_data import load_sample_data
from src.models.model import train_model
from src.utils.preprocess import smiles_to_fp


# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Drug Interaction Predictor",
    layout="centered"
)


# --- LOAD CSS ---
def load_css():
    with open("app/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# --- FEATURE ENGINEERING ---
def create_features(df):
    X = []
    for _, row in df.iterrows():
        fp_a = smiles_to_fp(row["drug_a"])
        fp_b = smiles_to_fp(row["drug_b"])
        X.append(np.concatenate([fp_a, fp_b]))
    return np.array(X), df["interaction"].values


# --- TRAIN MODEL (CACHED) ---
@st.cache_resource
def get_model():
    df = load_sample_data()
    X, y = create_features(df)
    model = train_model(X, y)
    return model


model = get_model()


# --- UI ---
st.title("💊 Drug Interaction Predictor")
st.write("Enter two drugs in **SMILES format** to predict interaction risk.")


drug_a = st.text_input("Enter Drug A (SMILES)")
drug_b = st.text_input("Enter Drug B (SMILES)")


# --- PREDICTION ---
if st.button("Predict Interaction"):
    if not drug_a or not drug_b:
        st.warning("⚠️ Please enter both drug inputs.")
    else:
        try:
            # Convert SMILES → fingerprints
            fp_a = smiles_to_fp(drug_a)
            fp_b = smiles_to_fp(drug_b)

            input_data = np.concatenate([fp_a, fp_b]).reshape(1, -1)

            # Prediction + probability
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            # Output
            if prediction == 1:
                st.error(f"⚠️ High risk interaction detected! (Confidence: {probability:.2f})")
            else:
                st.success(f"✅ No significant interaction predicted. (Confidence: {1 - probability:.2f})")

        except Exception:
            st.error("❌ Invalid SMILES input. Please check your entries.")