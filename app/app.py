import streamlit as st
import sys
import os
import numpy as np

# path fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data.sample_data import load_sample_data
from src.models.model import train_model
from src.utils.preprocess import smiles_to_fp
from src.utils.drug_lookup import get_smiles

# ---------------- UI CONFIG ----------------
st.set_page_config(page_title="Drug Interaction Predictor", layout="centered")

# ---------------- CSS ----------------
def load_css():
    with open("app/styles.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

# ---------------- DATA ----------------
def create_features(df):
    X = []
    for _, row in df.iterrows():
        fp_a = smiles_to_fp(row["drug_a"])
        fp_b = smiles_to_fp(row["drug_b"])
        X.append(np.concatenate([fp_a, fp_b]))
    return np.array(X), df["interaction"].values

@st.cache_resource
def get_model():
    df = load_sample_data()
    X, y = create_features(df)
    return train_model(X, y)

model = get_model()

# ---------------- UI ----------------
st.title("💊 Drug Interaction Predictor")
st.write("Enter drug names OR SMILES to predict interaction risk.")

drug_a = st.text_input("Drug A (e.g. Aspirin or SMILES)")
drug_b = st.text_input("Drug B (e.g. Warfarin or SMILES)")

# ---------------- PREDICTION ----------------
if st.button("Predict Interaction"):

    if not drug_a or not drug_b:
        st.warning("Please enter both drugs")
    else:
        try:
            # convert names → SMILES
            smiles_a = get_smiles(drug_a)
            smiles_b = get_smiles(drug_b)

            fp_a = smiles_to_fp(smiles_a)
            fp_b = smiles_to_fp(smiles_b)

            input_data = np.concatenate([fp_a, fp_b]).reshape(1, -1)

            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]

            # RESULT UI (clean + professional)
            if prediction == 1:
                st.error(f"⚠️ High risk interaction detected (Confidence: {probability:.2f})")
            else:
                st.success(f"✅ No significant interaction predicted (Confidence: {1 - probability:.2f})")

        except Exception as e:
            st.error(f"Error: {str(e)}")