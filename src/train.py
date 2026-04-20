# --- IMPORTS ---
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split

from src.data.sample_data import load_sample_data
from src.models.model import train_model
from src.utils.preprocess import smiles_to_fp


# --- FEATURE ENGINEERING ---
def create_features(df):
    X = []
    for _, row in df.iterrows():
        fp_a = smiles_to_fp(row["drug_a"])
        fp_b = smiles_to_fp(row["drug_b"])
        X.append(np.concatenate([fp_a, fp_b]))

    return np.array(X), df["interaction"].values


# --- MAIN TRAINING FUNCTION ---
def main():
    # Load dataset
    df = load_sample_data()

    # Convert to features
    X, y = create_features(df)

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

    # Train model
    model = train_model(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Evaluate
    print("=== MODEL EVALUATION ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))


# --- RUN SCRIPT ---
if __name__ == "__main__":
    main()