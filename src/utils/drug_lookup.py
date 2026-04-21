# Drug name → SMILES mapping (expand later if needed)

DRUG_DB = {
    "aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
    "warfarin": "CC1=CC(=O)C2=CC=CC=C2O1",
    "ibuprofen": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
    "paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
    "lisinopril": "CCN(CC)C(=O)C1CCCN1C(=O)O",
}

def get_smiles(drug_name: str):
    """
    Convert drug name → SMILES.
    If already SMILES, return as-is.
    """
    if not drug_name:
        return ""

    key = drug_name.lower().strip()

    return DRUG_DB.get(key, drug_name)