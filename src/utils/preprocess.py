from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

def smiles_to_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)
    arr = np.zeros((1024,))
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)

    return arr