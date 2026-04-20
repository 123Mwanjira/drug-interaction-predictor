from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import numpy as np

def smiles_to_fp(smiles, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((n_bits,))

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)

    arr = np.zeros((n_bits,))
    for i in range(n_bits):
        arr[i] = fp[i]

    return arr