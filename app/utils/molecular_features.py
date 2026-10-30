"""RDKit descriptors matching the KNIME-exported ESOL columns."""
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, rdFingerprintGenerator

BASE = ['AMW','SlogP','TPSA','NumHBD','NumHBA']
EXTRA = ['NumRotatableBonds','NumAromaticRings','FractionCSP3','NumHeavyAtoms','NumRings']
FEATURES = BASE + EXTRA

def parse_smiles(smiles):
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('Saisissez un SMILES non vide.')
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        raise ValueError('SMILES invalide : RDKit ne peut pas interpréter cette structure.')
    return mol

def descriptor_values(mol):
    return {
        'AMW': float(Descriptors.MolWt(mol)),
        'SlogP': float(Crippen.MolLogP(mol)),
        'TPSA': float(rdMolDescriptors.CalcTPSA(mol)),
        'NumHBD': float(rdMolDescriptors.CalcNumHBD(mol)),
        'NumHBA': float(rdMolDescriptors.CalcNumHBA(mol)),
        'NumRotatableBonds': float(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        'NumAromaticRings': float(rdMolDescriptors.CalcNumAromaticRings(mol)),
        'FractionCSP3': float(rdMolDescriptors.CalcFractionCSP3(mol)),
        'NumHeavyAtoms': float(mol.GetNumHeavyAtoms()),
        'NumRings': float(rdMolDescriptors.CalcNumRings(mol)),
    }

def feature_matrix(mol, representation, radius=2, bits=2048, base_features=None, extra_features=None):
    desc = descriptor_values(mol)
    feature_order = list(base_features or BASE) + list(extra_features or EXTRA)
    if not set(feature_order).issubset(desc):
        raise ValueError('Le modèle attend des descripteurs inconnus.')
    if representation in ('E0_5_descripteurs',):
        x = np.array([desc[k] for k in list(base_features or BASE)], dtype=np.float32)
    elif representation in ('E1_10_descripteurs', 'D10'):
        x = np.array([desc[k] for k in feature_order], dtype=np.float32)
    elif representation in ('E2_Morgan_2048',):
        x = np.empty(0, dtype=np.float32)
    elif representation in ('E3_10_plus_Morgan',) or 'Morgan' in representation:
        x = np.array([desc[k] for k in feature_order], dtype=np.float32)
    else:
        raise ValueError(f'Représentation inconnue : {representation}')
    if 'Morgan' in representation:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=int(radius), fpSize=int(bits))
        fp = gen.GetFingerprintAsNumPy(mol).astype(np.float32)
        x = np.concatenate((x, fp))
    return x.reshape(1, -1)
