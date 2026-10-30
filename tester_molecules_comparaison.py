"""Compare les deux modèles gagnants de l'expérimentation 3 sur 10 molécules.

À placer à la racine de prediction_moleculaire/.
Lancer : python tester_molecules_comparaison.py
"""
import sys
from pathlib import Path

import joblib
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "app"))
from utils.molecular_features import feature_matrix

MODELES = {
    "Scaffold (HistGradientBoosting)": ROOT / "results/optimisation/best_scaffold.joblib",
    "Random (XGBoost)": ROOT / "results/optimisation/best_random.joblib",
}

MOLECULES = {
    "Niacinamide": "NC(=O)c1cccnc1",
    "Cafeine": "Cn1c(=O)c2c(ncn2C)n(C)c1=O",
    "Paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "Aspirine": "CC(=O)Oc1ccccc1C(=O)O",
    "Ibuprofene": "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "Benzene": "c1ccccc1",
    "Toluene": "Cc1ccccc1",
    "Naphtalene": "c1ccc2ccccc2c1",
    "Acide benzoique": "O=C(O)c1ccccc1",
    "Uree": "NC(N)=O",
}


def interpreter(solubilite_g_l):
    if solubilite_g_l >= 100:
        return "Tres elevee"
    if solubilite_g_l >= 10:
        return "Elevee"
    if solubilite_g_l >= 1:
        return "Limitee"
    if solubilite_g_l >= 0.1:
        return "Faible"
    return "Tres faible"


def preparer_variables(mol, payload):
    rep = payload["representation"]
    # Formats attendus : D10_Morgan_r2_2048 ou D10_Morgan_r3_2048.
    morceaux = rep.split("_")
    if len(morceaux) != 4 or morceaux[:2] != ["D10", "Morgan"]:
        raise ValueError(f"Representation inattendue : {rep}")
    radius = int(morceaux[2].removeprefix("r"))
    bits = int(morceaux[3])
    x = feature_matrix(mol, rep, radius=radius, bits=bits)
    noms_attendus = payload.get("features")
    noms_calcules = ["AMW", "SlogP", "TPSA", "NumHBD", "NumHBA",
                     "NumRotatableBonds", "NumAromaticRings", "FractionCSP3",
                     "NumHeavyAtoms", "NumRings"] + [f"Morgan_{i}" for i in range(bits)]
    if noms_attendus is not None and list(noms_attendus) != noms_calcules:
        raise ValueError(f"Ordre des variables incompatible pour {rep}")
    if x.shape[1] != len(noms_calcules):
        raise ValueError(f"Nombre de variables incorrect : {x.shape[1]}")
    return x


def main():
    charges = {}
    for nom, chemin in MODELES.items():
        if not chemin.is_file():
            raise FileNotFoundError(f"Modele introuvable : {chemin}")
        charges[nom] = joblib.load(chemin)
        print(f"{nom} : {charges[nom]['representation']}")

    print("\n" + "=" * 105)
    print("COMPARAISON DES PREDICTIONS - EXPERIMENTATION 3")
    print("=" * 105)
    print(f"{'Molecule':<19}{'Scaffold logS':>15}{'Scaffold g/L':>16}"
          f"{'Random logS':>15}{'Random g/L':>15}{'Ecart g/L':>15}")
    print("-" * 105)

    for nom, smiles in MOLECULES.items():
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError("SMILES invalide")
            resultats = {}
            for label, payload in charges.items():
                x = preparer_variables(mol, payload)
                log_s = float(payload["model"].predict(x)[0])
                g_l = (10 ** log_s) * Descriptors.MolWt(mol)
                resultats[label] = (log_s, g_l)
            s_log, s_g = resultats["Scaffold (HistGradientBoosting)"]
            r_log, r_g = resultats["Random (XGBoost)"]
            print(f"{nom:<19}{s_log:>15.3f}{s_g:>16.3f}"
                  f"{r_log:>15.3f}{r_g:>15.3f}{abs(s_g-r_g):>15.3f}")
        except Exception as exc:
            print(f"{nom}: ERREUR - {exc}")

    print("=" * 105)
    print("\nInterpretation indicative des solubilites en g/L :")
    print("<0.1 tres faible | 0.1-1 faible | 1-10 limitee | 10-100 elevee | >=100 tres elevee")
    print("Note : des predictions differentes ne prouvent pas qu'un modele est plus precis.")
    print("Pour comparer la precision, il faut des mesures experimentales independantes.")


if __name__ == "__main__":
    main()
