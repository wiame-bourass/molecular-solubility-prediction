"""ESOL: comparison of 5 descriptors, 10 descriptors, Morgan, and combined.
Run from project root: python models/compare_representations.py
The existing baseline results remain untouched.
"""
from pathlib import Path
import time
import json
import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, GroupKFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data/processed/esol_processed_v2.csv'
OUT = ROOT / 'results/representations'
MODEL_DIR = ROOT / 'models/representations'
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
TARGET = 'measured log solubility in mols per litre'
BASE = ['AMW', 'SlogP', 'TPSA', 'NumHBD', 'NumHBA']
# RDKit Descriptor Calculation in KNIME can use different display names.
ALIASES = {
    'NumRotatableBonds': ['NumRotatableBonds', 'NumRotatableBond', 'NumRotBonds'],
    'NumRing': ['NumRing', 'RingCount', 'NumRings'],
    'NumAromaticRings': ['NumAromaticRings', 'NumAromaticRing'],
    'FractionCSP3': ['FractionCSP3', 'FractionCsp3'],
    'NumHeavyAtoms': ['NumHeavyAtoms', 'HeavyAtomCount', 'NumHeavyAtom'],
}

def metrics(y, p):
    return dict(MAE=float(mean_absolute_error(y, p)), RMSE=float(np.sqrt(mean_squared_error(y, p))), R2=float(r2_score(y, p)))

def main():
    if not INPUT.exists():
        raise FileNotFoundError(f'CSV introuvable : {INPUT}')
    df = pd.read_csv(INPUT)
    print(f'CSV : {len(df)} lignes, {len(df.columns)} colonnes')
    print('Colonnes :', list(df.columns))
    for standard, alternatives in ALIASES.items():
        found = next((c for c in alternatives if c in df.columns), None)
        if found is None:
            raise ValueError(f'Descripteur manquant : {standard}. Noms essayés : {alternatives}')
        if found != standard:
            df = df.rename(columns={found: standard})
    EXTRA = list(ALIASES)
    required = BASE + EXTRA + ['smiles', TARGET]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'Colonnes manquantes : {missing}')
    for c in BASE + EXTRA + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=required).copy()
    df = df[np.isfinite(df[BASE + EXTRA + [TARGET]].to_numpy()).all(axis=1)].copy()
    mols = [Chem.MolFromSmiles(str(s)) for s in df.smiles]
    valid = np.array([m is not None for m in mols])
    df = df.iloc[np.where(valid)[0]].copy().reset_index(drop=True)
    mols = [m for m in mols if m is not None]
    df['canonical'] = [Chem.MolToSmiles(m, canonical=True) for m in mols]
    df['scaffold'] = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    duplicates = len(df) - df.canonical.nunique()
    # Match previous baseline deduplication policy: median target and first descriptors.
    df = df.groupby('canonical', sort=True, as_index=False).agg({**{c:'first' for c in BASE+EXTRA}, TARGET:'median', 'scaffold':'first'})
    mols = [Chem.MolFromSmiles(s) for s in df.canonical]
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = np.array([fpgen.GetFingerprintAsNumPy(m) for m in mols], dtype=np.uint8)
    five = df[BASE].to_numpy(dtype=np.float32)
    ten = df[BASE + EXTRA].to_numpy(dtype=np.float32)
    representations = {'E0_5_descripteurs': five, 'E1_10_descripteurs': ten,
                       'E2_Morgan_2048': fingerprints, 'E3_10_plus_Morgan': np.hstack([ten, fingerprints])}
    y = df[TARGET].to_numpy(dtype=float)
    groups = df.scaffold.to_numpy()
    indices = np.arange(len(df))
    random_train, random_test = train_test_split(indices, test_size=.2, random_state=42)
    scaffold_train, scaffold_test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(indices, y, groups))
    all_rows = []
    for split_name, tr, te in [('random', random_train, random_test), ('scaffold', scaffold_train, scaffold_test)]:
        cv = KFold(n_splits=5, shuffle=True, random_state=42) if split_name == 'random' else GroupKFold(n_splits=5)
        cv_groups = None if split_name == 'random' else groups[tr]
        candidates = []
        for rep_name, X in representations.items():
            model = RandomForestRegressor(n_estimators=200, min_samples_leaf=1, random_state=42, n_jobs=-1)
            start = time.perf_counter()
            cvres = cross_validate(model, X[tr], y[tr], cv=cv, groups=cv_groups,
                scoring={'rmse':'neg_root_mean_squared_error','mae':'neg_mean_absolute_error','r2':'r2'}, n_jobs=1)
            cv_rmse = -cvres['test_rmse'].mean()
            row = {'split':split_name, 'representation':rep_name, 'n_train':len(tr), 'n_test':len(te),
                   'cv_RMSE_mean':float(cv_rmse), 'cv_RMSE_std':float(cvres['test_rmse'].std()),
                   'cv_MAE_mean':float(-cvres['test_mae'].mean()), 'cv_R2_mean':float(cvres['test_r2'].mean()),
                   'cv_seconds':round(time.perf_counter()-start, 2)}
            candidates.append((cv_rmse, rep_name, X, row))
            print(f'{split_name:8} {rep_name:20} CV RMSE={cv_rmse:.3f} (test non consulté)')
        # Select representation using train-only CV. Test is evaluated ONCE, for the winner.
        best_rmse, best_name, best_X, best_row = min(candidates, key=lambda item: item[0])
        winner = RandomForestRegressor(n_estimators=200, min_samples_leaf=1, random_state=42, n_jobs=-1)
        winner.fit(best_X[tr], y[tr])
        pred = winner.predict(best_X[te])
        test_scores = metrics(y[te], pred)
        best_row.update({f'test_{k}':v for k,v in test_scores.items()})
        for _, _, _, row in candidates:
            row['selected_by_cv'] = row is best_row
            all_rows.append(row)
        joblib.dump({'model':winner, 'representation':best_name, 'base_features':BASE, 'extra_features':EXTRA,
                     'morgan_radius':2, 'morgan_bits':2048, 'split':split_name}, MODEL_DIR / f'best_{split_name}.joblib')
        pd.DataFrame({'canonical_smiles':df.canonical.iloc[te].to_numpy(), 'y_true':y[te], 'y_pred':pred}).to_csv(OUT / f'predictions_{split_name}.csv', index=False)
        print(f'>>> {split_name} CHOISI : {best_name} | TEST RMSE={test_scores["RMSE"]:.3f}, R²={test_scores["R2"]:.3f}')
    pd.DataFrame(all_rows).to_csv(OUT / 'comparison.csv', index=False)
    (OUT / 'protocol.json').write_text(json.dumps({'source':str(INPUT.relative_to(ROOT)), 'raw_rows':int(len(pd.read_csv(INPUT))),
        'unique_molecules':int(len(df)), 'duplicate_rows':int(duplicates), 'unique_scaffolds':int(len(set(groups))),
        'model':'RandomForestRegressor(n_estimators=200,min_samples_leaf=1,random_state=42)',
        'selection':'minimum CV RMSE on training only', 'test':'only selected representation evaluated',
        'note':'previously viewed test results are exploratory; new split is not an untouched external holdout'}, indent=2), encoding='utf-8')
    print('Terminé. Résultats :', OUT / 'comparison.csv')

if __name__ == '__main__':
    main()
