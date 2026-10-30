"""Expérience 4 ESOL : comparaison scaffold-CV, sans consultation du jeu de test.
Depuis la racine du projet : python models/experiment_4_scaffold.py
Dépendances : pandas numpy rdkit scikit-learn joblib ; catboost facultatif.
Entrée : data/processed/esol_extended_descriptors.csv (16 descripteurs RDKit).
"""
from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_validate
from sklearn.base import clone

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data/processed/esol_extended_descriptors.csv'
OUT = ROOT / 'results/experiment_4_scaffold_D16'
OUT.mkdir(parents=True, exist_ok=True)
TARGET = 'measured log solubility in mols per litre'
BASE = ['AMW','SlogP','TPSA','NumHBD','NumHBA','NumRotatableBonds',
        'NumAromaticRings','FractionCSP3','NumHeavyAtoms','NumRings']
ALIASES = {'NumRings':['NumRings','NumRing','RingCount'],
           'NumRotatableBonds':['NumRotatableBonds','NumRotBonds'],
           'FractionCSP3':['FractionCSP3','FractionCsp3']}
# 6 descripteurs supplémentaires disponibles dans ton KNIME.
# Lus directement du CSV KNIME ; les noms et l'ordre sont conservés.
EXTRA_FUNCS = {
    'LabuteASA':Descriptors.LabuteASA,
    'NumHeteroAtoms':Descriptors.NumHeteroatoms,
    'NumAliphaticRings':Descriptors.NumAliphaticRings,
    'NumSaturatedRings':Descriptors.NumSaturatedRings,
    'HallKierAlpha':Descriptors.HallKierAlpha,
    'Chi0v':Descriptors.Chi0v,
}

def load_data():
    df = pd.read_csv(INPUT)
    for col, variants in ALIASES.items():
        if col not in df.columns:
            found = next((v for v in variants if v in df.columns), None)
            if found: df = df.rename(columns={found:col})
    required = ['smiles',TARGET] + BASE + list(EXTRA_FUNCS)
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f'Colonnes absentes : {missing}')
    for c in BASE + list(EXTRA_FUNCS) + [TARGET]: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=required)
    df = df[np.isfinite(df[BASE + list(EXTRA_FUNCS) + [TARGET]].to_numpy(dtype=float)).all(axis=1)].copy()
    df['mol'] = df['smiles'].map(lambda s: Chem.MolFromSmiles(str(s)))
    df = df[df['mol'].notna()].copy()
    df['canonical'] = df['mol'].map(lambda m: Chem.MolToSmiles(m,canonical=True))
    # Même politique de dédoublonnage que l'expérience représentations.
    agg = {c:'first' for c in BASE + list(EXTRA_FUNCS)}
    agg[TARGET] = 'median'
    df = df.groupby('canonical',as_index=False,sort=True).agg(agg)
    mols = [Chem.MolFromSmiles(s) for s in df.canonical]
    groups = np.array([MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols])
    y = df[TARGET].to_numpy(dtype=float)
    return df,mols,groups,y

def build_features(df,mols):
    ten = df[BASE].to_numpy(dtype=np.float32)
    extra = df[list(EXTRA_FUNCS)].to_numpy(dtype=np.float32)
    sixteen = np.column_stack((ten,extra))
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)
    morgan = np.asarray([gen.GetFingerprintAsNumPy(m) for m in mols],dtype=np.float32)
    maccs = []
    for mol in mols:
        fp = MACCSkeys.GenMACCSKeys(mol)
        bits = np.zeros((167,),dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp,bits)
        maccs.append(bits)
    maccs = np.asarray(maccs,dtype=np.float32)
    return {
        'E1_D10_Morgan':np.column_stack((ten,morgan)),
        'E2_D16_Morgan':np.column_stack((sixteen,morgan)),
        'E3_D16_MACCS':np.column_stack((sixteen,maccs)),
        'E4_D16_Morgan_MACCS':np.column_stack((sixteen,morgan,maccs)),
    }

def main():
    if not INPUT.exists(): raise FileNotFoundError(f'CSV introuvable : {INPUT}')
    df,mols,groups,y = load_data()
    print(f'{len(df)} molécules, {len(set(groups))} scaffolds')
    matrices = build_features(df,mols)
    # Conserver le même protocole de split que les expériences antérieures.
    idx = np.arange(len(df))
    tr,te = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(idx,y,groups))
    # Ici, te est totalement exclu de la sélection des modèles.
    cv = GroupKFold(n_splits=5)
    candidates = []
    try:
        from catboost import CatBoostRegressor
        has_catboost = True
    except ImportError:
        has_catboost = False
        print('CatBoost absent : installez-le avec python -m pip install catboost pour E3/E4.')
    for name,X in matrices.items():
        algorithms = {
            'RandomForest':RandomForestRegressor(n_estimators=250,min_samples_leaf=1,
                                                 random_state=42,n_jobs=-1),
        }
        if name.startswith(('E1_','E2_')):
            algorithms['HistGradientBoosting'] = HistGradientBoostingRegressor(
                max_iter=250,learning_rate=.06,max_leaf_nodes=15,min_samples_leaf=10,random_state=42)
        if name.startswith(('E3_','E4_')) and has_catboost:
            algorithms['CatBoost'] = CatBoostRegressor(iterations=500,depth=5,
                learning_rate=.04,loss_function='RMSE',verbose=False,thread_count=-1,random_seed=42)
        for alg,model in algorithms.items():
            start=time.perf_counter()
            scores=cross_validate(model,X[tr],y[tr],groups=groups[tr],cv=cv,
                scoring={'rmse':'neg_root_mean_squared_error','mae':'neg_mean_absolute_error','r2':'r2'},
                n_jobs=1,error_score='raise')
            row={'representation':name,'algorithm':alg,'n_features':X.shape[1],
                 'cv_RMSE_mean':float(-scores['test_rmse'].mean()),
                 'cv_RMSE_std':float(scores['test_rmse'].std()),
                 'cv_MAE_mean':float(-scores['test_mae'].mean()),
                 'cv_R2_mean':float(scores['test_r2'].mean()),
                 'seconds':round(time.perf_counter()-start,1)}
            candidates.append((row,model,X))
            print(f'{name:23s} {alg:22s} CV RMSE={row["cv_RMSE_mean"]:.4f} ({row["seconds"]} s)',flush=True)
    candidates.sort(key=lambda x:x[0]['cv_RMSE_mean'])
    rows=[{**row,'selected_by_cv':i==0} for i,(row,_,_) in enumerate(candidates)]
    pd.DataFrame(rows).to_csv(OUT/'cv_comparison.csv',index=False)
    winner,model,X = candidates[0]
    model=clone(model).fit(X[tr],y[tr])
    # Sauvegarde du modèle candidat uniquement ; le jeu de test n'est pas évalué.
    joblib.dump({'model':model,'representation':winner['representation'],
                 'base_features':BASE,'extra_features':list(EXTRA_FUNCS), 'feature_source':'KNIME CSV',
                 'morgan_radius':2,'morgan_bits':2048,'split':'scaffold',
                 'note':'Ne pas charger dans Streamlit sans adapter feature_matrix aux nouvelles representations'},
                OUT/'best_cv_candidate.joblib')
    protocol={'n_molecules':len(df),'n_scaffolds':len(set(groups)),
              'n_train':len(tr),'n_test_reserved':len(te),
              'holdout':'GroupShuffleSplit(test_size=0.2,random_state=42)',
              'cv':'GroupKFold(n_splits=5) on training only',
              'selection':'minimum CV RMSE','test_evaluated':False,
              'winner':winner,
              'caution':'Ancien holdout déjà consulté : validation externe indépendante nécessaire.'}
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2,ensure_ascii=False),encoding='utf-8')
    print('\nGAGNANT CV :',winner['representation'],winner['algorithm'],
          'RMSE=',round(winner['cv_RMSE_mean'],4))
    print('Fichiers :',OUT)
    print('Le test scaffold n’a PAS été utilisé pour sélectionner le gagnant.')

if __name__=='__main__':
    main()
