"""Optimisation ESOL. Depuis la racine : python models/optimiser_modeles.py
Dépendances: pandas numpy scikit-learn rdkit joblib matplotlib ; xgboost optionnel.
N'écrase pas les résultats des expériences précédentes.
"""
from pathlib import Path
import json, time, warnings
import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import (train_test_split, GroupShuffleSplit, KFold, GroupKFold,
                                     RandomizedSearchCV, cross_validate)
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'data/processed/esol_processed_v2.csv'
OUT = ROOT / 'results/optimisation'
OUT.mkdir(parents=True, exist_ok=True)
TARGET = 'measured log solubility in mols per litre'
BASE = ['AMW','SlogP','TPSA','NumHBD','NumHBA']
EXTRA = ['NumRotatableBonds','NumAromaticRings','FractionCSP3','NumHeavyAtoms','NumRings']
RANDOM_STATE = 42
N_ITER = 12  # augmenter à 25 après le premier essai si nécessaire
CV_FOLDS = 3


def load_data():
    if not INPUT.exists():
        raise FileNotFoundError(f'Fichier absent : {INPUT}')
    df = pd.read_csv(INPUT)
    aliases = {'NumRings':['NumRings','NumRing','RingCount'],
               'NumHeavyAtoms':['NumHeavyAtoms','NumHeavyAtom','HeavyAtomCount']}
    for standard, opts in aliases.items():
        if standard not in df.columns:
            found = next((v for v in opts if v in df.columns), None)
            if found: df = df.rename(columns={found:standard})
    cols = BASE + EXTRA + [TARGET,'smiles']
    missing = sorted(set(cols)-set(df.columns))
    if missing: raise ValueError(f'Colonnes manquantes : {missing}')
    for c in BASE + EXTRA + [TARGET]: df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.dropna(subset=cols).copy()
    df = df[np.isfinite(df[BASE+EXTRA+[TARGET]].to_numpy()).all(axis=1)].copy()
    mols = [Chem.MolFromSmiles(str(s)) for s in df['smiles']]
    valid = [m is not None for m in mols]
    df = df.loc[valid].copy().reset_index(drop=True)
    mols = [m for m in mols if m is not None]
    df['canonical'] = [Chem.MolToSmiles(m, canonical=True) for m in mols]
    df['scaffold'] = [MurckoScaffold.MurckoScaffoldSmiles(mol=m) for m in mols]
    df = df.groupby('canonical', sort=True, as_index=False).agg({
        **{c:'first' for c in BASE+EXTRA}, TARGET:'median', 'scaffold':'first'})
    mols = [Chem.MolFromSmiles(s) for s in df['canonical']]
    y = df[TARGET].to_numpy(float)
    groups = df['scaffold'].to_numpy()
    ten = df[BASE+EXTRA].to_numpy(np.float32)
    return df,mols,y,groups,ten


def representations(mols, ten):
    result = {'D10':(ten, BASE+EXTRA)}
    for radius, bits in [(2,1024),(2,2048),(3,2048)]:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius,fpSize=bits)
        fp = np.asarray([generator.GetFingerprintAsNumPy(m) for m in mols],dtype=np.float32)
        result[f'D10_Morgan_r{radius}_{bits}'] = (
            np.hstack([ten,fp]), BASE+EXTRA+[f'Morgan_{i}' for i in range(bits)])
    return result


def model_spaces():
    spaces = {
      'RandomForest': (RandomForestRegressor(random_state=42,n_jobs=-1), {
          'n_estimators':[150,250,400], 'max_depth':[None,8,15,25],
          'min_samples_leaf':[1,2,4], 'max_features':[0.5,0.8,1.0],
          'min_samples_split':[2,5,10]}),
      'ExtraTrees': (ExtraTreesRegressor(random_state=42,n_jobs=-1), {
          'n_estimators':[150,250,400], 'max_depth':[None,8,15,25],
          'min_samples_leaf':[1,2,4], 'max_features':[0.5,0.8,1.0],
          'min_samples_split':[2,5,10]}),
      'HistGradientBoosting': (HistGradientBoostingRegressor(random_state=42), {
          'max_iter':[100,200,350], 'learning_rate':[0.03,0.06,0.1],
          'max_leaf_nodes':[15,31,63], 'min_samples_leaf':[10,20,35],
          'l2_regularization':[0.,0.1,1.]})
    }
    try:
        from xgboost import XGBRegressor
        spaces['XGBoost'] = (XGBRegressor(objective='reg:squarederror',random_state=42,n_jobs=1,tree_method='hist'), {
            'n_estimators':[150,300,500], 'max_depth':[2,4,6],
            'learning_rate':[0.03,0.07,0.12], 'subsample':[0.7,0.9,1.0],
            'colsample_bytree':[0.7,0.9,1.0], 'reg_lambda':[1,5,10]})
    except ImportError:
        print('XGBoost absent : pour l’ajouter, exécuter pip install xgboost')
    return spaces


def main():
    df,mols,y,groups,ten = load_data()
    reps = representations(mols,ten)
    spaces = model_spaces()
    ids = np.arange(len(y))
    random_tr,random_te = train_test_split(ids,test_size=.2,random_state=42)
    scaffold_tr,scaffold_te = next(GroupShuffleSplit(n_splits=1,test_size=.2,random_state=42).split(ids,y,groups))
    rows=[]
    winners={}
    print(f'{len(df)} molécules, {len(set(groups))} scaffolds. {len(reps)} représentations, {len(spaces)} algorithmes.')
    print('Attention : la recherche peut durer longtemps (plusieurs dizaines de minutes selon le PC).',flush=True)
    for split,tr,te in [('random',random_tr,random_te),('scaffold',scaffold_tr,scaffold_te)]:
        cv = (KFold(n_splits=CV_FOLDS,shuffle=True,random_state=42) if split=='random'
              else GroupKFold(n_splits=CV_FOLDS))
        cv_groups = None if split=='random' else groups[tr]
        split_results=[]
        for rep_name,(X,names) in reps.items():
            for alg,(estimator,params) in spaces.items():
                t=time.perf_counter()
                search = RandomizedSearchCV(estimator,params,n_iter=N_ITER,scoring='neg_root_mean_squared_error',
                        cv=cv,random_state=42,n_jobs=1,refit=True,error_score='raise')
                search.fit(X[tr],y[tr],groups=cv_groups)
                row={'split':split,'representation':rep_name,'algorithm':alg,
                     'cv_RMSE':float(-search.best_score_),
                     'cv_RMSE_std':float(search.cv_results_['std_test_score'][search.best_index_]),
                     'best_params':json.dumps(search.best_params_),
                     'seconds':round(time.perf_counter()-t,1),
                     'n_train':len(tr),'n_test':len(te)}
                rows.append(row)
                split_results.append((row,search.best_estimator_,X,names))
                pd.DataFrame(rows).to_csv(OUT/'all_candidates.csv',index=False)
                print(f'{split:8} {rep_name:22} {alg:21} CV RMSE={row["cv_RMSE"]:.3f} ({row["seconds"]:.0f}s)',flush=True)
        best_row,best_model,best_X,best_names = min(split_results,key=lambda z:z[0]['cv_RMSE'])
        pred=best_model.predict(best_X[te])
        score={'MAE':float(mean_absolute_error(y[te],pred)),
               'RMSE':float(np.sqrt(mean_squared_error(y[te],pred))),
               'R2':float(r2_score(y[te],pred))}
        best_row.update({f'test_{k}':v for k,v in score.items()})
        best_row['selected_by_cv']=True
        winners[split]={'representation':best_row['representation'],'algorithm':best_row['algorithm'],
                        'cv_RMSE':best_row['cv_RMSE'],'test':score,'best_params':json.loads(best_row['best_params'])}
        joblib.dump({'model':best_model,'representation':best_row['representation'],
                     'features':best_names,'split':split},OUT/f'best_{split}.joblib')
        pd.DataFrame({'canonical_smiles':df['canonical'].iloc[te].to_numpy(),
                      'y_true':y[te],'y_pred':pred,
                      'residual':pred-y[te],'abs_error':abs(pred-y[te])}).to_csv(
                          OUT/f'predictions_{split}.csv',index=False)
        print(f'>>> {split} CHOISI {best_row["algorithm"]} / {best_row["representation"]}: test RMSE={score["RMSE"]:.3f}, R²={score["R2"]:.3f}',flush=True)
    for row in rows: row.setdefault('selected_by_cv',False)
    pd.DataFrame(rows).to_csv(OUT/'all_candidates.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps({'winners':winners,'n_molecules':len(df),
        'n_scaffolds':len(set(groups)),'n_iter':N_ITER,'cv_folds':CV_FOLDS,
        'selection':'minimum training CV RMSE',
        'caution':'Prior test sets have been examined; results are exploratory, not external validation.'},indent=2),encoding='utf-8')
    print('Résultats :',OUT)

if __name__=='__main__':
    main()
