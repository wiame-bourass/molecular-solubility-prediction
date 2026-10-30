"""ESOL: auditable baseline and scaffold generalization benchmark.
Run from project root: python train.py
"""
from pathlib import Path
import json
import warnings
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split, GroupShuffleSplit, KFold, GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SEED = 42
DATA = Path('data/processed/esol_processed.csv')
OUT = Path('results')
MODELS = Path('models')
OUT.mkdir(parents=True, exist_ok=True)
MODELS.mkdir(parents=True, exist_ok=True)
FEATURES = ['AMW', 'SlogP', 'TPSA', 'NumHBD', 'NumHBA']
TARGET = 'measured log solubility in mols per litre'
SMILES = 'smiles'


def score(y_true, y_pred):
    return {'MAE': float(mean_absolute_error(y_true, y_pred)),
            'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'R2': float(r2_score(y_true, y_pred))}


def scaffold(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None, None
    canonical = Chem.MolToSmiles(mol, canonical=True)
    # Acyclic molecules often have empty Murcko scaffolds; grouped together conservatively.
    return canonical, MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def quality_and_eda(df):
    required = FEATURES + [TARGET, SMILES]
    absent = sorted(set(required) - set(df.columns))
    if absent:
        raise ValueError(f'Colonnes manquantes: {absent}. Vérifier le CSV KNIME.')
    print(f'CSV brut: {len(df)} lignes, {df.shape[1]} colonnes')
    quality = {'initial_rows': int(len(df)), 'missing_by_column': df[required].isna().sum().astype(int).to_dict()}
    df = df.copy()
    for col in FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    parsed = df[SMILES].apply(scaffold)
    df['canonical_smiles'] = parsed.map(lambda x: x[0])
    df['scaffold'] = parsed.map(lambda x: x[1])
    quality['invalid_smiles'] = int(df['canonical_smiles'].isna().sum())
    df = df.dropna(subset=FEATURES + [TARGET, 'canonical_smiles']).copy()
    finite = np.isfinite(df[FEATURES + [TARGET]].to_numpy(dtype=float)).all(axis=1)
    quality['non_finite_rows'] = int((~finite).sum())
    df = df.loc[finite].copy()
    quality['duplicate_canonical_smiles'] = int(df['canonical_smiles'].duplicated().sum())
    # If duplicates have conflicting measurements, aggregate target by median; keep first descriptors.
    conflicting = df.groupby('canonical_smiles')[TARGET].nunique().gt(1)
    quality['molecules_with_conflicting_measurements'] = int(conflicting.sum())
    df = df.groupby('canonical_smiles', as_index=False).agg(
        {**{f: 'first' for f in FEATURES}, TARGET: 'median', SMILES: 'first', 'scaffold': 'first'}
    )
    quality['clean_unique_molecules'] = int(len(df))
    quality['unique_scaffolds'] = int(df['scaffold'].nunique())
    (OUT / 'data_quality.json').write_text(json.dumps(quality, indent=2, ensure_ascii=False), encoding='utf-8')
    df[FEATURES + [TARGET]].describe(percentiles=[.05, .25, .5, .75, .95]).T.to_csv(OUT / 'descriptive_statistics.csv')
    df[FEATURES + [TARGET]].corr(numeric_only=True).to_csv(OUT / 'correlations.csv')
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(df[TARGET], bins=30)
    ax.set(xlabel='Solubilité expérimentale (log mol/L)', ylabel='Nombre de molécules', title='Distribution de la cible ESOL')
    fig.tight_layout(); fig.savefig(OUT / 'target_distribution.png', dpi=160); plt.close(fig)
    print('Contrôle qualité:', quality)
    print('Cible:', df[TARGET].describe().round(3).to_dict())
    return df


def candidates():
    return {
        'Dummy': (DummyRegressor(strategy='mean'), {}),
        'Ridge': (Pipeline([('scale', StandardScaler()), ('model', Ridge())]), {'model__alpha': [0.1, 1.0, 10.0, 100.0]}),
        'RandomForest': (RandomForestRegressor(random_state=SEED, n_jobs=-1), {'n_estimators': [200], 'min_samples_leaf': [1, 3, 5], 'max_features': [0.7, 1.0]}),
        'HistGradientBoosting': (HistGradientBoostingRegressor(random_state=SEED, early_stopping=False), {'learning_rate': [0.05, 0.1], 'max_leaf_nodes': [7, 15], 'l2_regularization': [0.0, 1.0]})
    }


def benchmark(df, strategy):
    if strategy == 'random':
        tr, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=SEED)
        cv = KFold(n_splits=5, shuffle=True, random_state=SEED)
        cv_groups = None
    else:
        groups = df['scaffold'].to_numpy()
        tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(df, groups=groups))
        n_groups = len(np.unique(groups[tr]))
        if n_groups < 5:
            raise ValueError('Moins de 5 scaffolds distincts dans le train.')
        cv = GroupKFold(n_splits=5)
        cv_groups = groups[tr]
        assert not (set(groups[tr]) & set(groups[te]))
    train, test = df.iloc[tr], df.iloc[te]
    Xtr, ytr = train[FEATURES], train[TARGET]
    Xte, yte = test[FEATURES], test[TARGET]
    records, fitted = [], {}
    for name, (estimator, grid) in candidates().items():
        search = GridSearchCV(estimator, grid, scoring='neg_root_mean_squared_error', cv=cv, refit=True, n_jobs=-1)
        search.fit(Xtr, ytr, groups=cv_groups)
        fitted[name] = search.best_estimator_
        metrics = score(yte, search.predict(Xte))
        records.append({'split': strategy, 'model': name, 'cv_RMSE': float(-search.best_score_), **metrics,
                        'train_n': len(train), 'test_n': len(test), 'params': str(search.best_params_)})
        print(f'{strategy:8s} {name:21s} CV RMSE={-search.best_score_:.3f} | test RMSE={metrics["RMSE"]:.3f} R²={metrics["R2"]:.3f}')
    # Choose ONLY by CV on train; never use test metrics for selection.
    selected = min(records, key=lambda r: r['cv_RMSE'])['model']
    final = fitted[selected]
    preds = final.predict(Xte)
    pd.DataFrame({'smiles': test[SMILES].to_numpy(), 'actual': yte.to_numpy(), 'predicted': preds,
                  'error': preds - yte.to_numpy()}).to_csv(OUT / f'predictions_{strategy}.csv', index=False)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yte, preds, alpha=.6)
    lim = [min(yte.min(), preds.min()), max(yte.max(), preds.max())]
    ax.plot(lim, lim, '--'); ax.set(xlabel='Mesuré', ylabel='Prédit', title=f'{strategy}: {selected}')
    fig.tight_layout(); fig.savefig(OUT / f'actual_vs_predicted_{strategy}.png', dpi=160); plt.close(fig)
    joblib.dump({'model': final, 'features': FEATURES, 'model_name': selected, 'split_strategy': strategy,
                 'cv_rmse': min(r['cv_RMSE'] for r in records)}, MODELS / f'best_model_{strategy}.joblib')
    print(f'→ Modèle choisi par CV ({strategy}): {selected}; test utilisé une seule fois pour évaluation.')
    return records


def main():
    if not DATA.exists():
        raise FileNotFoundError(f'CSV introuvable: {DATA.resolve()}')
    df = quality_and_eda(pd.read_csv(DATA))
    if len(df) < 50:
        raise ValueError('Trop peu de molécules valides après nettoyage.')
    all_results = benchmark(df, 'random') + benchmark(df, 'scaffold')
    pd.DataFrame(all_results).to_csv(OUT / 'model_comparison.csv', index=False)
    print('\nFichiers produits dans results/ et models/.')
    print('Attention: scores exploratoires sur un seul split; valider sur jeu externe si disponible.')


if __name__ == '__main__':
    main()
