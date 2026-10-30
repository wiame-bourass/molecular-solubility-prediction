# Molecular Solubility Lab

Application de **prédiction de solubilité aqueuse (logS)** à partir de structures moléculaires SMILES, développée avec **Python, RDKit, scikit-learn, XGBoost, KNIME et Streamlit**.

## Démonstration

L'application comprend quatre pages :

- **Accueil** : présentation du projet et du jeu de données.
- **Prédiction** : saisie SMILES, structure 2D RDKit, identification du nom via PubChem et prédiction de logS, mol/L et g/L.
- **Performance des modèles** : résultats de validation, analyse des erreurs et comparaison des représentations.
- **Exploration moléculaire** : distribution de logS, descripteurs RDKit et structures ESOL.

## Données et préparation

Le projet utilise le jeu de données **ESOL** (solubilité aqueuse expérimentale).
Après normalisation et dédoublonnage : **1 117 molécules uniques**, regroupées en **269 scaffolds**.

La cible est `measured log solubility in mols per litre` (logS). La colonne de solubilité *prédite* fournie par le jeu de données ne doit pas être utilisée comme variable d'entrée, afin d'éviter une fuite de données.

Les variables incluent **10 descripteurs RDKit** et des **empreintes Morgan** (2 048 bits). Le projet compare les séparations **random** et **scaffold**.

## Résultats principaux — expérimentation 3

| Protocole | Modèle | Représentation | RMSE CV | RMSE test | MAE test | R² test |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Scaffold | HistGradientBoosting | D10 + Morgan r2 / 2048 | 0,772 | 0,907 | 0,663 | 0,769 |
| Random | XGBoost | D10 + Morgan r3 / 2048 | 0,629 | 0,581 | 0,428 | 0,919 |

Les erreurs sont exprimées en **unités logS**. Les jeux de test des deux protocoles sont différents : leurs scores ne permettent **pas** de classer directement les algorithmes. La sélection des modèles a été faite sur la RMSE en validation croisée. Les jeux de test ont été consultés au cours du travail exploratoire : les scores ne constituent pas une validation externe indépendante.

L'**expérimentation 2** compare les représentations moléculaires ; ses résultats sont conservés séparément.

## Installation locale

Python et les bibliothèques natives RDKit doivent être compatibles avec votre système.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app/Home.py
```

**Important :** les fichiers `.joblib` ont été sérialisés avec les versions de bibliothèques de l'environnement d'entraînement. En cas d'incompatibilité de chargement, reproduire les versions utilisées lors de l'entraînement (notamment scikit-learn et XGBoost).

## Structure

```text
app/                  # Interface Streamlit et fonctions d'inférence
data/raw/             # Données ESOL d'origine
data/processed/       # Données nettoyées et descripteurs
models/               # Scripts d'entraînement et modèles de l'exp. 2
results/optimisation/ # Modèles et prédictions de l'exp. 3
results/representations/ # Comparaison des représentations
results/baseline/     # Résultats de la baseline
knime/                # Emplacement réservé au workflow KNIME
```

## Reproductibilité et limites

- La recherche du nom d'une molécule via **PubChem** nécessite Internet et transmet le SMILES au service externe ; l'inférence peut fonctionner sans cette identification.
- La conversion utilisée est `S (mol/L) = 10**logS`, puis `S (g/L) = S (mol/L) × masse molaire (g/mol)`.
- Les catégories de solubilité sont indicatives, pas réglementaires.
- Les prédictions dépendent des conditions expérimentales (température, pH, état d'ionisation).
- Les scripts d'expérimentation ne constituent pas un pipeline automatisé de production.
- Les modèles `.joblib` doivent provenir d'une source de confiance ; ne jamais charger des fichiers arbitraires.

## Technologies

Python · pandas · NumPy · RDKit · scikit-learn · XGBoost · Plotly · Streamlit · KNIME
