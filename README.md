# Molecular Solubility Lab

### Prédiction de solubilité aqueuse par machine learning et exploration de données moléculaires

**Python · RDKit · KNIME · scikit-learn · XGBoost · Streamlit · PubChem**

[**🚀 Tester l’application en ligne**](https://molecular-solubility-prediction-kcvaadncyhnurxdnayjgpt.streamlit.app/) · [**💻 Explorer le code source**](https://github.com/wiame-bourass/molecular-solubility-prediction)

Molecular Solubility Lab est une application interactive dédiée à la **prédiction de la solubilité aqueuse de composés chimiques à partir de leur structure moléculaire**. Le projet couvre toute la chaîne de traitement, de la préparation des données et de la représentation des molécules jusqu'à l'évaluation des modèles et à leur utilisation dans une interface web.

L'objectif est de construire un pipeline reproductible de **chémo-informatique et de machine learning**, tout en étudiant l'impact des représentations moléculaires et des stratégies de validation sur la qualité des prédictions.

## Démonstration interactive

L’application est accessible directement dans le navigateur, sans installation : **[ouvrir Molecular Solubility Lab](https://molecular-solubility-prediction-kcvaadncyhnurxdnayjgpt.streamlit.app/)**.

Pour essayer la prédiction, ouvrir l’onglet **Prediction**, saisir un SMILES (par exemple `NC(=O)c1cccnc1` pour la niacinamide), sélectionner un modèle et lancer l’analyse.

## Fonctionnalités

| Module | Fonctionnalités |
| --- | --- |
| **Prédiction** | Saisie d'un SMILES, validation et représentation 2D avec RDKit, identification automatique via PubChem, estimation de logS, mol/L et g/L. |
| **Performance des modèles** | Comparaison des protocoles random et scaffold, métriques de régression, graphiques mesuré/prédit et analyse des résidus. |
| **Exploration moléculaire** | Analyse du jeu de données ESOL, distributions, corrélations des descripteurs, visualisation et recherche de molécules. |
| **Accueil** | Présentation du projet, indicateurs clés et accès aux différents modules. |

## Données et préparation

Le projet repose sur **ESOL**, un jeu de données de solubilité aqueuse expérimentale. Les structures sont traitées sous forme de **SMILES**.

- **1 117 molécules uniques** après préparation et dédoublonnage.
- **269 scaffolds moléculaires** identifiés.
- **10 descripteurs physicochimiques RDKit** : masse moléculaire, lipophilie, surface polaire, donneurs et accepteurs de liaisons hydrogène, entre autres.
- **Empreintes Morgan** de 2 048 bits pour représenter les caractéristiques structurales.

La cible est la solubilité mesurée, exprimée en **logS** (logarithme décimal de la solubilité molaire). Les transformations et les représentations sont comparées dans plusieurs expérimentations.

## Démarche de modélisation

**1. Établir des références.** Comparaison de modèles de régression simples et d'ensembles d'arbres sur les premiers descripteurs RDKit.

**2. Étudier la représentation moléculaire.** Évaluation de descripteurs physicochimiques seuls, d'empreintes Morgan seules et de leur combinaison.

**3. Optimiser les modèles.** Recherche d'hyperparamètres et comparaison de plusieurs algorithmes, dont Random Forest, Extra Trees, HistGradientBoosting et XGBoost.

**4. Évaluer la généralisation.** Utilisation de deux protocoles complémentaires :

- **Random split** : mesure la performance sur des molécules tenues à l'écart de l'entraînement.
- **Scaffold split** : sépare les molécules selon leur squelette chimique pour étudier la généralisation à de nouvelles familles structurales.

Les modèles sont sélectionnés sur leur **RMSE en validation croisée**, puis évalués sur leur jeu de test respectif.

## Résultats

### Modèles retenus — expérimentation 3

| Protocole | Modèle | Représentation | RMSE CV ↓ | RMSE test ↓ | MAE test ↓ | R² test ↑ |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| **Scaffold** | **HistGradientBoosting** | 10 RDKit + Morgan r2 (2 048 bits) | 0,772 | **0,907** | **0,663** | **0,769** |
| **Random** | **XGBoost** | 10 RDKit + Morgan r3 (2 048 bits) | 0,629 | **0,581** | **0,428** | **0,919** |

Les métriques sont calculées en **logS**. Les deux protocoles utilisent des jeux de test différents : ils renseignent sur des conditions de généralisation distinctes et ne constituent pas une comparaison directe des algorithmes à jeu de test identique.

Le modèle **HistGradientBoosting évalué par scaffold split** est mis en avant dans l'application ; le modèle **XGBoost évalué par random split** est disponible pour comparaison.

## Architecture du dépôt

```text
app/
  Home.py                         # Page d'accueil
  pages/
    1_Prediction.py               # Inférence et identification moléculaire
    2_Model_Performance.py        # Métriques et analyse des erreurs
    3_Molecular_Explorer.py       # Exploration des données ESOL
  utils/                           # Fonctions moléculaires et chargement des modèles

data/
  raw/                             # Données ESOL sources
  processed/                       # Données nettoyées et enrichies

models/                             # Scripts d'entraînement et comparaison
results/
  baseline/                        # Premières évaluations
  representations/                 # Étude des représentations moléculaires
  optimisation/                    # Modèles optimisés et prédictions test

requirements.txt
packages.txt                       # Dépendances système Linux pour RDKit
README.md
```

## Lancer l'application

Depuis la racine du projet, sous **Windows PowerShell** :

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app/Home.py
```

L'application est ensuite accessible à l'adresse locale indiquée par Streamlit (généralement `localhost:8501`).

La démonstration publique est déployée sur **Streamlit Community Cloud** avec **Python 3.12**. Les dépendances Python sont déclarées dans `requirements.txt` et les bibliothèques système nécessaires à RDKit dans `packages.txt`. La récupération des noms via PubChem nécessite une connexion Internet ; elle est indépendante du calcul de la prédiction.

## Compétences mobilisées

**Chémo-informatique :** SMILES, validation de structures, descripteurs RDKit, empreintes Morgan, scaffolds moléculaires, KNIME.

**Machine learning :** préparation de données, régression, validation croisée, recherche d'hyperparamètres, analyse des erreurs et évaluation de la généralisation.

**Développement :** Python, pandas, NumPy, scikit-learn, XGBoost, Streamlit, Plotly, intégration de PubChem et organisation d'un projet reproductible.

---

**[Accéder à la démonstration](https://molecular-solubility-prediction-kcvaadncyhnurxdnayjgpt.streamlit.app/)** · **[Consulter le dépôt GitHub](https://github.com/wiame-bourass/molecular-solubility-prediction)**
