import re
import json
from urllib.request import Request, urlopen
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

from utils.molecular_features import BASE, EXTRA, descriptor_values, feature_matrix, parse_smiles
from utils.project import model_details, model_options, read_model

st.markdown("""
<style>
.pred-hero {padding:2.1rem 2.3rem;border-radius:20px;background:linear-gradient(115deg,#102d46,#155e70);color:white;margin-bottom:1.5rem;}
.pred-kicker {color:#9ce5dc;font-size:.85rem;font-weight:700;letter-spacing:.12em;text-transform:uppercase;}
.pred-hero h1 {color:white!important;font-size:2.25rem;margin:.35rem 0 .6rem;}
.pred-hero p {color:#e3f1f4;font-size:1.04rem;margin:0;max-width:750px;}
.pred-label {font-weight:700;color:#15334b;font-size:1.2rem;margin:1rem 0 .6rem;}
.pred-note {background:#eef7f7;border-left:4px solid #159a9c;border-radius:9px;padding:1rem 1.2rem;color:#264b59;margin:1rem 0;}
.pred-result {background:linear-gradient(110deg,#e7f7f4,#f3f9fc);border:1px solid #b9e3dc;border-radius:16px;padding:1.2rem 1.5rem;margin:1rem 0;}
.pred-result h2 {font-size:1.55rem;margin:.15rem 0 .5rem;color:#133b4b;}
.pred-result p {margin:0;color:#355462;}
[data-testid="stMetric"] {background:#f7fafc;border:1px solid #e2ebf0;border-radius:14px;padding:1rem 1.2rem;}
</style>
""", unsafe_allow_html=True)


def interpreter_solubilite(g_l):
    if g_l >= 100:
        return "Très élevée", "La solubilité aqueuse estimée est très élevée."
    if g_l >= 10:
        return "Élevée", "La molécule présente une solubilité aqueuse estimée élevée."
    if g_l >= 1:
        return "Limitée", "La solubilité aqueuse estimée est modérée à limitée."
    if g_l >= 0.1:
        return "Faible", "La molécule présente une faible solubilité aqueuse estimée."
    return "Très faible", "La molécule présente une très faible solubilité aqueuse estimée."


def model_label(path):
    p = str(path).replace('\\', '/').lower()
    if 'optimisation/best_scaffold' in p:
        return 'HistGradientBoosting · Scaffold (recommandé)'
    if 'optimisation/best_random' in p:
        return 'XGBoost · Random (comparaison)'
    if 'representations/best_scaffold' in p:
        return 'Random Forest · Scaffold (expérience 2)'
    if 'representations/best_random' in p:
        return 'Random Forest · Random (expérience 2)'
    return path.stem


# Identification via PubChem uniquement : aucune liste locale de noms.
@st.cache_data(show_spinner=False, ttl=86400)
def molecule_name(canonical):
    """Retourne (nom usuel PubChem, nom IUPAC, source) pour un SMILES."""
    from urllib.parse import urlencode

    endpoint = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
    body = urlencode({"smiles": canonical}).encode("utf-8")
    iupac = None
    cid = None
    try:
        request = Request(
            f"{endpoint}/compound/smiles/property/IUPACName/JSON",
            data=body,
            headers={
                "User-Agent": "MolecularSolubilityLab/1.0",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        with urlopen(request, timeout=8) as response:
            data = json.load(response)
        properties = data["PropertyTable"]["Properties"][0]
        iupac = properties.get("IUPACName")
        cid = properties.get("CID")
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        pass

    # PubChem renvoie des synonymes classés par sa propre base.
    # Le premier synonyme lisible est utilisé comme nom courant ; ce n'est
    # pas nécessairement le nom commercial ou le plus connu en français.
    common = None
    if cid is not None:
        try:
            request = Request(
                f"{endpoint}/compound/cid/{int(cid)}/synonyms/JSON",
                headers={"User-Agent": "MolecularSolubilityLab/1.0"},
            )
            with urlopen(request, timeout=8) as response:
                data = json.load(response)
            synonyms = data["InformationList"]["Information"][0].get("Synonym", [])
            for candidate in synonyms:
                name = candidate.strip()
                if (name and len(name) <= 65 and not re.fullmatch(r"[0-9\- ]+", name)
                        and not re.fullmatch(r"[A-Z0-9]{12,}", name)
                        and name.casefold() != str(iupac).casefold()):
                    common = name
                    break
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            pass

    if common:
        return common, iupac, "Nom courant ou synonyme · PubChem"
    if iupac:
        return iupac, iupac, "Nom IUPAC · PubChem"
    return "Nom non disponible", None, "Recherche PubChem indisponible ou molécule non référencée"


def compute_prediction(smiles_value, model_path):
    mol = parse_smiles(smiles_value)
    canonical = Chem.MolToSmiles(mol, canonical=True)
    descriptors = descriptor_values(mol)
    name, iupac_name, name_source = molecule_name(canonical)
    result = {"mol": mol, "canonical": canonical, "name": name, "iupac_name": iupac_name, "name_source": name_source, "descriptors": descriptors, "prediction": None}
    if model_path is None:
        return result
    payload = model_details(read_model(model_path), model_path)
    rep = payload['representation']
    if rep.startswith('D10_Morgan_r'):
        match = re.fullmatch(r'D10_Morgan_r(\d+)_(\d+)', rep)
        if match is None:
            raise ValueError(f'Représentation inconnue : {rep}')
        radius, bits = int(match.group(1)), int(match.group(2))
        payload['morgan_radius'], payload['morgan_bits'] = radius, bits
        expected = BASE + EXTRA + [f'Morgan_{i}' for i in range(bits)]
        if payload.get('features') != expected:
            raise ValueError('Ordre des variables incompatible avec celui du modèle entraîné.')
    x = feature_matrix(
        mol, rep,
        radius=payload.get('morgan_radius', 2),
        bits=payload.get('morgan_bits', 2048),
        base_features=payload.get('base_features'),
        extra_features=payload.get('extra_features'),
    )
    estimator = payload['model']
    if hasattr(estimator, 'n_features_in_') and x.shape[1] != estimator.n_features_in_:
        raise ValueError(f'Nombre de variables incompatible : {x.shape[1]} calculées, {estimator.n_features_in_} attendues.')
    log_s = float(estimator.predict(x)[0])
    concentration = 10 ** log_s
    result['prediction'] = {
        'log_s': log_s,
        'mol_l': concentration,
        'g_l': concentration * Descriptors.MolWt(mol),
        'molar_mass': Descriptors.MolWt(mol),
        'representation': rep,
        'algorithm': type(estimator).__name__,
        'split': payload.get('split', 'Non précisé'),
        'model_label': model_label(model_path),
    }
    return result


st.markdown('''<div class="pred-hero"><div class="pred-kicker">Molecular Solubility Lab · Analyse moléculaire</div>
<h1>Prédiction de solubilité aqueuse</h1>
<p>Entrez une structure moléculaire au format SMILES et obtenez une estimation de sa solubilité dans l’eau, accompagnée de sa représentation 2D.</p></div>''', unsafe_allow_html=True)

st.markdown('<div class="pred-label">01 · Choisir une molécule</div>', unsafe_allow_html=True)
examples = {
    'Caféine': 'Cn1c(=O)c2c(ncn2C)n(C)c1=O',
    'Paracétamol': 'CC(=O)Nc1ccc(O)cc1',
    'Ibuprofène': 'CC(C)Cc1ccc(C(C)C(=O)O)cc1',
    'Aspirine': 'CC(=O)Oc1ccccc1C(=O)O',
}
if 'pred_smiles' not in st.session_state:
    st.session_state['pred_smiles'] = examples['Caféine']

st.caption('Exemples rapides — cliquez pour remplir automatiquement le champ SMILES :')
example_cols = st.columns(4)
for col, (name, value) in zip(example_cols, examples.items()):
    with col:
        if st.button(name, key=f'example_{name}', width="stretch"):
            st.session_state['pred_smiles'] = value
            st.session_state.pop('pred_result', None)
            st.rerun()

st.text_input('Structure moléculaire (SMILES)', key='pred_smiles', placeholder='Ex. : CCO', help='Le SMILES représente une structure chimique sous forme de texte.')

st.markdown('<div class="pred-label">02 · Sélectionner un modèle</div>', unsafe_allow_html=True)
models = model_options()
# Préférer le modèle optimisé scaffold même si la liste fournie par project.py est dans un autre ordre.
models = sorted(models, key=lambda p: (0 if 'optimisation/best_scaffold' in str(p).replace('\\', '/') else 1,
                                      1 if 'optimisation/best_random' in str(p).replace('\\', '/') else 2))
selected = st.selectbox('Modèle de prédiction', models, format_func=model_label, disabled=not models) if models else None
if selected is None:
    st.warning('Aucun modèle de prédiction n’est disponible pour le moment.')
else:
    st.caption('Le modèle scaffold optimisé est sélectionné par défaut lorsqu’il est disponible.')

run = st.button('Analyser la molécule →', type='primary', width="stretch")
if run:
    if not st.session_state['pred_smiles'].strip():
        st.error('Veuillez saisir une structure SMILES.')
        st.session_state.pop('pred_result', None)
    else:
        try:
            with st.spinner('Analyse de la structure moléculaire…'):
                st.session_state['pred_result'] = compute_prediction(st.session_state['pred_smiles'], selected)
                st.session_state['pred_signature'] = (st.session_state['pred_smiles'], str(selected))
        except Exception as exc:
            st.session_state.pop('pred_result', None)
            st.error(f'Analyse impossible : {exc}')

result = st.session_state.get('pred_result')
signature = st.session_state.get('pred_signature')
if result is not None and signature == (st.session_state['pred_smiles'], str(selected)):
    st.divider()
    st.markdown('<div class="pred-label">03 · Résultats de l’analyse</div>', unsafe_allow_html=True)
    st.markdown(f"### {result['name']}")
    st.caption(result['name_source'])
    if result.get('iupac_name') and result['iupac_name'].casefold() != result['name'].casefold():
        st.markdown(f"**Nom IUPAC :** {result['iupac_name']}")
    left, right = st.columns([1, 1.55], gap='large')
    with left:
        st.markdown('**Structure moléculaire 2D**')
        st.image(Draw.MolToImage(result['mol'], size=(420, 300)), width="stretch")
    with right:
        st.markdown('**SMILES canonique**')
        st.code(result['canonical'], language='text', wrap_lines=True)
        st.caption('Structure normalisée par RDKit. Le nom courant et le nom IUPAC proviennent de PubChem, sans liste locale. La prédiction reste possible si PubChem est indisponible.')
        with st.expander('Consulter les descripteurs moléculaires'):
            df = pd.DataFrame([{'Descripteur': k, 'Valeur': round(float(v), 4)} for k, v in result['descriptors'].items()])
            st.dataframe(df, hide_index=True, width="stretch")

    prediction = result['prediction']
    if prediction is not None:
        level, explanation = interpreter_solubilite(prediction['g_l'])
        st.markdown(f'<div class="pred-result"><div style="font-weight:650;color:#317f82">INTERPRÉTATION DE LA SOLUBILITÉ</div><h2>{level}</h2><p>{explanation}</p></div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3, gap='medium')
        c1.metric('Solubilité estimée', f"{prediction['g_l']:.3g} g/L")
        c2.metric('Solubilité (logS)', f"{prediction['log_s']:.3f}")
        c3.metric('Concentration molaire', f"{prediction['mol_l']:.3g} mol/L")
        with st.expander('Comprendre les résultats'):
            st.markdown("""
**Que prédit le modèle ?**

Le modèle estime la **solubilité aqueuse sous forme de logS**. Les valeurs en mol/L et en g/L sont ensuite calculées à partir de cette prédiction : ce ne sont pas trois prédictions différentes.

**1. De logS à mol/L**

La valeur logS est le logarithme décimal de la solubilité molaire. Pour retrouver la solubilité en mol/L, on utilise :
""")
            st.latex(r"S_{\mathrm{mol/L}} = 10^{\mathrm{logS}}")
            st.markdown("**2. De mol/L à g/L**\n\nOn multiplie la solubilité molaire par la masse molaire de la molécule :")
            st.latex(r"S_{\mathrm{g/L}} = S_{\mathrm{mol/L}} \times M_{\mathrm{g/mol}}")
            st.markdown("**Application à cette molécule :**")
            st.latex(
                rf"10^{{{prediction['log_s']:.3f}}} \approx {prediction['mol_l']:.3g}\ \mathrm{{mol/L}}"
            )
            st.latex(
                rf"{prediction['mol_l']:.3g} \times {prediction['molar_mass']:.2f} \approx {prediction['g_l']:.3g}\ \mathrm{{g/L}}"
            )
            st.caption("La catégorie de solubilité est indicative et ne constitue pas une classification réglementaire. Ces résultats restent des estimations, pas des mesures expérimentales.")
        with st.expander('Détails du modèle utilisé'):
            st.write(f"**Modèle :** {prediction['model_label']}")
            st.write(f"**Algorithme :** {prediction['algorithm']}")
            st.write(f"**Représentation :** {prediction['representation']}")
            st.write(f"**Séparation des données :** {prediction['split']}")
        st.markdown('<div class="pred-note"><b>À savoir</b> — Il s’agit d’une prédiction issue d’un modèle de machine learning, et non d’une mesure expérimentale. La solubilité réelle dépend notamment des conditions de mesure (température, pH, ionisation). L’incertitude de cette prédiction individuelle n’est pas quantifiée.</div>', unsafe_allow_html=True)
    else:
        st.info('La structure a été analysée. La prédiction de solubilité sera disponible lorsqu’un modèle sera installé.')
