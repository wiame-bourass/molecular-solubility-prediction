import streamlit as st

st.set_page_config(
    page_title="Molecular Solubility Lab",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.block-container {max-width: 1200px; padding-top: 2.5rem; padding-bottom: 3rem;}
.hero {background: linear-gradient(115deg,#102c46 0%,#145a70 100%); border-radius: 22px; padding: 44px 46px; color: #fff; margin-bottom: 26px;}
.hero .eyebrow {font-size: .85rem; font-weight: 700; letter-spacing: .15em; color: #8ee6d7; margin-bottom: 12px;}
.hero h1 {font-size: clamp(2.1rem,4vw,3.3rem); color: #fff; margin: 0 0 12px; line-height: 1.15;}
.hero p {font-size: 1.08rem; color: #e0edf3; max-width: 730px; margin: 0; line-height: 1.65;}
.stat-card {background: #f5f9fb; border: 1px solid #e4eef2; border-radius: 16px; padding: 23px 25px; margin-bottom: 22px;}
.stat-value {font-size: 2.15rem; font-weight: 800; color: #113b53; line-height: 1.2;}
.stat-label {color: #567081; font-size: .92rem; margin-top: 7px;}
.section-title {font-size: 1.65rem; font-weight: 750; color: #16354b; margin: 18px 0 8px;}
.section-subtitle {color: #617586; margin-bottom: 22px;}
.nav-card {border: 1px solid #e2ebef; background: #fff; border-radius: 17px; padding: 23px 22px; min-height: 175px; box-shadow: 0 5px 20px rgba(20,65,85,.045);}
.nav-icon {font-size: 1.75rem; margin-bottom: 13px;}
.nav-title {font-size: 1.14rem; font-weight: 750; color: #173b50; margin-bottom: 9px;}
.nav-desc {font-size: .94rem; color: #5b7080; line-height: 1.55;}
div.stButton > button {width: 100%; border-radius: 11px; border: 1px solid #0d827d; color: #086a69; font-weight: 700; min-height: 44px;}
div.stButton > button:hover {background: #e9f8f6; border-color: #08716d; color: #064f4c;}
.footnote {font-size: .88rem; color: #647887; background: #f5f8fa; padding: 16px 20px; border-left: 3px solid #56b5aa; border-radius: 8px; margin-top: 32px;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <div class="eyebrow">CHEMOINFORMATIQUE · MACHINE LEARNING</div>
  <h1>🧪 Molecular Solubility Lab</h1>
  <p>Explorez les propriétés des molécules et estimez leur solubilité aqueuse à partir de leur structure chimique, grâce à des modèles de machine learning.</p>
</div>
""", unsafe_allow_html=True)

stats = [
    ("1 117", "Molécules uniques"),
    ("269", "Scaffolds moléculaires"),
    ("10", "Descripteurs RDKit"),
]
cols = st.columns(3, gap="medium")
for col, (value, label) in zip(cols, stats):
    with col:
        st.markdown(f'<div class="stat-card"><div class="stat-value">{value}</div><div class="stat-label">{label}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="section-title">Explorez l’application</div>', unsafe_allow_html=True)
st.markdown('<div class="section-subtitle">Choisissez une fonctionnalité pour commencer.</div>', unsafe_allow_html=True)

pages = [
    ("🧪", "Prédiction moléculaire", "Saisissez un SMILES, visualisez la structure 2D et obtenez une estimation de solubilité.", "Lancer une prédiction →", "pages/1_Prediction.py"),
    ("📊", "Performance des modèles", "Découvrez les résultats des modèles et comparez les stratégies d’évaluation.", "Voir les performances →", "pages/2_Model_Performance.py"),
    ("🔬", "Exploration moléculaire", "Explorez le jeu de données ESOL et les propriétés de ses molécules.", "Explorer les molécules →", "pages/3_Molecular_Explorer.py"),
]
cols = st.columns(3, gap="medium")
for col, (icon, title, description, button_text, page_path) in zip(cols, pages):
    with col:
        st.markdown(f'<div class="nav-card"><div class="nav-icon">{icon}</div><div class="nav-title">{title}</div><div class="nav-desc">{description}</div></div>', unsafe_allow_html=True)
        if st.button(button_text, key=page_path):
            st.switch_page(page_path)

#st.markdown('<div class="footnote">ℹ️ Les prédictions sont fournies à titre expérimental et ne remplacent pas une mesure de laboratoire ni une validation pharmaceutique.</div>', unsafe_allow_html=True)
