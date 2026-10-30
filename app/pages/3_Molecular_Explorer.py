import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors
from utils.project import read_csv

st.markdown("""
<style>
.explorer-hero {background:linear-gradient(115deg,#112b43,#126e78);color:white;
 padding:29px 34px;border-radius:20px;margin-bottom:20px}
.explorer-eyebrow {font-size:12px;letter-spacing:2px;color:#a9e9e4;font-weight:700}
.explorer-hero h1 {color:white!important;font-size:34px;margin:8px 0 8px}
.explorer-hero p {color:#e1f2f2;font-size:15px;margin:0}
.explorer-note {background:#edf7f7;border-left:4px solid #148d8b;border-radius:9px;
 padding:12px 16px;color:#173c48;margin:12px 0 18px}
</style>
""", unsafe_allow_html=True)

st.markdown("""<div class="explorer-hero"><div class="explorer-eyebrow">MOLECULAR SOLUBILITY LAB · DATA EXPLORER</div>
<h1>Exploration moléculaire</h1><p>Découvrez les structures chimiques, les propriétés RDKit et les mesures expérimentales du jeu de données ESOL.</p></div>""", unsafe_allow_html=True)

raw = read_csv('data/processed/esol_processed_v2.csv')
if raw is None or raw.empty:
    st.error("Les données d'exploration ne sont pas disponibles. Contactez l'administrateur de l'application.")
    st.stop()

df = raw.copy()
target = 'measured log solubility in mols per litre'
smiles_col = next((c for c in ['smiles', 'canonical_smiles', 'SMILES'] if c in df.columns), None)
features = [c for c in ['AMW','SlogP','TPSA','NumHBD','NumHBA','NumRotatableBonds',
                       'NumAromaticRings','FractionCSP3','NumHeavyAtoms','NumRings'] if c in df.columns]
if target in df.columns:
    df[target] = pd.to_numeric(df[target], errors='coerce')
for c in features:
    df[c] = pd.to_numeric(df[c], errors='coerce')

t1, t2, t3 = st.tabs(['📊 Vue d’ensemble', '🔎 Propriétés moléculaires', '🧬 Explorer les molécules'])

with t1:
    n1, n2, n3 = st.columns(3)
    n1.metric('Molécules du dataset', f'{len(df):,}'.replace(',', ' '))
    if target in df.columns and df[target].notna().any():
        n2.metric('Solubilité moyenne (logS)', f'{df[target].mean():.2f}')
        n3.metric('Intervalle logS', f'{df[target].min():.1f} à {df[target].max():.1f}')
    else:
        n2.metric('Descripteurs disponibles', len(features))
        n3.metric('Colonnes disponibles', len(df.columns))
    st.markdown('<div class="explorer-note">Les valeurs de solubilité présentées ici sont des <b>mesures expérimentales du dataset ESOL</b>, pas des prédictions des modèles.</div>', unsafe_allow_html=True)
    if target in df.columns and df[target].notna().any():
        st.subheader('Distribution de la solubilité expérimentale')
        fig = px.histogram(df.dropna(subset=[target]), x=target, nbins=38,
                           labels={target:'Solubilité expérimentale (logS)'},
                           color_discrete_sequence=['#168e94'])
        fig.update_layout(template='plotly_white', bargap=0.07, yaxis_title='Nombre de molécules',
                          margin=dict(l=15,r=15,t=20,b=20))
        st.plotly_chart(fig, width='stretch')
        st.caption('Plus logS est élevé, plus la solubilité molaire mesurée est élevée.')
    st.subheader('Aperçu des données')
    shown = [c for c in ([smiles_col, target] + features[:4]) if c and c in df.columns]
    st.dataframe(df[shown].head(15), hide_index=True, width='stretch')

with t2:
    st.subheader('Relation entre propriétés et solubilité')
    if not features or target not in df.columns:
        st.info('Les descripteurs ou les mesures de solubilité nécessaires ne sont pas disponibles.')
    else:
        descriptions = {
            'AMW':'Masse molaire', 'SlogP':'Lipophilie (SlogP)', 'TPSA':'Surface polaire (TPSA)',
            'NumHBD':'Donneurs de liaison hydrogène', 'NumHBA':'Accepteurs de liaison hydrogène',
            'NumRotatableBonds':'Liaisons rotatives', 'NumAromaticRings':'Cycles aromatiques',
            'FractionCSP3':'Fraction de carbones sp³', 'NumHeavyAtoms':'Atomes lourds',
            'NumRings':'Nombre de cycles'}
        left, right = st.columns([1, 2])
        with left:
            selected = st.selectbox('Propriété à analyser', features,
                                    index=features.index('SlogP') if 'SlogP' in features else 0,
                                    format_func=lambda x: descriptions.get(x, x))
            st.caption(f'Descripteur RDKit : {selected}')
        with right:
            st.markdown('**Chaque point représente une molécule du dataset.** Survolez-le pour consulter ses valeurs.')
        plot_df = df.dropna(subset=[selected,target]).copy()
        hover = [smiles_col] if smiles_col else None
        fig = px.scatter(plot_df, x=selected, y=target, hover_data=hover, opacity=.65,
                         labels={selected:descriptions.get(selected,selected), target:'Solubilité mesurée (logS)'},
                         color_discrete_sequence=['#148d8b'])
        fig.update_traces(marker=dict(size=7))
        fig.update_layout(template='plotly_white', margin=dict(l=15,r=15,t=20,b=20))
        st.plotly_chart(fig, width='stretch')
        st.subheader('Corrélations avec la solubilité')
        correlations = df[features + [target]].corr(numeric_only=True)[target].drop(target).dropna().sort_values()
        if not correlations.empty:
            corr_df = correlations.reset_index()
            corr_df.columns = ['Descripteur', 'Corrélation de Pearson']
            fig_corr = px.bar(corr_df, x='Corrélation de Pearson', y='Descripteur', orientation='h',
                              color='Corrélation de Pearson', color_continuous_scale='BrBG',
                              range_color=[-1,1])
            fig_corr.update_layout(template='plotly_white', height=max(330, len(corr_df)*34),
                                   margin=dict(l=15,r=15,t=15,b=15), coloraxis_showscale=False)
            st.plotly_chart(fig_corr, width='stretch')
            st.caption('La corrélation décrit une association statistique ; elle ne démontre pas un lien de causalité.')

with t3:
    st.subheader('Rechercher et inspecter une molécule')
    if not smiles_col:
        st.warning('Aucune colonne SMILES n’est disponible dans le dataset.')
    else:
        search = st.text_input('Rechercher un SMILES dans le dataset',
                               placeholder='Ex. : c1ccccc1',
                               help='Recherche textuelle dans les SMILES du dataset.')
        filtered = df.copy()
        if search.strip():
            filtered = filtered[filtered[smiles_col].astype(str).str.contains(search.strip(),
                                                                           case=False, regex=False, na=False)]
        if target in filtered.columns and filtered[target].notna().any():
            values = filtered[target].dropna()
            min_v, max_v = float(values.min()), float(values.max())
            if min_v < max_v:
                bounds = st.slider('Filtrer par solubilité expérimentale (logS)',
                                   min_value=float(df[target].min()), max_value=float(df[target].max()),
                                   value=(float(df[target].min()), float(df[target].max())), step=0.1)
                filtered = filtered[filtered[target].between(bounds[0], bounds[1])]
        st.caption(f'{len(filtered)} molécule(s) correspondent aux filtres.')
        cols_to_show = [c for c in [smiles_col, target, 'AMW', 'SlogP', 'TPSA'] if c in filtered.columns]
        st.dataframe(filtered[cols_to_show].head(100), hide_index=True, width='stretch')
        if not filtered.empty:
            options = list(filtered.index)
            selected_idx = st.selectbox('Sélectionner une molécule à visualiser', options,
                                        format_func=lambda i: f'Molécule {i + 1} · {str(df.at[i,smiles_col])[:75]}')
            row = df.loc[selected_idx]
            smiles = str(row[smiles_col])
            mol = Chem.MolFromSmiles(smiles)
            left, right = st.columns([1, 1.5], gap='large')
            with left:
                st.markdown('**Structure 2D**')
                if mol:
                    st.image(Draw.MolToImage(mol, size=(440,310)), width='stretch')
                else:
                    st.warning('La structure ne peut pas être représentée par RDKit.')
            with right:
                st.markdown('**Structure SMILES**')
                st.code(smiles, language='text')
                if target in df.columns and pd.notna(row.get(target)):
                    log_s = float(row[target])
                    st.metric('Solubilité expérimentale (logS)', f'{log_s:.3f}')
                    if mol:
                        mass = Descriptors.MolWt(mol)
                        st.metric('Solubilité expérimentale estimée en g/L', f'{10**log_s * mass:.3g} g/L')
                        st.caption('Conversion de la mesure logS en g/L à partir de la masse molaire RDKit.')
                with st.expander('Afficher toutes les propriétés de cette molécule'):
                    details = pd.DataFrame({'Propriété': row.index.astype(str), 'Valeur': row.astype(str).values})
                    st.dataframe(details, hide_index=True, width='stretch')
        else:
            st.info('Aucune molécule ne correspond à cette recherche. Modifiez les filtres.')

st.divider()
st.caption('Molecular Solubility Lab · ESOL · Données expérimentales à visée exploratoire.')
