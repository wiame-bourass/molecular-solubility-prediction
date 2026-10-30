import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.project import read_csv

st.markdown("""
<style>
.performance-hero {background:linear-gradient(115deg,#11283c,#12556b);border-radius:22px;
 padding:31px 36px;color:white;margin-bottom:22px}
.performance-eyebrow {font-size:.78rem;letter-spacing:.15em;text-transform:uppercase;color:#9ce1dd;font-weight:700}
.performance-hero h1 {color:white!important;font-size:2.15rem!important;margin:12px 0 9px!important}
.performance-hero p {color:#e0eff3;font-size:1.02rem;margin:0}
.perf-card {background:#f3f8fa;border:1px solid #e0edf0;border-radius:15px;padding:18px 22px;min-height:120px}
.perf-card .label {color:#577181;font-size:.84rem;font-weight:650}
.perf-card .value {color:#14374d;font-size:1.65rem;font-weight:750;margin:7px 0}
.perf-card .hint {color:#597280;font-size:.8rem}
.section-note {color:#536b7c;font-size:.93rem}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="performance-hero">
  <div class="performance-eyebrow">Molecular Solubility Lab · Évaluation des modèles</div>
  <h1>Performance des modèles</h1>
  <p>Explorez la précision des prédictions de solubilité, les représentations moléculaires et les erreurs des modèles.</p>
</div>
""", unsafe_allow_html=True)

st.caption("Les métriques portent sur la solubilité logS. Plus MAE et RMSE sont faibles, meilleure est la précision ; plus R² est proche de 1, mieux le modèle explique la variabilité.")

# Ces résultats résument l'expérimentation 3, tels que documentés dans le projet.
# Les jeux de test random et scaffold sont différents : leurs scores ne constituent pas un classement direct.
summary = {
    "Scaffold": {"name": "HistGradientBoosting", "representation": "D10 + Morgan r2 (2 048 bits)",
                 "rmse": 0.90667, "mae": 0.66308, "r2": 0.76940, "cv": 0.77187},
    "Random": {"name": "XGBoost", "representation": "D10 + Morgan r3 (2 048 bits)",
               "rmse": 0.58142, "mae": 0.42840, "r2": 0.91879, "cv": 0.62949},
}

tab_overview, tab_rep, tab_errors, tab_optim = st.tabs([
    "Vue d'ensemble", "Représentations", "Analyse des erreurs", "Optimisation"
])

with tab_overview:
    st.markdown("### Modèles retenus · Expérimentation 3")
    c1, c2 = st.columns(2, gap="medium")
    for col, split in ((c1, "Scaffold"), (c2, "Random")):
        item = summary[split]
        with col:
            st.markdown(f"""<div class="perf-card">
              <div class="label">{split.upper()} SPLIT</div>
              <div class="value">{item['name']}</div>
              <div class="hint">{item['representation']}</div>
            </div>""", unsafe_allow_html=True)
            st.markdown(f"**{summary[split]['name']} · {summary[split]['representation']}**")
            a, b, c = st.columns(3)
            a.metric("RMSE test", f"{item['rmse']:.3f}")
            b.metric("MAE test", f"{item['mae']:.3f}")
            c.metric("R² test", f"{item['r2']:.3f}")
            st.caption(f"RMSE CV : {item['cv']:.3f} · Mesures en logS")

    st.info("**Comment lire cette comparaison ?** Le split random teste des molécules retenues aléatoirement, tandis que le split scaffold réserve des structures centrales distinctes pour le test. Les jeux de test sont différents : il ne faut pas conclure que XGBoost est intrinsèquement meilleur à partir de ces scores.")
    st.markdown("### Parcours expérimental")
    st.markdown("""
    **01 · Baseline** — Descripteurs RDKit et modèles classiques.  
    **02 · Représentations** — Comparaison des descripteurs, empreintes Morgan et combinaisons.  
    **03 · Optimisation** — Recherche d'hyperparamètres et sélection des meilleurs modèles par validation croisée.
    """)

with tab_rep:
    st.markdown("### Effet de la représentation moléculaire")
    df = read_csv("results/representations/comparison.csv")
    required = {"split", "representation", "cv_RMSE_mean"}
    if df is None or not required.issubset(df.columns):
        st.info("Les résultats détaillés des représentations ne sont pas disponibles dans les fichiers du projet.")
    else:
        df = df.copy()
        df["cv_RMSE_mean"] = pd.to_numeric(df["cv_RMSE_mean"], errors="coerce")
        if "cv_RMSE_std" in df:
            df["cv_RMSE_std"] = pd.to_numeric(df["cv_RMSE_std"], errors="coerce")
        df = df.dropna(subset=["cv_RMSE_mean"])
        if df.empty:
            st.info("Aucune métrique RMSE exploitable dans le fichier de comparaison.")
        else:
            fig = px.bar(
                df, x="representation", y="cv_RMSE_mean", color="split", barmode="group",
                error_y="cv_RMSE_std" if "cv_RMSE_std" in df else None,
                labels={"representation": "Représentation", "cv_RMSE_mean": "RMSE moyen (CV)", "split": "Split"},
                color_discrete_map={"random": "#1497a5", "scaffold": "#1a3650"},
            )
            fig.update_layout(template="plotly_white", legend_title_text="Stratégie", margin=dict(t=30, b=15))
            st.plotly_chart(fig, width="stretch")
            st.caption("Comparaison de l'expérimentation 2 : validation croisée uniquement. Les résultats de test ne servent pas à sélectionner la représentation.")
            cols = [x for x in ["split", "representation", "cv_RMSE_mean", "cv_RMSE_std", "cv_MAE_mean", "cv_R2_mean", "selected_by_cv"] if x in df.columns]
            with st.expander("Consulter les résultats détaillés"):
                st.dataframe(df[cols], hide_index=True, width="stretch")

with tab_errors:
    st.markdown("### Mesuré vs prédit · Expérimentation 3")
    st.caption("Prédictions sur les jeux de test des modèles optimisés de l’expérimentation 3. Sélectionnez une stratégie pour examiner ses erreurs.")
    split = st.segmented_control("Stratégie de séparation", ["Scaffold", "Random"], default="Scaffold")
    if split is None:
        split = "Scaffold"
    pred = read_csv(f"results/optimisation/predictions_{split.lower()}.csv")
    if pred is None or not {"y_true", "y_pred"}.issubset(pred.columns):
        st.info(f"Les prédictions individuelles pour le split {split.lower()} ne sont pas disponibles.")
    else:
        pred = pred.copy()
        pred["y_true"] = pd.to_numeric(pred["y_true"], errors="coerce")
        pred["y_pred"] = pd.to_numeric(pred["y_pred"], errors="coerce")
        pred = pred.dropna(subset=["y_true", "y_pred"])
        if pred.empty:
            st.info("Aucune prédiction numérique exploitable.")
        else:
            y = pred["y_true"].to_numpy()
            p = pred["y_pred"].to_numpy()
            pred["Résidu (logS)"] = p - y
            pred["Erreur absolue (logS)"] = np.abs(p - y)
            mae = float(np.mean(np.abs(p - y)))
            rmse = float(np.sqrt(np.mean((p - y) ** 2)))
            denominator = float(np.sum((y - y.mean()) ** 2))
            r2 = 1 - float(np.sum((y - p) ** 2)) / denominator if denominator > 0 else float("nan")
            st.markdown(f"**{summary[split]['name']} · {summary[split]['representation']}** — {len(pred)} molécules évaluées")
            a, b, c = st.columns(3)
            a.metric("RMSE test", f"{rmse:.3f}")
            b.metric("MAE test", f"{mae:.3f}")
            c.metric("R² test", f"{r2:.3f}" if np.isfinite(r2) else "N/D")

            hover = ["canonical_smiles"] if "canonical_smiles" in pred.columns else None
            chart1, chart2 = st.columns(2, gap="medium")
            with chart1:
                fig = px.scatter(pred, x="y_true", y="y_pred", hover_data=hover,
                                 labels={"y_true": "logS mesuré", "y_pred": "logS prédit"},
                                 title="Prédictions vs mesures", opacity=0.72)
                low, high = float(min(y.min(), p.min())), float(max(y.max(), p.max()))
                fig.add_trace(go.Scatter(x=[low, high], y=[low, high], mode="lines",
                                         line=dict(color="#f09b60", dash="dash"), name="Prédiction parfaite"))
                fig.update_traces(marker=dict(size=7), selector=dict(mode="markers"))
                fig.update_layout(template="plotly_white", margin=dict(t=50, b=10))
                st.plotly_chart(fig, width="stretch")
            with chart2:
                fig = px.histogram(pred, x="Résidu (logS)", nbins=28, title="Distribution des erreurs",
                                   labels={"Résidu (logS)": "Prédit − mesuré (logS)"}, color_discrete_sequence=["#1497a5"])
                fig.add_vline(x=0, line_dash="dash", line_color="#f09b60")
                fig.update_layout(template="plotly_white", margin=dict(t=50, b=10))
                st.plotly_chart(fig, width="stretch")
            st.markdown("#### Molécules présentant les plus grandes erreurs")
            cols = [x for x in ["canonical_smiles", "y_true", "y_pred", "Résidu (logS)", "Erreur absolue (logS)"] if x in pred.columns]
            st.dataframe(pred.nlargest(10, "Erreur absolue (logS)")[cols], hide_index=True, width="stretch")
            st.caption("Un résidu positif signifie une surestimation de la solubilité ; un résidu négatif, une sous-estimation. Les jeux de test scaffold et random sont distincts.")

with tab_optim:
    st.markdown("### Recherche et sélection des modèles")
    st.markdown("Les candidats ont été comparés en validation croisée, puis les modèles sélectionnés ont été évalués sur leurs jeux de test respectifs.")
    opt = read_csv("results/optimisation/all_candidates.csv")
    if opt is None or opt.empty:
        st.info("Le tableau détaillé de l'optimisation n'est pas disponible.")
    else:
        opt = opt.copy()
        split_col = next((c for c in ("split", "Split") if c in opt.columns), None)
        if split_col:
            values = sorted(opt[split_col].dropna().astype(str).unique().tolist())
            chosen = st.multiselect("Filtrer par stratégie", values, default=values)
            opt = opt[opt[split_col].astype(str).isin(chosen)]
        st.dataframe(opt, hide_index=True, width="stretch")
        st.download_button("Télécharger les résultats (CSV)", data=opt.to_csv(index=False).encode("utf-8-sig"),
                           file_name="optimisation_modeles.csv", mime="text/csv")
    st.caption("Les scores CV des différentes expérimentations peuvent dépendre de protocoles de validation distincts et ne doivent pas être comparés sans précaution.")

st.markdown("---")
st.caption("Molecular Solubility Lab · Projet expérimental de machine learning sur le jeu de données ESOL. Les scores ne constituent pas une validation pharmaceutique.")
