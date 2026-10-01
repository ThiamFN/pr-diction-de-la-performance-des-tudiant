"""
Application Streamlit — Prédiction de la performance d'un étudiant

Lancement en local :  streamlit run app.py
"""

import pandas as pd
import joblib as jb
import streamlit as st

# Configuration de la page
st.set_page_config(
    page_title="Prédiction de la performance d'un étudiant",
    page_icon="🎓",
    layout="centered",
)

DESCRIPTION = (
    "Ce modèle de régression linéaire permet de prédire l'indice de performance "
    "(Performance Index) d'un étudiant à partir de ses habitudes d'étude et de vie."
)

# Colonnes attendues, dans l'ordre utilisé à l'entraînement
COLONNES_FEATURES = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]


# Chargement des artefacts (mis en cache : chargés une seule fois)
@st.cache_resource
def load_artifacts():
    model = jb.load("best_model.joblib")               # modèle de régression (Linear Regression)
    scaler = jb.load("scaler.joblib")                   # normaliseur (RobustScaler)
    return model, scaler, activity_encoder


model, scaler = load_artifacts()

ACTIVITY_MAPPING = {"No": 0, "Yes": 1}

# Fonction de prédiction simple
def pred_func(valeurs: dict):
    entree = pd.DataFrame([valeurs], columns=COLONNES_FEATURES)
    entree["Extracurricular Activities"] = entree["Extracurricular Activities"].map(ACTIVITY_MAPPING)
    x_new = scaler.transform(entree)
    y_pred = model.predict(x_new)
    return float(y_pred[0])


# Fonction de prédiction multiple à partir d'un CSV
def pred_func_csv(file):
    df = pd.read_csv(file)
    predictions = []
    for _, row in df[COLONNES_FEATURES].iterrows():
        y_pred = pred_func(row.to_dict())
        predictions.append(y_pred)
    df["Performance Index prédit"] = predictions
    return df


# Interface
st.title("🎓 Prédiction de la performance d'un étudiant")

onglet1, onglet2 = st.tabs(["Prédiction simple", "Prédiction multiple"])

# ----------------------------- Onglet 1 -------------------------------
with onglet1:
    st.subheader("Prédire l'indice de performance")
    st.write(DESCRIPTION)

    with st.form("formulaire_simple"):
        col1, col2 = st.columns(2)

        with col1:
            hours_studied = st.number_input("Heures d'étude (par jour)", min_value=0, max_value=24, value=6, step=1)
            previous_scores = st.number_input("Score précédent", min_value=0, max_value=100, value=70, step=1)
            extracurricular = st.selectbox(
                "Activités extrascolaires",
                options=list(ACTIVITY_MAPPING.keys()),
            )

        with col2:
            sleep_hours = st.number_input("Heures de sommeil", min_value=0, max_value=24, value=7, step=1)
            papers_practiced = st.number_input(
                "Nombre d'exercices/sujets pratiqués", min_value=0, max_value=20, value=5, step=1
            )

        soumettre = st.form_submit_button("Prédire", type="primary")

    if soumettre:
        try:
            valeurs = {
                "Hours Studied": hours_studied,
                "Previous Scores": previous_scores,
                "Extracurricular Activities": extracurricular,
                "Sleep Hours": sleep_hours,
                "Sample Question Papers Practiced": papers_practiced,
            }
            resultat = pred_func(valeurs)
            st.success(f"**Indice de performance prédit : {resultat:.2f}**")
        except Exception as e:
            st.error(f"Erreur lors de la prédiction : {e}")

# ----------------------------- Onglet 2 -------------------------------
with onglet2:
    st.subheader("Prédire la performance de plusieurs étudiants à partir d'un fichier CSV")
    st.write(DESCRIPTION)
    st.caption(
        "Le fichier CSV doit contenir, avec ces noms de colonnes exacts : "
        + ", ".join(COLONNES_FEATURES)
    )

    fichier = st.file_uploader("Importer un fichier CSV", type=["csv"])

    if fichier is not None:
        try:
            with st.spinner("Prédictions en cours…"):
                df_resultat = pred_func_csv(fichier)

            st.success(f"{len(df_resultat)} prédiction(s) effectuée(s).")
            st.dataframe(df_resultat, use_container_width=True)

            st.download_button(
                label="⬇️ Télécharger le fichier CSV",
                data=df_resultat.to_csv(index=False).encode("utf-8"),
                file_name="predictions.csv",
                mime="text/csv",
                type="primary",
            )
        except Exception as e:
            st.error(f"Erreur lors du traitement du fichier : {e}")
