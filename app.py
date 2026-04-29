import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
import plotly.express as px














# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Prédicteur d’Embauche  Maroc",
    page_icon="",
    layout="wide"
)

st.title("Prédicteur d’Embauche  Maroc")

# ─────────────────────────────────────────────
# LOAD DATASET
# ─────────────────────────────────────────────
DATA_PATH = "data/maroc_tech_students_dataset.csv"

if not os.path.exists(DATA_PATH):
    st.error("❌ Dataset introuvable")
    st.stop()

df = pd.read_csv(DATA_PATH)

# ─────────────────────────────────────────────
# LOAD MODEL + ENCODERS
# ─────────────────────────────────────────────
MODEL_PATH = os.path.join("model", "model.pkl")

if not os.path.exists(MODEL_PATH):
    st.error("❌ model.pkl introuvable")
    st.stop()

@st.cache_resource
def load_all():
    model = pickle.load(open("model/model.pkl", "rb"))
    le_filiere = pickle.load(open("model/le_filiere.pkl", "rb"))
    le_ecole = pickle.load(open("model/le_ecole.pkl", "rb"))
    le_ville = pickle.load(open("model/le_ville.pkl", "rb"))
    return model, le_filiere, le_ecole, le_ville

model, le_filiere, le_ecole, le_ville = load_all()

# ─────────────────────────────────────────────
# OPTIONS (affichage seulement)
# ─────────────────────────────────────────────
FILIERES = df["filiere"].unique().tolist()
VILLES = df["ville"].unique().tolist()
TYPE_ECOLE = df["type_ecole"].unique().tolist()

# ─────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────
st.markdown("### 👇 Remplir les informations du candidat")

col1, col2, col3 = st.columns(3)

with col1:
    filiere = st.selectbox("Filière", FILIERES)

with col2:
    type_ecole = st.selectbox("Type école", TYPE_ECOLE)

with col3:
    ville = st.selectbox("Ville", VILLES)

moyenne = st.slider("Moyenne générale", 0.0, 20.0, 12.0)
stages = st.slider("Stages", 0, 5, 0)

niveau_francais = st.slider("Français", 0, 10, 5)
niveau_anglais = st.slider("Anglais", 0, 10, 5)
niveau_programmation = st.slider("Programmation", 0, 10, 5)
niveau_algorithme = st.slider("Algorithmique", 0, 10, 5)

projets = st.slider("Projets", 0, 10, 1)
certifications = st.slider("Certifications", 0, 10, 0)

participation_hackathon = st.slider("Hackathons", 0, 10, 0)
open_source = st.slider("Open Source Contributions", 0, 10, 0)

soft_skills = st.slider("Soft Skills", 0, 10, 5)

linkedin_profile = st.selectbox("LinkedIn actif ?", ["Non", "Oui"])

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if st.button("🔍 Predict"):

    try:
        input_data = np.array([[
            le_filiere.transform([filiere])[0],
            le_ecole.transform([type_ecole])[0],
            le_ville.transform([ville])[0],
            moyenne,
            stages,
            niveau_francais,
            niveau_anglais,
            niveau_programmation,
            niveau_algorithme,
            projets,
            certifications,
            participation_hackathon,
            open_source,
            soft_skills,
            1 if linkedin_profile == "Oui" else 0
        ]])

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]

        st.markdown("---")
        st.markdown("### 📊 Résultat")

        if prediction == 1:
            st.success(f"✅ RECRUTÉ ({proba[1]*100:.2f}%)")
        else:
            st.error(f"❌ NON RECRUTÉ ({proba[0]*100:.2f}%)")

        st.write("Probabilité recruté :", f"{proba[1]*100:.2f}%")
        st.write("Probabilité non recruté :", f"{proba[0]*100:.2f}%")

        # ─────────────────────────────────────────────
        # IMPORTANCE FEATURES (ICI CORRIGÉ)
        # ─────────────────────────────────────────────

        st.markdown("### 📊 Importance des features")

        features = [
            "filiere", "type_ecole", "ville", "moyenne", "stages",
            "francais", "anglais", "programmation", "algorithme",
            "projets", "certifications", "hackathon",
            "open_source", "soft_skills", "linkedin"
        ]

        importances = model.feature_importances_

        fig = px.bar(
            x=importances,
            y=features,
            orientation='h',
            title="Importance des features",
            labels={"x": "Importance", "y": "Features"}
        )

        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Erreur : {e}")
