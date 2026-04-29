import streamlit as st
import numpy as np
import pickle
import os

# ─── PAGE CONFIG ─────────────────────────────────────────────
st.set_page_config(
    page_title="Job Hiring Predictor AI",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── SESSION STATE INIT ──────────────────────────────────────
defaults = {
    "branch": "CSE",
    "college_tier": 1,
    "cgpa": 7.0,
    "backlogs": 0,
    "aptitude_score": 50,
    "coding_skills": 5,
    "dsa_score": 50,
    "ml_knowledge": 5,
    "system_design": 5,
    "communication_skills": 5,
    "internships": 0,
    "projects_count": 1,
    "certifications": 0,
    "hackathons": 0,
    "open_source": 0,
    "extracurriculars": 0,
    "show_result": False,
    "prediction": None,
    "probability": None,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─── LOAD MODEL ──────────────────────────────────────────────
if not os.path.exists("model.pkl"):
    st.error("model.pkl introuvable")
    st.stop()

@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()


# ─── UI HEADER ───────────────────────────────────────────────
st.title("🎯 Job Hiring Predictor AI")


# ─── INPUTS ──────────────────────────────────────────────────
BRANCHES = ["CSE", "IT", "ECE", "Autre"]
TIERS = [1, 2, 3]

col1, col2, col3 = st.columns(3)

with col1:
    st.session_state.branch = st.selectbox(
        "Filière",
        BRANCHES,
        index=BRANCHES.index(st.session_state.branch)
    )

with col2:
    st.session_state.college_tier = st.selectbox(
        "Classement établissement",
        TIERS,
        index=TIERS.index(st.session_state.college_tier),
        format_func=lambda x: f"Tier {x}"
    )

with col3:
    st.session_state.cgpa = st.slider("CGPA", 0.0, 10.0, st.session_state.cgpa)


st.session_state.backlogs = st.slider("Backlogs", 0, 10, st.session_state.backlogs)
st.session_state.aptitude_score = st.slider("Aptitude", 0, 100, st.session_state.aptitude_score)

st.session_state.coding_skills = st.slider("Coding", 0, 10, st.session_state.coding_skills)
st.session_state.dsa_score = st.slider("DSA", 0, 100, st.session_state.dsa_score)
st.session_state.ml_knowledge = st.slider("ML", 0, 10, st.session_state.ml_knowledge)

st.session_state.system_design = st.slider("System Design", 0, 10, st.session_state.system_design)
st.session_state.communication_skills = st.slider("Communication", 0, 10, st.session_state.communication_skills)

st.session_state.internships = st.slider("Internships", 0, 5, st.session_state.internships)
st.session_state.projects_count = st.slider("Projects", 0, 10, st.session_state.projects_count)
st.session_state.certifications = st.slider("Certifications", 0, 10, st.session_state.certifications)
st.session_state.hackathons = st.slider("Hackathons", 0, 10, st.session_state.hackathons)
st.session_state.open_source = st.slider("Open Source", 0, 20, st.session_state.open_source)
st.session_state.extracurriculars = st.slider("Extra", 0, 10, st.session_state.extracurriculars)


# ─── PREDICTION ──────────────────────────────────────────────
if st.button("🔍 Predict"):

    branch_map = {"CSE": 0, "IT": 1, "ECE": 2, "Autre": 3}

    input_data = np.array([[

        branch_map[st.session_state.branch],
        st.session_state.college_tier,
        st.session_state.cgpa,
        st.session_state.backlogs,
        st.session_state.coding_skills,
        st.session_state.dsa_score,
        st.session_state.aptitude_score,
        st.session_state.communication_skills,
        st.session_state.ml_knowledge,
        st.session_state.system_design,
        st.session_state.internships,
        st.session_state.projects_count,
        st.session_state.certifications,
        st.session_state.hackathons,
        st.session_state.open_source,
        st.session_state.extracurriculars,

    ]])

    st.session_state.prediction = int(model.predict(input_data)[0])
    st.session_state.probability = model.predict_proba(input_data)[0]
    st.session_state.show_result = True


# ─── RESULT ──────────────────────────────────────────────────
if st.session_state.show_result:

    pred = st.session_state.prediction
    prob = st.session_state.probability

    if pred == 1:
        st.success(f"✅ HIRED ({prob[1]*100:.2f}%)")
    else:
        st.error(f"❌ NOT HIRED ({prob[0]*100:.2f}%)")