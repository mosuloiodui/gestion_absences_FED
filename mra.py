import streamlit as st


# Titre de l'application
st.title("🔐 Détection de Ransomwares - Interface Streamlit")

# Choix du dataset
dataset_options = {
    "Original": "original.xlsx",
    "CTGAN": "ctgan.xlsx",
    "CTGAN (100000)": "ctgan_100000.xlsx",
    "TVAE": "tvae.xlsx",
    "Coulangan": "coulangan.xlsx"
}

choix = st.selectbox("📁 Choisissez un dataset à analyser :", list(dataset_options.keys()))
