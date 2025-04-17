import streamlit as st

# Créer 3 colonnes (20%, 60%, 20%)
left, center, right = st.columns([0.2, 0.6, 0.2])

with left:
    st.header("Filtres")
    st.selectbox("Option", ["A", "B"])

with center:
    st.header("Contenu Principal")
    st.dataframe({"Data": [1, 2, 3]})

with right:
    st.header("Infos")
    st.metric("Valeur", 42)