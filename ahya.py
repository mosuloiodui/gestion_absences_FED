import streamlit as st
import time

# Définir les couleurs inspirées de Ben10
ben10_colors = {
    'transform_green': '#00FF00',  # Couleur verte pour le flash
}

# Titre de l'application
st.title("Ben10 Transformation")

# Ajouter une description
st.write("Cliquez sur un bouton pour activer la transformation de Ben10.")

# Fonction pour créer l'effet de flash vert
def flash_green_effect():
    # Ajouter du CSS et JavaScript pour créer le flash vert
    st.markdown(f"""
        <style>
            /* Création du flash vert */
            .flash-green {{
                position: fixed;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background-color: {ben10_colors['transform_green']};
                opacity: 0.8;
                z-index: 9999;
                animation: flash 1s forwards;
            }}
            
            /* Animation du flash vert */
            @keyframes flash {{
                0% {{
                    opacity: 0;
                }}
                50% {{
                    opacity: 0.8;
                }}
                100% {{
                    opacity: 0;
                }}
            }}
        </style>
        <div class="flash-green"></div>
    """, unsafe_allow_html=True)

# Boutons pour lancer la transformation
if st.button("Se Transformer en Ben10"):
    flash_green_effect()
    st.write("Vous vous êtes transformé! Le Omnitrix est activé!")
    
elif st.button("Réinitialiser"):
    st.write("Transformation terminée. Vous êtes revenu à la normale.")
