import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.utils import shuffle

# Configuration Streamlit pour le thème personnalisé
st.set_page_config(
    page_title="Ben10 Prediction",  # Titre de la page
    page_icon="🛸",  # Emoji de l'icône
    layout="wide",  # Disposition de la page
    initial_sidebar_state="expanded",  # Sidebar étendue
)

# Définition des couleurs inspirées de Ben10 (vert et noir)
ben10_colors = {
    "primary": "#00B140",  # Vert Ben10 (Couleur de l'Omnitrix)
    "secondary": "#000000",  # Noir
    "accent": "#111",  # Noir
    "background": "#E5E5E5",  # Fond clair
    "text": "#111",  # Texte principal (noir)
}

# Application du thème via la CSS (injecter du CSS personnalisé dans Streamlit)
st.markdown(f"""
    <style>
        .stApp {{
            background-color: {ben10_colors['background']};
            background-image: url('https://th.bing.com/th/id/OIP.5whEloKo6ntXGO68HhwD_AHaEj?w=280&h=180&c=7&r=0&o=5&pid=1.7');  /* Ajouter une image de fond représentant la montre Omnitrix */
            background-size: cover;
            background-position: center center;
        }}
        .stSidebar {{
            background-color: {ben10_colors['primary']};
        }}
        .stButton, .stSelectbox, .stMultiselect {{
            background-color: {ben10_colors['primary']};
            color: white;
            border-radius: 5px;
        }}
        .stButton:hover {{
            background-color: #006C2F;
        }}
        .stTitle {{
            color: {ben10_colors['accent']};
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.6);
        }}
        .stTextInput input {{
            background-color: {ben10_colors['secondary']};
            color: {ben10_colors['text']};
        }}
        .stMarkdown {{
            color: {ben10_colors['text']};
        }}
    </style>
""", unsafe_allow_html=True)

# Dictionnaire des chemins de fichiers (Dataset)
dataset_paths = {
    "Original": "C:\\Users\\hp\\Downloads\\dataset_reduit_et_normaliser (2).csv",
    "CTGAN": "C:\\Users\\hp\\Downloads\\dataset_synthetique_ctgan (1).xlsx",
    "CTGAN(100K)": "C:\\Users\\hp\\Downloads\\synthetic_data (1) (1).xlsx",
    "TVAE": "C:\\Users\\hp\\Downloads\\dataset_synthetique_tvae.xlsx",
    "CouplaGAN": "C:\\Users\\hp\\Downloads\\dataset_synthetique_copulagan2.xlsx"
}

# Fonction pour lire les fichiers (CSV ou Excel)
def lire_fichier(chemin):
    if chemin.endswith('.csv'):
        return pd.read_csv(chemin)
    elif chemin.endswith('.xlsx'):
        return pd.read_excel(chemin)

# Fonction de chargement du dataset
def load_dataset(choix_utilisateur):
    datasets = []
    if 'All' in choix_utilisateur:
        # Charger tous les datasets
        for dataset in dataset_paths.values():
            data = lire_fichier(dataset)
            datasets.append(data)
    else:
        for nom in choix_utilisateur:
            chemin = dataset_paths.get(nom)
            if chemin:
                data = lire_fichier(chemin)
                datasets.append(data)
    if datasets:
        return pd.concat(datasets, axis=0)
    else:
        return pd.DataFrame()

# Fonction pour initialiser les modèles
def create_model(model_name):
    if model_name == 'XGBoost':
        model = xgb.XGBClassifier()
    elif model_name == 'RF':  # RandomForest
        model = RandomForestClassifier()
    elif model_name == 'SVM':  # Support Vector Machine
        model = SVC(probability=True)
    elif model_name == 'MLP':  # Multi-layer Perceptron
        model = MLPClassifier()
    elif model_name == 'Stack(XGBoost + SVM + MLP)':
        model = VotingClassifier(estimators=[
            ('xgb', xgb.XGBClassifier()), 
            ('svm', SVC(probability=True)),
            ('mlp', MLPClassifier())
        ], voting='soft')
    elif model_name == 'Stack(XGBoost + RF + MLP)':
        model = VotingClassifier(estimators=[
            ('xgb', xgb.XGBClassifier()), 
            ('rf', RandomForestClassifier()), 
            ('mlp', MLPClassifier())
        ], voting='soft')
    return model

# Interface de Streamlit
st.title("Ben10 Prediction - Dataset and Model Selection")
dataset_choice = st.multiselect("Choisir le type de données", options=['All', 'Original', 'CTGAN', 'CTGAN(100K)', 'TVAE', 'CouplaGAN'], default=['Original'])

# Chargement des données
dataset = load_dataset(dataset_choice)

# Afficher les premières lignes du dataset si le bouton est cliqué
if st.button('Afficher Dataset'):
    if not dataset.empty:
        st.write(f"Dataset sélectionné: {dataset_choice}")
        df = shuffle(dataset, random_state=42)
        df['Sample_Type'] = df['Sample_Type'].replace({0: "G", 1: "R"})
        st.write(df.head(10))  # Afficher les 10 premières lignes
        df['Sample_Type'] = df['Sample_Type'].replace({"G": 0, "R": 1})

        # Définir X et y
        X = df.drop('Sample_Type', axis=1)
        y = df['Sample_Type']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Choix des modèles
choix_models = st.multiselect(
    "Choisissez les modèles à utiliser",
    options=["XGBoost", "RF", "SVM", "MLP", "Stack(XGBoost + SVM + MLP)", "Stack(XGBoost + RF + MLP)"],
    default=["XGBoost"]
)

models = [create_model(model_name) for model_name in choix_models]

# Affichage des résultats si un modèle est choisi
if models:
    st.write(f"{len(models)} modèle(s) chargé(s):")
    for model in models:
        st.write(f"- {model.__class__.__name__}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Évaluation du modèle
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)

        st.markdown(f"""
        ✅ **Accuracy**: {acc:.4f}
        ✅ **Recall Score**: {recall:.4f}
        ✅ **Precision Score**: {precision:.4f}
        ✅ **F1 Score**: {f1:.4f}
        ✅ **AUC-ROC**: {auc:.4f}
        """)

        # Option pour afficher d'autres courbes, matrices, etc.
        # Affichage de la matrice de confusion, courbes ROC, etc.
