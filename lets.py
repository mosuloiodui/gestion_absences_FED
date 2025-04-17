import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
import streamlit as st
import pandas as pd
import numpy as np
import openpyxl

from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, roc_curve
)
import xgboost as xgb  # pour shap et XGBoost
from sklearn.model_selection import train_test_split
import streamlit as st
side=st.sidebar

side.title("Choix du dataset")

# Initialisation des états
if 'selected_all' not in st.session_state:
    st.session_state.selected_all = True
if 'selected_options' not in st.session_state:
    st.session_state.selected_options = {
        'Original': False,
        'CTGAN': False,
        'CTGAN_100K': False,
        'TVAE': False,
        'CopulaGAN': False
    }

def update_selections():
    # Si "All" est coché
    if st.session_state.selected_all:
        # Décocher toutes les autres options
        for option in st.session_state.selected_options:
            st.session_state.selected_options[option] = False
    # Si une option est cochée
    else:
        # Vérifier si au moins une option est cochée
        any_selected = any(st.session_state.selected_options.values())
        # Si aucune option n'est cochée, forcer "All" à True
        if not any_selected:
            st.session_state.selected_all = True

def update_all():
    # Si une option est cochée alors que "All" était sélectionné
    if st.session_state.selected_all and any(st.session_state.selected_options.values()):
        st.session_state.selected_all = False

# Checkbox pour "All"
all_checkbox = side.checkbox("All", 
                          value=st.session_state.selected_all,
                          key='all_checkbox',
                          on_change=update_selections)

# Checkboxes pour les autres options
side.write("### Options disponibles:")
for option in st.session_state.selected_options:
    side.checkbox(option,
               value=st.session_state.selected_options[option],
               key=f'opt_{option}',
               on_change=update_all)

# Affichage de la sélection actuelle
side.write("### Sélection actuelle:")
if st.session_state.selected_all:
    side.write("- All (tous les datasets sélectionnés)")
    dataset=['All']
else:
    dataset = [opt for opt, val in st.session_state.selected_options.items() if val]
    if dataset:
        for opt in dataset:
            side.write(f"- {opt}")
    else:
        side.write("- Aucun dataset sélectionné")


# Dictionnaire des chemins
dataset_paths = {
    "Original": "C:\\Users\\hp\\Downloads\\dataset_reduit_et_normaliser (2).csv",
    "CouplaGAN": "C:\\Users\\hp\\Downloads\\dataset_synthetique_copulagan2.csv",
    "CTGAN_100K":"C:\\Users\\hp\\Downloads\\synthetic_data (1) (1).csv",
    "TVAE": "C:\\Users\\hp\\Downloads\\dataset_synthetique_tvae (1).csv",
    "CTGAN": "C:\\Users\\hp\\Downloads\\dataset_synthetique_ctgan.csv"
}
def lire_fichier(chemin):
        return pd.read_csv(chemin,engine="openpyxl")
   
def load_dataset(choix_utilisateur):
    datasets = []
    if 'ALL' in choix_utilisateur:
        # Charger tous les datasets (réels + synthétiques)
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
        from sklearn.ensemble import VotingClassifier

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
    


from sklearn.utils import shuffle
while dataset==[]:
    i=0

datasetpanda=load_dataset(dataset)
df=shuffle(datasetpanda, random_state=42)
da=df
if side.button('afficher ------afficher ') :
      if not df.empty:
        side.title(f"Dataset sélectionné: {df}")
        df['Sample_Type'] = df['Sample_Type'].replace({0: "G", 1: "R"})




        st.write(df.head(11)) 
        df['Sample_Type'] = df['Sample_Type'].replace({"G": 0, "R":1})


        choix_models = side.multiselect(
        "Choisissez les modèles à utiliser",
        options=["XGBoost", "RF", "SVM", "MLP", "Stack(XGBoost + SVM + MLP)", "Stack(XGBoost + RF + MLP)"],
        default=["XGBoost"] # Modèles par défaut
        )
        models = [create_model(model_name) for model_name in choix_models]




    # Display selected models
        if models:
             st.title(f"{len(models)} modèle(s) chargé(s):")
             st.write(da.columns.tolist())
             if st.button("click here to see"):  # Button to show models:
              for model in models:
                side.write(f"- {model.__class__.__name__}")
                st.write(da.columns.tolist())  # List all columns in the DataFrame)
     # Affiche les premières lignes du dataset
                X = da.drop(['Sample_Type'], axis=1)  # Assuming 'Sample_Type' is the target variable
                y = da['Sample_Type']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


            # Train the model
                model.fit(X_train, y_train)

            # Make predictions
                y_pred = model.predict(X_test)
            
            # Evaluate accuracy
                y_true=y_test

                acc = accuracy_score(y_true, y_pred)
                recall = recall_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred)
                f1 = f1_score(y_true, y_pred)
                auc = roc_auc_score(y_true, y_pred)

                st.markdown(f"""
                ✅ **Accuracy**        : {acc:.4f}  
                ✅ **Recall Score**    : {recall:.4f}  
                ✅ **Precision Score** : {precision:.4f}  
                ✅ **F1 Score**        : {f1:.4f}  
                ✅ **AUC-ROC**         : {auc:.4f}  
                """)
            
            

    # Courbe AUC-ROC
    







