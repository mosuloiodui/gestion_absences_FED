import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

# ---- Chargement des datasets ----
dataset_paths = {
    "Original": "C:\\Users\\hp\\Downloads\\dataset_reduit_et_normaliser (2).csv",
    "CTGAN": "C:\\Users\\hp\\Downloads\\dataset_synthetique_ctgan (1).xlsx",
    "CTGAN(100K)": "C:\\Users\\hp\\Downloads\\synthetic_data (1) (1).xlsx",
    "TVAE": "C:\\Users\\hp\\Downloads\\dataset_synthetique_tvae.xlsx",
    "CouplaGAN": "C:\\Users\\hp\\Downloads\\dataset_synthetique_copulagan2.xlsx"
}

def lire_fichier(chemin):
    if chemin.endswith('.csv'):
        return pd.read_csv(chemin)
    elif chemin.endswith('.xlsx'):
        return pd.read_excel(chemin)

def load_dataset(choix_utilisateur):
    datasets = []
    if 'All' in choix_utilisateur:
        for dataset in dataset_paths.values():
            data = lire_fichier(dataset)
            datasets.append(data)
    else:
        for nom in choix_utilisateur:
            chemin = dataset_paths.get(nom)
            if chemin:
                data = lire_fichier(chemin)
                datasets.append(data)
    return pd.concat(datasets, axis=0) if datasets else pd.DataFrame()

# ---- Initialisation des modèles ----
def create_model(model_name):
    if model_name == 'XGBoost':
        return xgb.XGBClassifier()
    elif model_name == 'RF':
        return RandomForestClassifier()
    elif model_name == 'SVM':
        return SVC(probability=True)
    elif model_name == 'MLP':
        return MLPClassifier()
    elif model_name == 'Stack(XGBoost + SVM + MLP)':
        return VotingClassifier(estimators=[
            ('xgb', xgb.XGBClassifier()),
            ('svm', SVC(probability=True)),
            ('mlp', MLPClassifier())
        ], voting='soft')
    elif model_name == 'Stack(XGBoost + RF + MLP)':
        return VotingClassifier(estimators=[
            ('xgb', xgb.XGBClassifier()),
            ('rf', RandomForestClassifier()),
            ('mlp', MLPClassifier())
        ], voting='soft')

# ---- Interface Streamlit ----
st.title("🧠 Classification des données synthétiques et réelles")

dataset_choice = st.selectbox("🎯 Choisissez le ou les datasets", options=['All'] + list(dataset_paths.keys()), index=0)


if 'datset' not in st.session_state:
    st.session_state['dataset']=False
dataset = load_dataset(dataset_choice)

if st.button('📊 Afficher et préparer le dataset'):
    st.session_state['dataset']=True
    if not dataset.empty:
        df = shuffle(dataset, random_state=42)
        st.success("✅ Dataset chargé et mélangé !")
        st.write(df.head(10))
    
       
choix_models = st.selectbox(
"🛠️ Choisissez les modèles à entraîner",
options=["XGBoost", "RF", "SVM", "MLP", "Stack(XGBoost + SVM + MLP)", "Stack(XGBoost + RF + MLP)"],
index=0
)
if 'model' not in st.session_state:
    st.session_state['model']=False
if st.button('📊 Afficher le modèle'):
        st.session_state['model']=True
        df = shuffle(dataset, random_state=42)
        df['Sample_Type'] = df['Sample_Type'].replace({0: "G", 1: "R"})
        df['Sample_Type'] = df['Sample_Type'].replace({"G": 0, "R": 1})

        # Split des données
        X = df.drop('Sample_Type', axis=1)
        y = df['Sample_Type']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
       
        models = [create_model(name) for name in choix_models]

        for name in choix_models:
            model = create_model(name)
            with st.spinner(f"⏳ Entraînement du modèle {name} en cours..."):
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            st.markdown(f"---\n### 🔍 Résultats pour le modèle : `{name}`")

            acc = accuracy_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

            st.markdown(f"""
            ✅ **Accuracy**        : `{acc:.4f}`  
            ✅ **Recall Score**    : `{recall:.4f}`  
            ✅ **Precision Score** : `{precision:.4f}`  
            ✅ **F1 Score**        : `{f1:.4f}`  
            ✅ **AUC-ROC**         : `{auc}`
            """)

            # Enregistrement pour tableau comparatif
            from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, classification_report
)


            # Rapport détaillé
            with st.expander("📋 Rapport détaillé"):
                rapport = classification_report(y_test, y_pred, output_dict=False)
                st.text(rapport)

            # Matrice de confusion
            cm = confusion_matrix(y_test, y_pred)
            fig_cm, ax_cm = plt.subplots()
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["G", "R"])
            disp.plot(ax=ax_cm, cmap="Blues")
            ax_cm.set_title(f"Matrice de confusion - {name}")
            st.pyplot(fig_cm)

            # Courbe ROC
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig_roc, ax_roc = plt.subplots()
                ax_roc.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.2f})")
                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set_xlabel('Taux de faux positifs')
                ax_roc.set_ylabel('Taux de vrais positifs')
                ax_roc.set_title(f"Courbe ROC - {name}")
                ax_roc.legend(loc="lower right")
                st.pyplot(fig_roc)
            if 'app' not in st.session_state:
                  st.session_state['app']=False
            # Courbe d'apprentissage
            if True:
                with st.spinner("⏳ Chargement de la courbe d'apprentissage..."):
                    train_sizes, train_scores, test_scores = learning_curve(
                        model, X, y, cv=5, scoring='accuracy',
                        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
                    )
                    train_mean = np.mean(train_scores, axis=1)
                    test_mean = np.mean(test_scores, axis=1)

                    fig_lc, ax_lc = plt.subplots()
                    ax_lc.plot(train_sizes, train_mean, 'o-', label="Train")
                    ax_lc.plot(train_sizes, test_mean, 'o-', label="Validation")
                    ax_lc.set_xlabel("Taille de l’échantillon")
                    ax_lc.set_ylabel("Précision")
                    ax_lc.set_title(f"Courbe d’apprentissage - {name}")
                    ax_lc.legend()
                    st.pyplot(fig_lc)
                    print(st.session_state)
        