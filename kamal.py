import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, classification_report
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
from sklearn.utils import shuffle
import openpyxl
from sklearn.manifold import TSNE

dataset_paths = {
    "Original": "https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier.csv",
    "TVAE": "https://github.com/mosuloiodui/gestion_absences_FED/raw/main/dataset_synthetique_tvae%20(1).csv",
    "CTGAN_100K_1":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_1.csv",
    "CTGAN_100K_2":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_2.csv",
    "CTGAN_100K_3":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_3.csv",
    "CTGAN_100K_4":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_4.csv",
    "CTGAN_100K_5":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_5.csv",
    "CTGAN_100K_6":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_6.csv",
    "CTGAN_100K_7":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_7.csv",
    "CTGAN_100K_8":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_8.csv",
    "CTGAN_100K_9":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_9.csv",
    "CTGAN_100K_10":"https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier_part_10.csv",
    
}
@st.cache_data
def read_tvae():
    return pd.read_csv('https://github.com/mosuloiodui/gestion_absences_FED/raw/main/dataset_synthetique_tvae%20(1).csv')
@st.cache_data
def read_ctgan():
    return pd.concat([pd.read_csv(dataset_paths[f'CTGAN_100K_{i+1}']) for i in range(10)], ignore_index=True)
@st.cache_data
def read_mixte():
    return pd.concat([read_tvae(), read_original()], ignore_index=True).sample(frac=1)

@st.cache_data
def read_original():
    return  pd.read_csv('https://github.com/mosuloiodui/gestion_absences_FED/raw/main/fichier.csv')
df_tvae=read_tvae()
df_original=read_original()
df_ctgan_100K=read_ctgan()
df_mixte=read_mixte()
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
@st.cache_data
def modx(df_original):
    df=df_otiginal
    X = df.drop('Sample_Type', axis=1)
    y = df['Sample_Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model=xgb.XGBoost()
    st.session_state.go_model=True

    str=""
    j=0
    for i in range(len(choix_models)):
        j+=1
        str=str+choix_models[i]
        if j<len(choix_models):
              str=str+" et "
    st.sidebar.write("Chargement :",str)

    if True:
        with st.spinner(f"⏳ Entraînement du modèle {name} en cours..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

        st.title(f"---\n### 🔍 Résultats pour le modèle : `{name}`")
        st.markdown(f"""
        ✅ **Accuracy**        : `{acc:.4f}`  
        ✅ **Recall Score**    : `{recall:.4f}`  
        ✅ **Precision Score** : `{precision:.4f}`  
        ✅ **F1 Score**        : `{f1:.4f}`  
        ✅ **AUC-ROC**         : `{auc}`
        """)

        with st.expander("📋 Rapport détaillé"):
            rapport = classification_report(y_test, y_pred,target_names=["G", "R"], output_dict=False)
            st.text(rapport)
            st.markdown("### 🧾 Matrice de confusion")

            with st.spinner(f"⏳ Entraînement du modèle  en cours..."):
                 cm = confusion_matrix(y_test, y_pred)
                 fig_cm, ax_cm = plt.subplots()
                 disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["G", "R"])
                 disp.plot(ax=ax_cm, cmap="Blues")
                 st.pyplot(fig_cm)
        
    
@st.cache_data
def modx(df_tave):
    df=df_tvae
    X = df.drop('Sample_Type', axis=1)
    y = df['Sample_Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model=xgb.XGBClassifier()
    st.session_state.go_model=True

    str=""
    j=0
    for i in range(len(choix_models)):
        j+=1
        str=str+choix_models[i]
        if j<len(choix_models):
              str=str+" et "
    st.sidebar.write("Chargement :",str)
    if True:
        with st.spinner(f"⏳ Entraînement du modèle  en cours..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

        st.title(f"---\n### 🔍 Résultats pour le modèle")
        st.markdown(f"""
        ✅ **Accuracy**        : `{acc:.4f}`  
        ✅ **Recall Score**    : `{recall:.4f}`  
        ✅ **Precision Score** : `{precision:.4f}`  
        ✅ **F1 Score**        : `{f1:.4f}`  
        ✅ **AUC-ROC**         : `{auc}`
        """)

        with st.expander("📋 Rapport détaillé"):
            rapport = classification_report(y_test, y_pred,target_names=["G", "R"], output_dict=False)
            st.text(rapport)
            st.markdown("### 🧾 Matrice de confusion")

            with st.spinner(f"⏳ Entraînement du modèle en cours..."):
                 cm = confusion_matrix(y_test, y_pred)
                 fig_cm, ax_cm = plt.subplots()
                 disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["G", "R"])
                 disp.plot(ax=ax_cm, cmap="Blues")
                 st.pyplot(fig_cm)
       
@st.cache_data
def modx(df_mixte):
    df=df_mixte
    X = df.drop('Sample_Type', axis=1)
    y = df['Sample_Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model=xgb.XGBClassifier()
    st.session_state.go_model=True

    str=""
    j=0
    for i in range(len(choix_models)):
        j+=1
        str=str+choix_models[i]
        if j<len(choix_models):
              str=str+" et "
    st.sidebar.write("Chargement :",str)
    if True:
        with st.spinner(f"⏳ Entraînement du modèle en cours..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

        st.title(f"---\n### 🔍 Résultats pour le modèle")
        st.markdown(f"""
        ✅ **Accuracy**        : `{acc:.4f}`  
        ✅ **Recall Score**    : `{recall:.4f}`  
        ✅ **Precision Score** : `{precision:.4f}`  
        ✅ **F1 Score**        : `{f1:.4f}`  
        ✅ **AUC-ROC**         : `{auc}`
        """)

       
           
name='XGBoost'       
@st.cache_data
def modx(df_ctgan_100K):
    df=df_ctgan_100K
    X = df.drop('Sample_Type', axis=1)
    y = df['Sample_Type']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model=xgb.XGBClassifier()
    st.session_state.go_model=True

    str=""
    j=0
    for i in range(len(choix_models)):
        j+=1
        str=str+choix_models[i]
        if j<len(choix_models):
              str=str+" et "
    st.sidebar.write("Chargement :",str)
    if True:
        with st.spinner(f"⏳ Entraînement du modèle en cours..."):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

        st.title(f"---\n### 🔍 Résultats pour le modèle")
        st.markdown(f"""
        ✅ **Accuracy**        : `{acc:.4f}`  
        ✅ **Recall Score**    : `{recall:.4f}`  
        ✅ **Precision Score** : `{precision:.4f}`  
        ✅ **F1 Score**        : `{f1:.4f}`  
        ✅ **AUC-ROC**         : `{auc}`
        """)

        with st.expander("📋 Rapport détaillé"):
            rapport = classification_report(y_test, y_pred,target_names=["G", "R"], output_dict=False)
            st.text(rapport)
            st.markdown("### 🧾 Matrice de confusion")

            with st.spinner(f"⏳ Entraînement du modèle en cours..."):
                 cm = confusion_matrix(y_test, y_pred)
                 fig_cm, ax_cm = plt.subplots()
                 disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["G", "R"])
                 disp.plot(ax=ax_cm, cmap="Blues")
                 st.pyplot(fig_cm)
   
       
        

@st.cache_data
def cou(mo):

    
         st.markdown("### 🎓 Courbe d'apprentissage")

         with st.spinner(f"⏳ Courpe d'apprentissage en cours..."):
          
            train_sizes, train_scores, test_scores = learning_curve(
                model, X, y, cv=5, scoring='accuracy',
                train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
            )
            train_mean = np.mean(train_scores, axis=1)
            test_mean = np.mean(test_scores, axis=1)
            fig, ax = plt.subplots()
            ax.plot(train_sizes, train_mean, label="Train")
            ax.plot(train_sizes, test_mean, label="Validation")
            ax.set_title("Learning Curve")
            ax.set_xlabel("Training Size")
            ax.set_ylabel("Accuracy")
            ax.legend()
            st.pyplot(fig)
         
@st.cache_data
def roc(mo):

         st.markdown("### ROC curve")

         with st.spinner(f"⏳ ROC Curve {name} en cours..."):
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

 
            
            if y_proba is not None:
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                fig_roc, ax_roc = plt.subplots()
                auc = roc_auc_score(y_test, y_proba) if y_proba is not None else "N/A"

                ax_roc.plot(fpr, tpr, label=f"ROC (AUC = {auc:.2f})")
                ax_roc.plot([0, 1], [0, 1], 'k--')
                ax_roc.set_xlabel('Faux positifs')

                ax_roc.set_ylabel('Vrais positifs')
                ax_roc.legend()
                st.pyplot(fig_roc)
        

     
@st.cache_data
def tsne(df_tvae):
    
    y = df_tvae["Sample_Type"]
    X = df_tvae.drop("Sample_Type", axis=1)

    # On garde que les colonnes numériques pour t-SNE
    X_numeric = X.select_dtypes(include='number')

    # Projection t-SNE
    tsne_model = TSNE(n_components=2, random_state=42)
    X_tsne = tsne_model.fit_transform(X_numeric)

    # Couleurs personnalisées
    colors = {1: "red", 0: "blue"}
    point_colors = y.map(colors)

    # Affichage
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=point_colors)
    plt.title("Projection t-SNE")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    st.pyplot(fig)
@st.cache_data
def tsne(df_original):
    
    y = df_original["Sample_Type"]
    X = df_original.drop("Sample_Type", axis=1)

    # On garde que les colonnes numériques pour t-SNE
    X_numeric = X.select_dtypes(include='number')

    # Projection t-SNE
    tsne_model = TSNE(n_components=2, random_state=42)
    X_tsne = tsne_model.fit_transform(X_numeric)

    # Couleurs personnalisées
    colors = {1: "red", 0: "blue"}
    point_colors = y.map(colors)

    # Affichage
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=point_colors)
    plt.title("Projection t-SNE")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    st.pyplot(fig)
@st.cache_data
def tsne(df_mixte):
    
    y = df_mixte["Sample_Type"]
    X = df_mixte.drop("Sample_Type", axis=1)

    # On garde que les colonnes numériques pour t-SNE
    X_numeric = X.select_dtypes(include='number')

    # Projection t-SNE
    tsne_model = TSNE(n_components=2, random_state=42)
    X_tsne = tsne_model.fit_transform(X_numeric)

    # Couleurs personnalisées
    colors = {1: "red", 0: "blue"}
    point_colors = y.map(colors)

    # Affichage
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=point_colors)
    plt.title("Projection t-SNE")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    st.pyplot(fig)
@st.cache_data
def tsne(df_ctgan_100K):
    
    y = df_ctgan_100K["Sample_Type"]
    X = df_ctgan_100K.drop("Sample_Type", axis=1)

    # On garde que les colonnes numériques pour t-SNE
    X_numeric = X.select_dtypes(include='number')

    # Projection t-SNE
    tsne_model = TSNE(n_components=2, random_state=42)
    X_tsne = tsne_model.fit_transform(X_numeric)

    # Couleurs personnalisées
    colors = {1: "red", 0: "blue"}
    point_colors = y.map(colors)

    # Affichage
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=point_colors)
    plt.title("Projection t-SNE")
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.grid(True)
    st.pyplot(fig)


st.sidebar.title("📂 Dataset Selection")
if "datasets_choisis" not in st.session_state:
    st.session_state.datasets_choisis = []
if "df" not in st.session_state:
    st.session_state.df = []
dataset_choice = st.sidebar.multiselect(
    "Choisissez un ou plusieurs datasets",
    options=['Original','CTGAN_100k','CTGAN','TVAE','CouplaGAN'],default=['Original']
)   
st.session_state.datasets_choisis = dataset_choice
df = pd.DataFrame()
str=''
if dataset_choice==['TVAE']:
    df = df_tvae
    st.session_state.df = df
    str='TVAE'
elif dataset_choice==['Original']:
    df = df_original
    st.session_state.df = df
    str='Original'
elif dataset_choice==['CTGAN_100k']:
      df = df_ctgan_100K
      st.session_state.df = df
      str='CTGAN_100K'
elif 'Original' in dataset_choice and 'TVAE'  in dataset_choice:
      df = df_mixte
      st.session_state.df = df
      str='CTGAN_100K'
else:
    df = df_original
    st.session_state.df = df
    str='Original'
    
st.title("Ransomeware vs Goodware")
go_data = st.sidebar.button(f"🚀 Go (Charger:{str})")
if 'go_data' not in st.session_state:
    st.session_state.go_data=False

if dataset_choice:
  if go_data:
    df=st.session_state.df 

    dtt=df
    dtt['Sample_Type']=dtt['Sample_Type'].replace({0:"G",1:"R"})
    st.sidebar.write("Chargement: ",str)
    
    st.dataframe(dtt.head(11))
    dtt['Sample_Type']=dtt['Sample_Type'].replace({"G":0,"R":1})
    st.session_state.go_data=True



else:
      
      st.sidebar.markdown("pas de choix de dataset")
if dataset_choice==['TVAE']:
   if st.session_state.get("go_data", True):
    if st.sidebar.button("t-SNE"):
        st.title(f' t-SNE : {str}')
        with st.spinner("⏳ Exécution de t-SNE..."):
           tsne(df_tvae)
elif dataset_choice==['Original']:
   if st.session_state.get("go_data", True):
    if st.sidebar.button("t-SNE"):
        st.title(f' t-SNE : {str}')
        with st.spinner("⏳ Exécution de t-SNE..."):
            tsne(df_original)
elif dataset_choice==['CTGAN_100k']:
 if st.session_state.get("go_data", True):
    if st.sidebar.button("t-SNE"):
        st.title(f' t-SNE : {str}')
        with st.spinner("⏳ Exécution de t-SNE..."):
            tsne(df_ctgan_100K)
elif 'Original' in dataset_choice and 'TVAE'  in dataset_choice:
 if st.session_state.get("go_data", True):
    if st.sidebar.button("t-SNE"):
        st.title(f' t-SNE : {str}')
        with st.spinner("⏳ Exécution de t-SNE..."):
            tsne(df_mixte)
else:
 if st.session_state.get("go_data", True):
    if st.sidebar.button("t-SNE"):
        st.title(f' t-SNE : {str}')
        with st.spinner("⏳ Exécution de t-SNE..."):
            tsne(df_original)
    


# ---- SELECTION MODELE ----
st.sidebar.write("---")
choix_models = st.sidebar.multiselect(
    "🛠️ Choisissez les modèles à entraîner",
    options=["XGBoost", "RF", "SVM", "MLP", "Stack(XGBoost + SVM + MLP)", "Stack(XGBoost + RF + MLP)"],
    default=["XGBoost"]
)
go_model = st.sidebar.button("🚀 Go (Exécuter Modèle)")
if dataset_choice:
  df=st.session_state.df
  X = df.drop('Sample_Type', axis=1)
  y = df['Sample_Type']
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
if 'go_model' not in st.session_state:
    st.session_state.go_model=False
if go_model:
 if dataset_choice==['TVAE']:
    st.session_state.go_model=True
    modx(df_tvae)
 elif dataset_choice==['Original']:
    st.session_state.go_model=True
    modx(df_original)

 elif dataset_choice==['CTGAN_100k']:
    st.session_state.go_model=True
    modx(df_ctgan_100K)

 elif 'Original' in dataset_choice and 'TVAE'  in dataset_choice:
    st.session_state.go_model=True
    modx(df_mixte)
 else:
    modx(df_original)
     

    


   

   

   

    

    
    

