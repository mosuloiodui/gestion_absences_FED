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

# Fonction de lecture du fichier
def lire_fichier(chemin):
    if chemin.endswith('.csv'):
        return pd.read_csv(chemin)
    elif chemin.endswith('.xlsx'):
        return pd.read_excel(chemin)

# Fonction pour créer des modèles
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

# Fonction principale
def main():
    # Sidebar : choix entre télécharger un fichier ou travailler avec un type de dataset
    st.title("🔧 Sélection des options")
    choix_option = st.radio(
        "Choisissez votre option",
        options=["Télécharger un fichier", "Ransomware vs Goodware"]
    )
    
    if choix_option == "Télécharger un fichier":
        uploaded_file = st.file_uploader("Télécharger un fichier CSV ou Excel", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith('.xlsx'):
                df = pd.read_excel(uploaded_file)
            st.session_state.df = df
            st.success("✅ Fichier chargé avec succès!")
            st.dataframe(df.head())  
            if st.button("clic ,clic"):
                import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random
from scipy.stats import mannwhitneyu

# Chargement du dataset
df, = pd.read_excel('C:\\Users\\hp\\Downloads\\~$dataset_synthetique_ctgan (1).xlsx')

# Conversion de Sample_Type en valeurs numériques
df["Sample_Type"],unique = pd.factorize(df['Sample_Type'])

# Vérificat
# Vérification de l'équilibre Ransomware vs Goodware
type_counts = df['Sample_Type'].value_counts()
print("\nRépartition des types:")
print(type_counts)

# Visualisation de la répartition
plt.figure(figsize=(8, 6))
type_counts.plot(kind='bar', color=['blue', 'red'])
plt.title('Répartition Ransomware vs Goodware')
plt.xlabel('Type')
plt.ylabel("Nombre d'instances")
plt.xticks(ticks=[0, 1], labels=['Goodware', 'Ransomware'], rotation=0)
plt.tight_layout()
plt.savefig('type_distribution.png')
plt.close()

# Sélectionner 50 colonnes aléatoires (en excluant 'Sample_Type')
all_columns = df.columns.drop('Sample_Type')
random_columns = random.sample(list(all_columns), min(50, len(all_columns)))

# Statistiques descriptives générales sur les 50 colonnes sélectionnées
print("\nStatistiques descriptives générales (50 colonnes aléatoires):")
print(df[random_columns].describe())

# Statistiques séparées pour Goodware et Ransomware sur les 50 colonnes sélectionnées
print("\nStatistiques descriptives pour les Ransomwares (50 colonnes aléatoires):")
print(df.loc[df['Sample_Type'] == 1, random_columns].describe())

print("\nStatistiques descriptives pour les Goodwares (50 colonnes aléatoires):")
print(df.loc[df['Sample_Type'] == 0, random_columns].describe())

# Vérification des valeurs manquantes (toutes colonnes)
print("\nValeurs manquantes (toutes colonnes):")
print(df.isnull().sum())

# Analyse de la distribution des APIs (toutes colonnes)
api_usage = df.drop('Sample_Type', axis=1)
api_usage_mean = api_usage.mean().sort_values(ascending=False)

# Visualisation des APIs les plus utilisées (top 30)
plt.figure(figsize=(18, 8))
bars = api_usage_mean[:30].plot(kind='bar', color='purple')
plt.title('Top 30 des APIs les plus utilisées', pad=20)
plt.xlabel('APIs')
plt.ylabel('Moyenne d\'utilisation')
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.tight_layout()
plt.subplots_adjust(bottom=0.25)
plt.savefig('api_usage_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Boxplot des 5 APIs les plus utilisées par classe
top_5_apis = api_usage_mean.head(5).index
plt.figure(figsize=(12, 6))
sns.boxplot(data=df.melt(id_vars='Sample_Type', value_vars=top_5_apis),
            x='variable', y='value', hue='Sample_Type')
plt.yscale('log')
plt.title('Distribution des Top 5 APIs par Classe')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('top_apis_boxplot.png', dpi=300)
plt.close()

# Test de Mann-Whitney pour les top APIs
print("\nTests de Mann-Whitney pour les top APIs:")
for api in top_5_apis:
    stat, p = mannwhitneyu(df.loc[df['Sample_Type'] == 1, api],
                           df.loc[df['Sample_Type'] == 0, api])
    print(f"{api}: p-value = {p:.4f} → {'Différence significative' if p < 0.05 else 'Pas de différence'}")

# Matrice de corrélation et paires fortement corrélées
corr_matrix = api_usage.corr()

# Heatmap des corrélations (top 5 APIs)
plt.figure(figsize=(15, 12))
sns.heatmap(api_usage[top_5_apis].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Corrélations entre Top 5 APIs')
plt.tight_layout()
plt.savefig('top_apis_heatmap.png', dpi=300)
plt.close()


# Extraction des paires fortement corrélées
threshold = 0.95
high_corr_pairs = (
    corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    .stack()
    .reset_index()
)
high_corr_pairs.columns = ["API_1", "API_2", "Correlation"]
high_corr_pairs = high_corr_pairs.loc[
    (high_corr_pairs["Correlation"] >= threshold) |
    (high_corr_pairs["Correlation"] <= -threshold)
]

print("\nPaires d'APIs avec une forte corrélation :")
print(high_corr_pairs.sort_values(by="Correlation", ascending=False))
high_corr_pairs.to_csv("high_correlation_pairs.csv", index=False)

# Séparation des features et labels
x = df.drop('Sample_Type', axis=1)
y = df['Sample_Type']
print("\nFormes des données:")
print("x shape:", x.shape)
print("y shape:", y.shape)
        else:
            st.write("Aucun fichier téléchargé. Vous pouvez télécharger un fichier CSV ou Excel.")
    
    elif choix_option == "Ransomware vs Goodware":
      with st.sidebar:
        st.write("Sélectionnez un modèle pour classer entre Ransomware et Goodware.")
        choix_models = st.multiselect(
            "🛠️ Choisissez les modèles à entraîner",
            options=["XGBoost", "RF", "SVM", "MLP", "Stack(XGBoost + SVM + MLP)", "Stack(XGBoost + RF + MLP)"],
            default=["XGBoost"]
        )
        go_model = st.button("🚀 Go (Exécuter Modèle)")
        
        if go_model:
            # Charger un dataset par défaut (tu peux remplacer par un dataset spécifique)
            # Exemple : un dataset de Ransomware vs Goodware (c'est juste un placeholder ici)
            df = pd.read_csv('dataset_ransomware_goodware.csv')  # Remplace par ton propre dataset
            
            if df.empty:
                st.error("⚠️ Aucun dataset chargé. Veuillez télécharger un fichier d'abord.")
            else:
                X = df.drop('Sample_Type', axis=1)
                y = df['Sample_Type']
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                for name in choix_models:
                    model = create_model(name)
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
                    
                    with st.expander("##📋 Rapport détaillé"):
                        rapport = classification_report(y_test, y_pred, output_dict=False)
                        st.text(rapport)

                    with st.expander("### 🎓 Courbe d’apprentissage"):
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

                    with st.expander("### Voire ROC curve"):
                        if y_proba is not None:
                            fpr, tpr, _ = roc_curve(y_test, y_proba)
                            fig_roc, ax_roc = plt.subplots()
                            ax_roc.plot(fpr, tpr, label=f"ROC (AUC = {auc:.2f})")
                            ax_roc.plot([0, 1], [0, 1], 'k--')
                            ax_roc.set_xlabel('Faux positifs')
                            ax_roc.set_ylabel('Vrais positifs')
                            ax_roc.legend()
                            st.pyplot(fig_roc)

                    with st.markdown("### 🧾 Matrice de confusion"):
                        cm = confusion_matrix(y_test, y_pred)
                        fig_cm, ax_cm = plt.subplots()
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Ransomware", "Goodware"])
                        disp.plot(ax=ax_cm, cmap="Blues")
                        st.pyplot(fig_cm)

# Exécution de l'application
if __name__ == "__main__":
    main()
