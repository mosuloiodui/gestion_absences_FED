import streamlit as st
import pandas as pd




st.title('TEST 1')
sidebar=st.sidebar
All=sidebar.checkbox('ALL')
Original=sidebar.checkbox('Original')
CTGAN=sidebar.checkbox('CTGAN')
CTGAN_100k=sidebar.checkbox('CTGAN_100K')
TVAE=sidebar.checkbox('TVAE')
CouplaGAN=sidebar.checkbox('CouplaGAN')
clic=[All,Original,CTGAN,CTGAN_100k,TVAE,CouplaGAN]
out=["ALL","Original","CTGAN","CTGAN_100k","TVAE","CouplaGAN"]
#############""""
#############"
# ##########"
dataset=[]
for i in range(len(clic)):
     if clic[i]:
        if out[i]=='ALL':
            dataset='ALL'
            break
        dataset.append[out[i]]

if sidebar.selectbuton('i choose'):
    choix_models = side.multiselect(
        "Choisissez les modèles à utiliser",
        options=["XGBoost", "RF", "SVM", "MLP", "Stack(XGBoost + SVM + MLP)", "Stack(XGBoost + RF + MLP)"],# Modèles par défaut
        )
    for i in choix_model:
         sidebar.write(f"i choose{i}")
    
    
     
     