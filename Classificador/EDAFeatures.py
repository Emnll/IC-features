# -*- coding: utf-8 -*-
"""
Created on Sun May 18 19:52:23 2025

@author: Emanu
"""

#%%
import numpy as np
import pandas as pd
#%%
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import set_printoptions

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV 

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score, KFold

import os

#%%
pasta_atual = os.getcwd()
print(pasta_atual)

#%%
df_cel = pd.read_csv(os.path.join(pasta_atual, 'Todas_Features/Features_cel.csv'))
df_sce = pd.read_csv(os.path.join(pasta_atual, 'Todas_Features/Features_sce.csv'))
df_dme = pd.read_csv(os.path.join(pasta_atual, 'Todas_Features/Features_dme.csv'))

df_mus = pd.read_csv(os.path.join(pasta_atual, 'Todas_Features/Features_mus.csv'))

df_man = pd.read_csv(os.path.join(pasta_atual, 'Todas_Features/Features_man.csv'))

#%%
df = pd.concat([df_cel, df_sce, df_dme], ignore_index=True)

#%%
""" Fase de visualização """

#Heatmap de correlação
heatmap_df = df.drop(['Locus','Sequence'], axis=1)
sns.heatmap(heatmap_df.corr(), cmap='RdBu')

plt.rcParams["figure.figsize"] = (20,12)
plt.title('Heatmap de Correlação entre as Features')
plt.savefig(os.path.join(pasta_atual, "Graficos/heatmap_correlation.jpg"))

#%%
features = ['DegreeCentrality', 'EigenvectorCentrality', 'BetweennessCentrality',
       'ClosenessCentrality', 'Clustering', 'Local Average Connectivity',
       'Density of Maximum neighborhood Component', 'Topology Potential', 
       'Edge Clustering Coefficient']

plt.rcParams['figure.figsize'] = (20,15)


# Define as configurações dos plots
# Cada plot terá o mesmo tamanho de figuras (8,10)
#plt.style.use("ggplot")

plt.figure(1)

sns.set_theme(style="whitegrid", palette="dark")

plt.title('Distribuição dos dados - Baseadas em Sequência')

# Dados para cada subplot

for index, value in zip(range(1,10), features):
    plt.subplot(3, 3, index)
    sns.histplot(data=df, x=value, hue="IsEssential", 
             stat="probability", common_norm=False, kde=True)

plt.subplots_adjust(top=0.95, bottom=0.05, left=0.10, right=0.95, hspace=0.4,
                    wspace=0.25)

plt.savefig(os.path.join(pasta_atual, "Graficos/results_centrality.jpg"))
plt.show()

#%%
features_seq = ['Sequence_Length', 'Aromaticity', 'Sec_Struct_Helix',
       'Sec_Struct_Turn', 'Sec_Struct_Sheet', 'Percent_A', 'Percent_C',
       'Percent_D', 'Percent_E', 'Percent_F', 'Percent_G', 'Percent_H',
       'Percent_I', 'Percent_K', 'Percent_L', 'Percent_M', 'Percent_N',
       'Percent_P', 'Percent_Q', 'Percent_R', 'Percent_S', 'Percent_T',
       'Percent_V', 'Percent_W', 'Percent_Y']


plt.rcParams['figure.figsize'] = (40,30)

plt.figure(1)

sns.set_theme(style="whitegrid", palette="dark")

plt.title('Distribuição dos dados - Medidas de Centralidade e Clustering')

# Dados para cada subplot

for index, value in zip(range(1,37), features_seq):
    plt.subplot(5, 5, index)
    sns.histplot(data=df, x=value, hue="IsEssential", 
             stat="probability", common_norm=False, kde=True)
    

plt.subplots_adjust(top=0.95, bottom=0.05, left=0.10, right=0.95, hspace=0.4,
                    wspace=0.25)

plt.savefig(os.path.join(pasta_atual, "Graficos/results_sequence.jpg"))
plt.show()

# %%
features_seq = ['IsoelectricPoint',
       'Tiny_Number', 'Small_Number', 'Aliphatic_Number', 'Aromatic_Number',
       'Non-polar_Number', 'Polar_Number', 'Charged_Number', 'Basic_Number']


plt.rcParams['figure.figsize'] = (20,20)

plt.figure(1)

sns.set_theme(style="whitegrid", palette="dark")

plt.title('Distribuição dos dados - Medidas de Centralidade e Clustering')

# Dados para cada subplot

for index, value in zip(range(1,37), features_seq):
    plt.subplot(3, 3, index)
    sns.histplot(data=df, x=value, hue="IsEssential", 
             stat="probability", common_norm=False, kde=True)
    

plt.subplots_adjust(top=0.95, bottom=0.05, left=0.10, right=0.95, hspace=0.4,
                    wspace=0.25)

plt.savefig(os.path.join(pasta_atual, "Graficos/results_emboss.jpg"))
plt.show()
# %%