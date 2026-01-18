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
""" Dropando Features para feature selection """
emboss_columns = [ 'Sequence_Length', 'Aromaticity', 'Sec_Struct_Helix', 'Sec_Struct_Turn',
    'Percent_A', 'Percent_D', 'Percent_E', 'Percent_F', 'Percent_G', 'Percent_H',
    'Percent_I', 'Percent_K', 'Percent_L', 'Percent_M', 'Percent_N', 'Percent_P', 'Percent_Q',
    'Percent_R', 'Percent_S', 'Percent_T', 'Percent_V', 'Percent_W', 'Percent_Y', 'IsoelectricPoint',
    'Tiny_Number', 'Small_Number', 'Aliphatic_Number', 'Aromatic_Number', 'Non-polar_Number',
    'Polar_Number', 'Charged_Number', 'Basic_Number', 'Acidic_Number']
df_cel = df_cel.drop(emboss_columns, axis = 1)
df_sce = df_sce.drop(emboss_columns, axis = 1)
df_dme = df_dme.drop(emboss_columns, axis = 1)

df_mus = df_mus.drop(emboss_columns, axis = 1)

df_man = df_man.drop(emboss_columns, axis = 1)

#%%
df = pd.concat([df_cel, df_sce, df_dme], ignore_index=True)

#%%

""" Início ML """

#%%
# Função para validação cruzada sem o uso de balanceamento na amostra de validação

def validacao_cruzada(model, X, y, sampling = False, method_sampling = None):
    kfold = KFold(n_splits = 5, shuffle=True)
    
    acuracias_split = []
    
    for idx, (idx_treino, idx_validacao) in enumerate(kfold.split(X)):
        X_split_treino = X.iloc[idx_treino, :]
        y_split_treino = y.iloc[idx_treino, :]
        
        
        if sampling: 
            # Método de Sampling
            sm = method_sampling
            X_split_treino, y_split_treino = sm.fit_resample(X_split_treino, y_split_treino)
            
        model.fit(X_split_treino, y_split_treino.values.ravel())  
        
        X_split_validacao = X.iloc[idx_validacao, :]
        y_split_validacao = y.iloc[idx_validacao, :]
        
        
        pred_validacoes = model.predict(X_split_validacao)
        
        acuracia_split = accuracy_score(y_split_validacao, pred_validacoes)
        
        acuracias_split.append(acuracia_split)
        
        print(f'Acurácia do split {idx}: {acuracia_split}')
        
        mean = np.mean(acuracias_split)
        
    return print(f'Média de Acurácia na validação cruzada: {mean}')

# %%
def gridsearch(X_train, y_train, model, param_grid, scoring, kfold):
    
    # busca exaustiva de hiperparâmetros com GridSearchCV
    grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=3, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, y_train)

    # imprime o melhor resultado
    print("Melhor: %f usando %s" % (grid_result.best_score_, grid_result.best_params_)) 

    # imprime todos os resultados
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f): %r" % (mean, stdev, param))

#%%
# Separação em conjuntos de treino e teste
X = df.drop(['Locus','IsEssential', 'Sequence'], axis=1)

y = df[['IsEssential']]
test_size = 0.2
seed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=test_size, 
                                                    random_state=seed, 
                                                    stratify=y)
#%%
# Dados mansoni
df_mansoni = df_man.set_index('Locus')
X_mansoni = df_mansoni.drop(['Sequence'], axis=1)
X_mansoni

#%%
# Dados musculus
df_musculus = df_mus.set_index('Locus')
X_musculus = df_musculus.drop(['Sequence', 'IsEssential'], axis=1)
y_musculus = df_mus['IsEssential']
y_musculus

#%%
## Undersampling

undersample = RandomUnderSampler(random_state=seed)

X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)

y_train_under.value_counts()

#%%
## Oversampling

oversample = SMOTE(random_state=seed)

X_train_over, y_train_over = oversample.fit_resample(X_train, y_train)

y_train_over.value_counts()

#%%
## Combine

sample = SMOTEENN(random_state=seed)

X_train_sample, y_train_sample = sample.fit_resample(X_train, y_train)

y_train_sample.value_counts()
# %%
param_grid = dict(max_depth =[5,6,7,8,9,10],
                  bootstrap = [True, False],
                  criterion = ["gini", "entropy"],
                  n_estimators= [100,200,300])

# %%

"""Algoritmo Random Forest"""

scoring = 'roc_auc'
kfold = 5

rfc = RandomForestClassifier(random_state = seed)

# Busca de Hiperparametros
gridsearch(X_train_sample, y_train_sample, rfc, param_grid, scoring, kfold)
# %%
rfc = RandomForestClassifier(bootstrap= False, criterion= 'entropy', 
                             max_depth= 10, n_estimators= 300, random_state = seed)

rfc.fit(X_train_under, y_train_under)
# Verificando a importância de cada feature

plt.rcParams['figure.figsize'] = (12,10)

importances = pd.Series(data=rfc.feature_importances_, index=X_train.columns.values)
sns.barplot(x=importances, y=importances.index, orient='h').set_title('Importância de cada feature')

#%%
# Validação cruzada 
validacao_cruzada(rfc, X_train, y_train, True, undersample)

#%%
y_pred = rfc.predict(X_test)

#%%
# Scikit-learn
print(confusion_matrix(y_test, y_pred))

#%%
print(classification_report(y_test, y_pred))

#%%
""" Organizando a ordem das features """
feature_order = [
    'Sequence_Length', 'Aromaticity', 'Sec_Struct_Helix', 'Sec_Struct_Turn', 'Sec_Struct_Sheet',
    'Percent_A', 'Percent_C', 'Percent_D', 'Percent_E', 'Percent_F', 'Percent_G', 'Percent_H',
    'Percent_I', 'Percent_K', 'Percent_L', 'Percent_M', 'Percent_N', 'Percent_P', 'Percent_Q',
    'Percent_R', 'Percent_S', 'Percent_T', 'Percent_V', 'Percent_W', 'Percent_Y', 'IsoelectricPoint',
    'Tiny_Number', 'Small_Number', 'Aliphatic_Number', 'Aromatic_Number', 'Non-polar_Number',
    'Polar_Number', 'Charged_Number', 'Basic_Number', 'Acidic_Number', 'Local Average Connectivity',
    'Density of Maximum neighborhood Component', 'Topology Potential', 'Edge Clustering Coefficient',
    'DegreeCentrality', 'EigenvectorCentrality', 'BetweennessCentrality', 'ClosenessCentrality', 'Clustering'
]

X_musculus = X_musculus[feature_order]
X_mansoni = X_mansoni[feature_order]

#%%
""" Organizando features sem emboss """

feature_order = [
    'Sequence_Length', 'Aromaticity', 'Sec_Struct_Helix', 'Sec_Struct_Turn', 'Sec_Struct_Sheet',
    'Percent_A', 'Percent_C', 'Percent_D', 'Percent_E', 'Percent_F', 'Percent_G', 'Percent_H',
    'Percent_I', 'Percent_K', 'Percent_L', 'Percent_M', 'Percent_N', 'Percent_P', 'Percent_Q',
    'Percent_R', 'Percent_S', 'Percent_T', 'Percent_V', 'Percent_W', 'Percent_Y', 'Local Average Connectivity',
    'Density of Maximum neighborhood Component', 'Topology Potential', 'Edge Clustering Coefficient',
    'DegreeCentrality', 'EigenvectorCentrality', 'BetweennessCentrality', 'ClosenessCentrality', 'Clustering'
]

X_musculus = X_musculus[feature_order]
X_mansoni = X_mansoni[feature_order]

#%%
""" Organizando features com feature selection """

feature_order = [
    'Sec_Struct_Sheet', 'Percent_C', 'Local Average Connectivity',
    'Density of Maximum neighborhood Component', 'Topology Potential', 'Edge Clustering Coefficient',
    'DegreeCentrality', 'EigenvectorCentrality', 'BetweennessCentrality', 'ClosenessCentrality', 'Clustering'
]

X_musculus = X_musculus[feature_order]
X_mansoni = X_mansoni[feature_order]
#%%
# Predizendo proteínas do mansoni

y_musculus_rf_over = rfc.predict(X_musculus)

print(y_musculus_rf_over)

#%%¨
print(classification_report(y_musculus, y_musculus_rf_over))

#%%

y_mansoni_rf_over = rfc.predict(X_mansoni)
print(y_mansoni_rf_over)

#%%

y_mansoni_rf_under = rfc.predict(X_mansoni)
print(y_mansoni_rf_under)



"""=======================Fim RF======================================="""
# %%
"""Algoritmo XGBoost"""

param_grid = {'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
              'max_depth': [6,7,8,9,10],
              'n_estimators': [100,200,300]}

#%%
xgbc = XGBClassifier(booster='gbtree', verbosity=1, random_state = seed)

scoring = 'roc_auc'
kfold = 5

# Busca de Hiperparametros
gridsearch(X_train_under, y_train_under, xgbc, param_grid, scoring, kfold)

#%%

# Training the XGB classifier

xgbc = XGBClassifier(booster='gbtree', learning_rate = 0.1,
                     max_depth=7, n_estimators = 300, verbosity=1, random_state = seed)

xgbc.fit(X_train_under, y_train_under)

#%%
# Verificando a importância das features

plt.rcParams['figure.figsize'] = (12,10)

importances = pd.Series(data=xgbc.feature_importances_, index=X_train.columns.values)

sns.barplot(x=importances, y=importances.index, orient='h').set_title('Importância de cada feature')

#%%

# Validação cruzada 
validacao_cruzada(xgbc, X_train, y_train, True, undersample)

#%%
y_pred = xgbc.predict(X_test)

#%%
# Scikit-learn
print(confusion_matrix(y_test, y_pred))

#%%
print(classification_report(y_test, y_pred))

#%%

# Predizendo proteínas do mus musculus

y_musculus_xgb = xgbc.predict(X_musculus)
print(y_musculus_xgb)

#%%
print(classification_report(y_musculus, y_musculus_xgb))

#%%

y_mansoni_xgb_under = xgbc.predict(X_mansoni)
print(y_mansoni_xgb_under)

#%%

X_mansoni['predict_xgboost'] = y_mansoni_xgb_under

X_mansoni['predict_rfc_under'] = y_mansoni_rf_under

X_mansoni

# %%
X_mansoni['predict_rfc_over'] = y_mansoni_rf_over

X_mansoni['predict_rfc_under'] = y_mansoni_rf_under

X_mansoni

#%%

X_mansoni['predict_rfc_over'].value_counts()
# %%

X_mansoni['predict_xgboost'].value_counts()

# %%
# Contando as proteínas candidatas a essenciais

X_mansoni['predict_rfc_under'].value_counts()
# %%
X_mansoni[(X_mansoni['predict_rfc_over'] == 1) & (X_mansoni['predict_rfc_under'] == 1)]

#%%

X_mansoni = X_mansoni.reset_index()


#%%
X_mansoni.to_csv('results_mansoni_feature_selection.csv', index = False)

# %%
# Dataset de descrição das proteínas

df_all_features = pd.read_csv("results_mansoni_feature_selection.csv")
df_features_selection = pd.read_csv("results_mansoni_all_features.csv")


#%%
# Merge Datasets
df_final = df_features_selection.merge(df_all_features, how='inner', left_on = 'Locus', right_on='Locus')

#%%
columns_to_drop = ['Local Average Connectivity_x', 'Density of Maximum neighborhood Component_x', 
                   'Topology Potential_x', 'Edge Clustering Coefficient_x', 
                   'DegreeCentrality_x', 'EigenvectorCentrality_x', 'EigenvectorCentrality_x', 
                   'BetweennessCentrality_x', 'ClosenessCentrality_x', 'Clustering_x', 
                   'Sec_Struct_Sheet_y', 'Percent_C_y']

df_final = df_final.drop(columns_to_drop, axis = 1)
# %%
df_final = df_final[(df_final['predict_rfc_over'] == 1) & (df_final['predict_rfc_under'] == 1) & (df_final['predict_xgboost']) & (df_final['predict_rfc'])]

# %%

df_final.to_csv("results_mansoni_final.csv", index = False)
# %%
df_final = pd.read_csv("results_mansoni_final.csv")
df_acessorio = pd.read_csv("protein_mansoni_details.txt", sep = "\t")

#%%
df_final_artigo = df_final.merge(df_acessorio, how= 'left', left_on = 'Locus', right_on = '#string_protein_id')

#Retirada das proteínas putativas
df_final_artigo = df_final_artigo[~df_final_artigo['annotation'].str.contains('putative', na=False)]
df_final_artigo = df_final_artigo[~df_final_artigo['annotation'].str.contains('Putative', na=False)]

df_final_artigo
# %%
df_final_artigo.to_csv("results_proteins_with_details.csv", index = False)

# %%
df = pd.read_csv("results_proteins_with_details.csv")
df.head(20)[['Locus', 'annotation']]
# %%