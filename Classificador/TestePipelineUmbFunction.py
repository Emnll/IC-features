#%%
import numpy as np
import pandas as pd
#%%
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import set_printoptions

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.metrics import roc_auc_score
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

#%%
""" Teste da função de Pipeline para os métodos de balanceamento """

from sklearn.model_selection import cross_validate, cross_val_score
import numpy as np

pipelines = {
    'undersample': ImbPipeline([
        ('sampler', RandomUnderSampler(random_state=seed)),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=seed))
    ]),
    
    'oversample': ImbPipeline([
        ('sampler', SMOTE(random_state=seed)),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=seed))
    ]),
    
    'smoteenn': ImbPipeline([
        ('sampler', SMOTEENN(random_state=seed)),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=seed))
    ])
}

# Grid de parâmetros
param_grid = {
    'classifier__max_depth': [5, 6, 7, 8, 9, 10],
    'classifier__bootstrap': [True, False],
    'classifier__criterion': ["gini", "entropy"],
    'classifier__n_estimators': [100, 200, 300]
}

# Testar cada pipeline
results = {}
for name, pipeline in pipelines.items():
    print(f"\n{'='*70}")
    print(f"Testando: {name}")
    print('='*70)
    
    # GridSearch
    grid_search = GridSearchCV(
        pipeline, 
        param_grid, 
        cv=5, 
        scoring='roc_auc',
        n_jobs=-1,
        verbose=1,
        return_train_score=True
    )
    
    grid_search.fit(X_train, y_train)
    
    # Pegar o melhor modelo
    best_pipeline = grid_search.best_estimator_
    
    # VALIDAÇÃO CRUZADA DETALHADA COM O MELHOR MODELO
    print(f"\n{'='*70}")
    print(f"VALIDAÇÃO CRUZADA DETALHADA - {name}")
    print('='*70)
    
    scoring_metrics = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    cv_results = cross_validate(
        best_pipeline, 
        X_train, 
        y_train, 
        cv=5, 
        scoring=scoring_metrics,
        return_train_score=True
    )
    
    # Calcular médias e desvios
    cv_summary = {}
    for metric in scoring_metrics.keys():
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        cv_summary[metric] = {
            'test_mean': test_scores.mean(),
            'test_std': test_scores.std(),
            'train_mean': train_scores.mean(),
            'train_std': train_scores.std()
        }
        
        print(f"\n{metric.upper()}:")
        print(f"  Treino: {train_scores.mean():.4f} (+/- {train_scores.std():.4f})")
        print(f"  Teste:  {test_scores.mean():.4f} (+/- {test_scores.std():.4f})")
    
    # Armazenar resultados
    results[name] = {
        'best_score': grid_search.best_score_,
        'best_params': grid_search.best_params_,
        'best_estimator': best_pipeline,
        'cv_summary': cv_summary
    }
    
    print(f"\nMelhores parâmetros: {grid_search.best_params_}")

# COMPARAÇÃO FINAL
print("\n" + "="*70)
print("COMPARAÇÃO FINAL - VALIDAÇÃO CRUZADA")
print("="*70)
print(f"{'Método':<15} {'ROC-AUC':<12} {'Accuracy':<12} {'F1-Score':<12} {'Recall':<12}")
print("-"*70)

for name, result in results.items():
    roc_auc = result['cv_summary']['roc_auc']['test_mean']
    accuracy = result['cv_summary']['accuracy']['test_mean']
    f1 = result['cv_summary']['f1']['test_mean']
    recall = result['cv_summary']['recall']['test_mean']
    
    print(f"{name:<15} {roc_auc:<12.4f} {accuracy:<12.4f} {f1:<12.4f} {recall:<12.4f}")

# Selecionar o melhor modelo (por ROC-AUC)
best_method = max(results.items(), key=lambda x: x[1]['cv_summary']['roc_auc']['test_mean'])
print(f"\n{'='*70}")
print(f"MELHOR MÉTODO: {best_method[0]}")
print(f"ROC-AUC: {best_method[1]['cv_summary']['roc_auc']['test_mean']:.4f}")
print('='*70)

# Usar o melhor modelo para predições no conjunto de teste
best_model = best_method[1]['best_estimator']

print("\n" + "="*70)
print("AVALIAÇÃO NO CONJUNTO DE TESTE")
print("="*70)

y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]



print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print(f"\nROC-AUC no conjunto de teste: {roc_auc_score(y_test, y_proba):.4f}")

# FEATURE IMPORTANCE
print("\n" + "="*70)
print("FEATURE IMPORTANCE")
print("="*70)

rfc = best_model.named_steps['classifier']

plt.rcParams['figure.figsize'] = (12, 10)
importances = pd.Series(data=rfc.feature_importances_, index=X_train.columns.values)
importances_sorted = importances.sort_values(ascending=True)

plt.figure(figsize=(12, 10))
sns.barplot(x=importances_sorted, y=importances_sorted.index, orient='h')
plt.title(f'Importância de cada feature - {best_method[0]}')
plt.xlabel('Importância')
plt.tight_layout()
plt.show()

# PREDIÇÕES EM DADOS EXTERNOS
print("\n" + "="*70)
print("PREDIÇÕES EM DADOS EXTERNOS")
print("="*70)

# Musculus
y_musculus_pred = best_model.predict(X_musculus)
print("\nMUSCULUS:")
print(f"Predições: {y_musculus_pred}")
print("\nClassification Report (Musculus):")
print(classification_report(y_musculus, y_musculus_pred))

# Mansoni
y_mansoni_pred = best_model.predict(X_mansoni)
print("\nMANSONI:")
print(f"Predições: {y_mansoni_pred}")

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
# Predizendo proteínas do mansoni

y_musculus_rf_over = best_model.predict(X_musculus)

print(y_musculus_rf_over)

#%%¨
print(classification_report(y_musculus, y_musculus_rf_over))

#%%

y_mansoni_rf_over = best_model.predict(X_mansoni)
print(y_mansoni_rf_over)

#%%

y_mansoni_rf_under = best_model.predict(X_mansoni)
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