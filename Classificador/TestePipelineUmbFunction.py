#%%
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import RandomForestClassifier

import pickle
import joblib

import os

#%%
def gridsearch(X_train, y_train, model, param_grid, scoring, kfold):
    
    # busca exaustiva de hiperparâmetros com GridSearchCV
    grid = GridSearchCV(estimator=model, param_grid=param_grid, verbose=3, scoring=scoring, cv=kfold)
    grid_result = grid.fit(X_train, y_train)

    # imprime o melhor resultado
    print("Melhor: %f usando %s" % (grid_result.best_score_, grid_result.best_params_)) 

    return grid_result

#%%
def salva_pickle(filename, best_grid, cv_summary, name, X_train):
    model_data = {
        'pipeline': best_grid.best_estimator_,
        'pipeline_name': name,
        'best_params': best_grid.best_params_,
        'cv_summary': cv_summary,
        'feature_names': X_train.columns.tolist(),
        'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    filename = f'Modelos-PKL/{name}_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model_data, f)
    
    print(f"✅ Salvo: {filename}")
    print(f"F1-Score: {cv_summary['f1']['test_mean']:.4f}")
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

""" Início ML """
# Separação em conjuntos de treino e teste
X = df.drop(['Locus','IsEssential', 'Sequence'], axis=1)

y = df['IsEssential']
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
" ================= Pipeline Random Forest ================== "

pipelines = {
    'rf-undersample': ImbPipeline([
        ('sampler', RandomUnderSampler(random_state=seed)),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=seed))
    ]),
    
    'rf-oversample': ImbPipeline([
        ('sampler', SMOTE(random_state=seed)),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=seed))
    ]),
    
    'rf-smoteenn': ImbPipeline([
        ('sampler', SMOTEENN(random_state=seed)),
        ('classifier', RandomForestClassifier(class_weight='balanced', random_state=seed))
    ])
}

# Grid de parâmetros
rfc_grid = {
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
    
    # Pegar o melhor modelo
    best_grid = gridsearch(X_train, y_train, pipeline, rfc_grid, 'roc_auc', 5)
    
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
        best_grid.best_estimator_, 
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
        'best_score': best_grid.best_score_,
        'best_params': best_grid.best_params_,
        'best_estimator': best_grid.best_estimator_,
        'cv_summary': cv_summary
    }
    
    print(f"\nMelhores parâmetros: {best_grid.best_params_}")
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

model_data = {
    'pipeline': best_method[1]['best_estimator'],
    'pipeline_name': best_method[0],
    'best_params': best_method[1]['best_params'],
    'cv_summary': best_method[1]['cv_summary'],
    'feature_names': X_train.columns.tolist(),
    'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

filename = f'Modelos-PKL/{best_method[0]}_model.pkl'
with open(filename, 'wb') as f:
    pickle.dump(model_data, f)
    
print(f"✅ Salvo: {filename}")
print(f"F1-Score: {best_method[1]['cv_summary']['f1']['test_mean']:.4f}")

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

"""=======================Fim RF======================================="""

#%%

"============ Pipeline XGBoost =============="""

pipelines = {
    'xgb_undersample': ImbPipeline([
        ('sampler', RandomUnderSampler(random_state=seed)),
        ('classifier', XGBClassifier(booster='gbtree', verbosity=0, random_state=seed))
    ]),
    
    'xgb_oversample': ImbPipeline([
        ('sampler', SMOTE(random_state=seed)),
        ('classifier', XGBClassifier(booster='gbtree', verbosity=0, random_state=seed))
    ]),
    
    'xgb_smoteenn': ImbPipeline([
        ('sampler', SMOTEENN(random_state=seed)),
        ('classifier', XGBClassifier(booster='gbtree', verbosity=0, random_state=seed))
    ])
}

# Grid de parâmetros
xgb_grid = {
  'classifier__learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
   'classifier__max_depth': [6, 7, 8, 9, 10],
   'classifier__n_estimators': [100, 200, 300]
}

# Testar cada pipeline
results = {}
for name, pipeline in pipelines.items():
    print(f"\n{'='*70}")
    print(f"Testando: {name}")
    print('='*70)
    
    # Pegar o melhor modelo
    best_grid = gridsearch(X_train, y_train, pipeline, xgb_grid, 'roc_auc', 5)
    
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
        best_grid.best_estimator_, 
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
        'best_score': best_grid.best_score_,
        'best_params': best_grid.best_params_,
        'best_estimator': best_grid.best_estimator_,
        'cv_summary': cv_summary
    }
    
    print(f"\nMelhores parâmetros: {best_grid.best_params_}")
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

model_data = {
    'pipeline': best_method[1]['best_estimator'],
    'pipeline_name': best_method[0],
    'best_params': best_method[1]['best_params'],
    'cv_summary': best_method[1]['cv_summary'],
    'feature_names': X_train.columns.tolist(),
    'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

filename = f'Modelos-PKL/{best_method[0]}_model.pkl'
with open(filename, 'wb') as f:
    pickle.dump(model_data, f)
    
print(f"✅ Salvo: {filename}")
print(f"F1-Score: {best_method[1]['cv_summary']['f1']['test_mean']:.4f}")

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

"============ Pipeline Gradient Boosting =============="""

pipelines = {
    'gb_undersample': ImbPipeline([
        ('sampler', RandomUnderSampler(random_state=seed)),
        ('classifier', GradientBoostingClassifier(random_state=seed))
    ]),

    'gb_oversample': ImbPipeline([
        ('sampler', SMOTE(random_state=seed)),
        ('classifier', GradientBoostingClassifier(random_state=seed))
    ]),
    
    'gb_smoteenn': ImbPipeline([
        ('sampler', SMOTEENN(random_state=seed)),
        ('classifier', GradientBoostingClassifier(random_state=seed))
    ])
}

# Grid de parâmetros
gb_grid= {
    'classifier__learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 1.0],
    'classifier__max_depth': [5, 6, 7, 8, 9, 10],
    'classifier__n_estimators': [100, 200, 300]
}

# Testar cada pipeline
results = {}
for name, pipeline in pipelines.items():
    print(f"\n{'='*70}")
    print(f"Testando: {name}")
    print('='*70)
    
    # Pegar o melhor modelo
    best_grid = gridsearch(X_train, y_train, pipeline, gb_grid, 'roc_auc', 5)
    
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
        best_grid.best_estimator_, 
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
        'best_score': best_grid.best_score_,
        'best_params': best_grid.best_params_,
        'best_estimator': best_grid.best_estimator_,
        'cv_summary': cv_summary
    }
    
    print(f"\nMelhores parâmetros: {best_grid.best_params_}")
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

model_data = {
    'pipeline': best_method[1]['best_estimator'],
    'pipeline_name': best_method[0],
    'best_params': best_method[1]['best_params'],
    'cv_summary': best_method[1]['cv_summary'],
    'feature_names': X_train.columns.tolist(),
    'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
}

filename = f'Modelos-PKL/{best_method[0]}_model.pkl'
with open(filename, 'wb') as f:
    pickle.dump(model_data, f)
    
print(f"✅ Salvo: {filename}")
print(f"F1-Score: {best_method[1]['cv_summary']['f1']['test_mean']:.4f}")

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

""" Carregando os melhores modelos dos tipos de classificadores """
# Os melhores modelos de cada tipo de classificador foram salvos e agora serão utilizados para predizer o mus musculus, então com esses resultados será feitas a interseção e por fim as métricas dos resultados
# Por fim o mesmo será feito no organismo alvo, mansoni


print("\n" + "="*70)
print("PREDIÇÕES NO MUS MUSCULUS")
print("="*70)

# Predições de cada modelo
predictions_musculus = {}

for model_type, info in best_models.items():
    model = info['model']
    model_name = info['name']
    
    # Fazer predições
    y_pred = model.predict(X_musculus)
    y_proba = model.predict_proba(X_musculus)[:, 1]
    
    predictions_musculus[model_type] = {
        'predictions': y_pred,
        'probabilities': y_proba,
        'model_name': model_name
    }
    
    # Avaliar
    print(f"\n{model_type} ({model_name}):")
    print(f"  Predições positivas: {y_pred.sum()}/{len(y_pred)}")
    print("\n" + classification_report(y_musculus, y_pred, target_names=['Não-Essencial', 'Essencial']))
