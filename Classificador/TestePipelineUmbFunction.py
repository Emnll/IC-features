#%%
from datetime import datetime
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report, accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import RFE, SelectKBest, f_classif, VarianceThreshold, SelectFromModel

import pickle
import joblib

import os
from tqdm import tqdm

#%%
pasta_atual = os.getcwd()
print(pasta_atual)

df_cel = pd.read_csv(os.path.join(pasta_atual, 'Todas_Features/Features_cel.csv'))
df_sce = pd.read_csv(os.path.join(pasta_atual, 'Todas_Features/Features_sce.csv'))
df_dme = pd.read_csv(os.path.join(pasta_atual, 'Todas_Features/Features_dme.csv'))

df_mus = pd.read_csv(os.path.join(pasta_atual, 'Todas_Features/Features_mus.csv'))

df_man = pd.read_csv(os.path.join(pasta_atual, 'Todas_Features/Features_man.csv'))

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
# Dados mansoni
df_mansoni = df_man.set_index('Locus')
X_mansoni = df_mansoni.drop(['Sequence'], axis=1)

# Dados musculus
df_musculus = df_mus.set_index('Locus')
X_musculus = df_musculus.drop(['Sequence', 'IsEssential'], axis=1)
y_musculus = df_mus['IsEssential']

#%%
def pipeline(param_grid, pipelines, X_train, y_train, X_test, y_test):

    results = {}

    for name, pipeline in tqdm(pipelines.items()):
        print(f"\n{'='*70}")
        print(f"Testando: {name}")
        print('='*70)
        
        # Pegar o melhor modelo
        search = GridSearchCV(estimator=pipeline, param_grid=param_grid, scoring='f1', cv=5, verbose=3, n_jobs = -1)

        search.fit(X_train, y_train)

        y_pred = search.best_estimator_.predict(X_test)

        results[name] = {
            'best_params': search.best_params_,
            'best_score_cv': search.best_score_,
            'test_f1': f1_score(y_test, y_pred),
            'cv_results': search.cv_results_
        }

    return results

def evaluate_on_musculus(results, X_musculus, y_musculus):

    all_metrics = {}

    for name in results:
        filename = f'Modelos-PKL/{name}_model.pkl'
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)

        model  = model_data['pipeline']
        y_pred = model.predict(X_musculus)

        all_metrics[name] = {
            'f1':        f1_score(y_musculus, y_pred),
            'precision': precision_score(y_musculus, y_pred),
            'recall':    recall_score(y_musculus, y_pred),
        }

    print(f"\n{'='*70}")
    print("RESULTADOS EM X_MUSCULUS")
    print(f"{'='*70}")
    print(f"{'Método':<18} {'F1':<12} {'Precision':<12} {'Recall':<12}")
    print("-"*70)
    
    for name, m in all_metrics.items():
        print(f"{name:<18} {m['f1']:<12.4f} {m['precision']:<12.4f} {m['recall']:<12.4f}")

    return all_metrics

def salvar_metricas_csv(results, X_musculus, y_musculus, output_path="resultados_modelos.csv"):
    
    rows = []

    for name in results:
        # Carregar modelo
        filename = f'Modelos-PKL/{name}_model.pkl'
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)

        model = model_data['pipeline']

        # Predição no mus musculus
        y_pred = model.predict(X_musculus)

        # Métricas
        f1 = f1_score(y_musculus, y_pred)
        precision = precision_score(y_musculus, y_pred)
        recall = recall_score(y_musculus, y_pred)

        # Extrair info
        best_params = results[name]['best_params']
        cv_score = results[name]['best_score_cv']
        test_f1 = results[name]['test_f1']

        # Extrair tipo de sampling (assumindo pipeline com 'sampler')
        sampler = model.named_steps.get('sampler', None)
        sampler_name = type(sampler).__name__ if sampler else "None"

        # Nome do modelo
        classifier = model.named_steps.get('classifier', None)
        model_name = type(classifier).__name__ if classifier else name

        rows.append({
            'pipeline_name': name,
            'model': model_name,
            'sampling': sampler_name,
            'best_params': str(best_params),
            'cv_f1': cv_score,
            'test_f1': test_f1,
            'mus_f1': f1,
            'precision': precision,
            'recall': recall
        })

    df = pd.DataFrame(rows)

    # Ordenar pelos melhores
    df = df.sort_values(by='cv_f1', ascending=False)

    # Salvar CSV
    df.to_csv(output_path, index=False)

    print(f"Arquivo salvo em: {output_path}")

    return df

def predict_external(best_model, X_mansoni):
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

def feature_selection(X_train, y_train, model, k, method='rfe'):
    if method == 'rfe':
        # Seleção de features usando RFE
        selector = RFE(estimator=model, n_features_to_select=k, step=1)
        selector = selector.fit(X_train, y_train)
    elif method == 'selectkbest':
        # Seleção de features usando SelectKBest
        selector = SelectKBest(score_func=f_classif, k=k)
        selector = selector.fit(X_train, y_train)
    elif method == 'variancethreshold':
        # Seleção de features usando VarianceThreshold
        selector = VarianceThreshold(threshold=0.01)
        selector = selector.fit(X_train, y_train)
    elif method == 'selectfrommodel':
        # Seleção de features usando SelectFromModel
        selector = SelectFromModel(estimator=model, threshold='median')
        selector = selector.fit(X_train, y_train)

    # Imprime as features selecionadas
    selected_features = X_train.columns[selector.support_]
    print(f"Features selecionadas: {selected_features.tolist()}")

    return selected_features
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

rfc_results = pipeline(rfc_grid, pipelines, X_train, y_train, X_test, y_test)
rfc_model, rfc_metrics = salvar_metricas_csv(rfc_results, X_musculus, y_musculus)


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

xgb_results = pipeline(xgb_grid, pipelines, X_train, y_train, X_test, y_test)
xgb_model = salvar_metricas_csv(xgb_results, X_musculus, y_musculus)
predict_external(xgb_model, X_musculus, y_musculus, X_mansoni)
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

gb_results = pipeline(gb_grid, pipelines, X_train, y_train, X_test, y_test)
gb_model = print_results(gb_results)
predict_external(gb_model, X_musculus, y_musculus, X_mansoni)
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
