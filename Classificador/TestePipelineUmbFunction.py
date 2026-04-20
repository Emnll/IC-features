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

import os
from tqdm import tqdm

import mlflow

#%%
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment(experiment_id= 1)

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

#%%
def pipeline(param_grid, pipelines, X_train, y_train, X_test, y_test, X_musculus=None, y_musculus=None):

    results = {}

    for name, pipeline in tqdm(pipelines.items()):

        with mlflow.start_run(run_name=name):

            print(f"\n{'='*70}")
            print(f"Testando: {name}")
            print('='*70)
            
            search = GridSearchCV(
                estimator=pipeline,
                param_grid=param_grid,
                scoring='f1',
                cv=5,
                n_jobs=-1
            )

            search.fit(X_train, y_train)

            best_model = search.best_estimator_

            # ===== TEST SET =====
            y_pred_test = best_model.predict(X_test)
            test_f1 = f1_score(y_test, y_pred_test)

            report_test = classification_report(y_test, y_pred_test, output_dict=True)

            # métricas por classe
            # Classe 0
            mlflow.log_metric("test_precision_class_0", report_test['0']['precision'])
            mlflow.log_metric("test_recall_class_0", report_test['0']['recall'])
            mlflow.log_metric("test_f1_class_0", report_test['0']['f1-score'])

            # Classe 1
            mlflow.log_metric("test_precision_class_1", report_test['1']['precision'])
            mlflow.log_metric("test_recall_class_1", report_test['1']['recall'])
            mlflow.log_metric("test_f1_class_1", report_test['1']['f1-score'])

            # ===== LOGS MANUAIS =====
            
            # 🔹 parâmetros
            mlflow.log_param("pipeline_name", name)
            mlflow.log_params(search.best_params_)

            # 🔹 tipo de sampler
            sampler = best_model.named_steps.get('sampler')
            mlflow.log_param("sampler", type(sampler).__name__ if sampler else "None")

            # 🔹 tipo de modelo
            classifier = best_model.named_steps.get('classifier')
            mlflow.log_param("model", type(classifier).__name__)

            # 🔹 métricas internas
            mlflow.log_metric("cv_f1", search.best_score_)
            mlflow.log_metric("test_f1", test_f1)

            # ===== MUS MUSCULUS (SE TIVER) =====
            if X_musculus is not None:

                y_pred_mus = best_model.predict(X_musculus)

                report = classification_report(y_musculus, y_pred_mus, output_dict=True)

                # métricas por classe
                # Classe 0
                mlflow.log_metric("mus_precision_class_0", report['0']['precision'])
                mlflow.log_metric("mus_recall_class_0", report['0']['recall'])
                mlflow.log_metric("mus_f1_class_0", report['0']['f1-score'])

                # Classe 1
                mlflow.log_metric("mus_precision_class_1", report['1']['precision'])
                mlflow.log_metric("mus_recall_class_1", report['1']['recall'])
                mlflow.log_metric("mus_f1_class_1", report['1']['f1-score'])

            # ===== SALVAR MODELO =====
            mlflow.sklearn.log_model(best_model, "model")

            results[name] = {
                'best_params': search.best_params_,
                'best_score_cv': search.best_score_,
                'test_f1': test_f1
            }

    return results

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

rfc_results = pipeline(rfc_grid, pipelines, X_train, y_train, X_test, y_test, X_musculus, y_musculus)
#rfc_model, rfc_metrics = salvar_metricas_csv(rfc_results, X_musculus, y_musculus)


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

xgb_results = pipeline(xgb_grid, pipelines, X_train, y_train, X_test, y_test, X_musculus, y_musculus)
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

gb_results = pipeline(gb_grid, pipelines, X_train, y_train, X_test, y_test, X_musculus, y_musculus)
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
