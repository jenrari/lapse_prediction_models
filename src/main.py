from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.data_utils.data_utils import read_data, prepare_dummy_cols, normalize_num_vars
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import shap
import pandas as pd
pd.options.mode.copy_on_write = True

from src.model_utils.mod_evaluation import model_evaluation_matrix, generate_auc_roc_pr_auc, run_model_and_evaluate
from src.model_utils.oversampling import over_sm, over_adasyn, over_bsm, over_svmsm
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

excel_file_path = "D:\\develop\\predict_lapse\\data\\BBDD_Impago_1.0.xlsx"
if __name__ == '__main__':

    df = read_data(excel_file_path)

    df_dummy = prepare_dummy_cols(df, ['Tipo', 'Zonas'])

    # Eliminamos la columna 'Edad' por tener una alta correlación con Carnet,
    feature_cols = ['Pago', 'Domiciliacion', 'Anyomatricula', 'Prima', 'Valor',
                    'ValorVehículo', 'Motor', 'Canal', 'Socioec', 'Antigüedad', 'Carnet',
                    'SegundoConductor', 'Figuras', 'Tipo_Furgoneta',
                    'Tipo_Moto', 'Tipo_Turismo', 'Zonas_Zona1', 'Zonas_Zona2',
                    'Zonas_Zona3', 'Zonas_Zona4', 'Zonas_Zona5', 'Zonas_Zona6',
                    'Zonas_Zona7', 'Zonas_Zona8']

    # Seleccionamos las variables numéricas para normalizar los datos
    num_vars = ['Anyomatricula', 'Prima', 'ValorVehículo', 'Socioec','Antigüedad', 'Carnet']
    cat_vars = ['Canal', 'Domiciliacion', 'Figuras', 'Motor', 'Pago', 'SegundoConductor', 'Tipo_Furgoneta', 'Tipo_Moto',
                'Tipo_Turismo', 'Valor', 'Zonas_Zona1', 'Zonas_Zona2', 'Zonas_Zona3', 'Zonas_Zona4', 'Zonas_Zona5',
                'Zonas_Zona6', 'Zonas_Zona7', 'Zonas_Zona8']

    # Creamos los dataset para los datos y la variable objetivo
    X = df_dummy[feature_cols]
    y = df_dummy['Impago']

    # Normalizamos los datos
    X_scaled = normalize_num_vars(X, num_vars)


    rf = RandomForestClassifier(
        n_estimators=500,
        n_jobs=-1,
        random_state=42
    )

    boruta = BorutaPy(
        estimator=rf,
        n_estimators='auto',
        verbose=2,
        random_state=42
    )

    boruta.fit(X_scaled.values, y.values)

    # Máscaras de selección
    selected_features = X_scaled.columns[boruta.support_]
    print(f"Estas son las variables seleccionadas con el algoritmo Boruta {selected_features}")
    tentative_features = X_scaled.columns[boruta.support_weak_]
    print(f"Estas son las variables tentativas con el algoritmo Boruta {tentative_features}")

    # pca_2d(X_scaled, y)
    # tsne_2d(X_scaled, y)
    # tsne_3d(X_scaled, y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    X = X_train[num_vars]
    X_train[num_vars] = normalize_num_vars(X, num_vars)
    X = X_test[num_vars]
    X_test[num_vars] = normalize_num_vars(X, num_vars)
    # logreg = LogisticRegression(random_state=42, max_iter=2000)


    # X_train_sm, y_train_sm = over_sm(X_train, y_train)
    # pca_2d(X_train_sm, y_train_sm)
    # tsne_2d(X_train_sm, y_train_sm)
    # tsne_3d(X_train_sm, y_train_sm)
    #logreg.fit(X_train_sm, y_train_sm)

    #
    # print("Entrenamos el modelo con lso datos oversampleados con el algoritmo Adasyn, seleccionadas las variables con Boruta y mostramos las métricas")
    # X_train_adasyn, y_train_adasyn = over_adasyn(X_train, y_train)
    # boruta.fit(X_train_adasyn.values, y_train_adasyn.values)
    #
    # # Máscaras de selección
    # selected_features_ada = X_train_adasyn.columns[boruta.support_]
    # print(f"Las variables seleccionadas con Boruta {selected_features_ada}")
    # tentative_features_ada = X_train_adasyn.columns[boruta.support_weak_]
    #
    # logreg.fit(X_train_adasyn[selected_features_ada], y_train_adasyn)
    #
    # y_train_pred = logreg.predict(X_train[selected_features_ada])
    # y_test_pred = logreg.predict(X_test[selected_features_ada])
    #
    # adasyn_metrics = model_evaluation_matrix(y_train, y_test, y_train_pred, y_test_pred, "ADASYN", (0,1))
    #
    # y_train_proba = logreg.predict_proba(X_train[selected_features_ada])[:, 1]
    # y_test_proba = logreg.predict_proba(X_test[selected_features_ada])[:, 1]
    #
    # roc_auc_dict = generate_auc_roc_pr_auc(y_train, y_test, y_train_proba, y_test_proba)
    #
    # umbral = 0.45
    # print(f"Evaluamos el modelo modificando el umbral a {umbral}")
    # y_pred_custom = (y_test_proba > umbral).astype(int)
    #
    # adasyn_metrics_proba = model_evaluation_matrix(y_train, y_test, y_train_pred, y_pred_custom, "ADASYN_proba",
    #                                          (0,1))
    #
    #
    #
    # print("Entrenamos el modelo con lso datos oversampleados con el algoritmo borderline-SMOTE, seleccionadas las variables con Boruta y mostramos las métricas")
    #
    # X_train_bsm, y_train_bsm = over_bsm(X_train, y_train)
    # boruta.fit(X_train_bsm.values, y_train_bsm.values)
    #
    # # Máscaras de selección
    # selected_features_bsm = X_train_bsm.columns[boruta.support_]
    # print(f"Las variables seleccionadas con Boruta {selected_features_bsm}")
    # tentative_features_bsm = X_train_bsm.columns[boruta.support_weak_]
    # print(f"Las variables tentativas con Boruta {tentative_features_bsm}")
    #
    # logreg.fit(X_train_bsm[selected_features_bsm], y_train_bsm)
    #
    # y_train_pred = logreg.predict(X_train[selected_features_bsm])
    # y_test_pred = logreg.predict(X_test[selected_features_bsm])
    #
    # bsm_metrics = model_evaluation_matrix(y_train, y_test, y_train_pred, y_test_pred, "borderline-SMOTE", (0, 1))
    #
    # y_train_proba = logreg.predict_proba(X_train[selected_features_bsm])[:, 1]
    # y_test_proba = logreg.predict_proba(X_test[selected_features_bsm])[:, 1]
    #
    # roc_auc_dict = generate_auc_roc_pr_auc(y_train, y_test, y_train_proba, y_test_proba)
    #
    # umbral = 0.45
    # print(f"Evaluamos el modelo modificando el umbral a {umbral}")
    # y_pred_custom = (y_test_proba > umbral).astype(int)
    # bsm_metrics_proba = model_evaluation_matrix(y_train, y_test, y_train_pred, y_pred_custom, "borderline-SMOTE_proba", (0, 1))


    print("Entrenamos el modelo con los datos oversampleados con el algoritmo SVMSMOTE, seleccionadas las variables con Boruta y mostramos las métricas")

    X_train_svmsm, y_train_svmsm = over_svmsm(X_train, y_train)
    print("Distribución de clases en y_train:")
    print(y_train_svmsm.value_counts(normalize=True))
    boruta.fit(X_train_svmsm.values, y_train_svmsm.values)

    # Máscaras de selección
    selected_features_svmsm = X_train_svmsm.columns[boruta.support_]
    print(f"Las variables seleccionadas con Boruta {selected_features_svmsm}")
    tentative_features_svmsm = X_train_svmsm.columns[boruta.support_weak_]
    print(f"Las variables tentativas con Boruta {tentative_features_svmsm}")

    # logreg.fit(X_train_svmsm[selected_features_svmsm], y_train_svmsm)
    #
    # y_train_pred = logreg.predict(X_train[selected_features_svmsm])
    # y_test_pred = logreg.predict(X_test[selected_features_svmsm])
    #
    # svmsm_metrics = model_evaluation_matrix(y_train, y_test, y_train_pred, y_test_pred, "SVM-SMOTE", (0, 1))
    #
    # y_train_proba = logreg.predict_proba(X_train[selected_features_svmsm])[:, 1]
    # y_test_proba = logreg.predict_proba(X_test[selected_features_svmsm])[:, 1]
    #
    # roc_auc_dict = generate_auc_roc_pr_auc(y_train, y_test, y_train_proba, y_test_proba)
    #
    # umbral = 0.45
    # print(f"Evaluamos el modelo modificando el umbral a {umbral}")
    # y_pred_custom = (y_test_proba > umbral).astype(int)
    # bsm_metrics_proba = model_evaluation_matrix(y_train, y_test, y_train_pred, y_pred_custom, "SVM-SMOTE_proba", (0, 1))



    # RANDOM FOREST

    # rf = RandomForestClassifier(n_estimators=100, random_state=42)
    # rf.fit(X_train_svmsm[selected_features_svmsm], y_train_svmsm)

    # y_train_pred = rf.predict(X_train[selected_features_svmsm])
    # y_test_pred = rf.predict(X_test[selected_features_svmsm])
    #
    # rf_metrics = model_evaluation_matrix(y_train, y_test, y_train_pred, y_test_pred, "RF", (0, 1))
    #
    # y_train_proba = rf.predict_proba(X_train[selected_features_svmsm])[:, 1]
    # y_test_proba = rf.predict_proba(X_test[selected_features_svmsm])[:, 1]
    #
    # roc_auc_dict = generate_auc_roc_pr_auc(y_train, y_test, y_train_proba, y_test_proba)
    # umbral = 0.45
    # print(f"Evaluamos el modelo modificando el umbral a {umbral}")
    # y_pred_custom = (y_test_proba > umbral).astype(int)
    # bsm_metrics_proba = model_evaluation_matrix(y_train, y_test, y_train_pred, y_pred_custom, "RF_proba", (0, 1))


    # XGBOOST
    models = {
        "Logistic_Regression": LogisticRegression(
            random_state=42, max_iter=2000),
        "Random_Forest": RandomForestClassifier(
            n_estimators=300, random_state=42,
            n_jobs=-1,
            max_depth=10,
            min_samples_leaf=20,
            min_samples_split=50,
            max_features='sqrt',
        ),
        "GBM_sklearn": GradientBoostingClassifier(
            random_state=42
        ),
        "XGBoost": XGBClassifier(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            eval_metric="logloss",
            use_label_encoder=False
        ),
        "LightGBM": LGBMClassifier(
            random_state=42,
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8
        ),
        "CatBoost": CatBoostClassifier(
            random_state=42,
            iterations=300,
            learning_rate=0.05,
            depth=6,
            verbose=False,
            loss_function="Logloss",
            eval_metric="AUC"
        )
    }

    # for name, model in models.items():
    #     rf_metrics, roc_auc_dict, explainer, fp_id, fn_id, X_test_fp, X_test_fn = run_model_and_evaluate(
    #         name, model, X_train_svmsm[selected_features_svmsm], y_train_svmsm,
    #         X_test[selected_features_svmsm], y_test
    #     )
    #
    #     # Subconjuntos para FP y FN
    #     X_test_fp = X_test.iloc[fp_id]
    #     X_test_fn = X_test.iloc[fn_id]




    for name, model in models.items():
        rf_metrics, roc_auc_dict, explainer, fp_id, fn_id, X_test_fp, X_test_fn  = run_model_and_evaluate(
            name, model, X_train_svmsm, y_train_svmsm,
            X_test, y_test
        )
        print(X_test_fp.head())
        print(X_test_fn.head())



