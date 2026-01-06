import shap
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, \
    roc_auc_score, average_precision_score, precision_recall_curve, balanced_accuracy_score, precision_score, \
    recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.model_utils.oversampling import over_bsm, over_sm, over_adasyn, over_svmsm, over_random


def model_evaluation_matrix(
    y_train, y_test,
    y_train_pred, y_test_pred,
    title="model",
    labels=(0, 1),
    show_plot=True,
    verbose=True
):
    # ---- Métricas principales (train/test) ----
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy  = accuracy_score(y_test,  y_test_pred)

    # En desbalanceo, mejor añadir también:
    train_bal_acc = balanced_accuracy_score(y_train, y_train_pred)
    test_bal_acc  = balanced_accuracy_score(y_test,  y_test_pred)

    # Precision/Recall/F1 para clase positiva (pos_label=1)
    precision_test = precision_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    recall_test    = recall_score(y_test, y_test_pred, pos_label=1, zero_division=0)
    f1_test        = f1_score(y_test, y_test_pred, pos_label=1, zero_division=0)

    # Matriz de confusión (test)
    conf_matrix = confusion_matrix(y_test, y_test_pred, labels=labels)
    tn, fp, fn, tp = conf_matrix.ravel()

    # Classification report en formato dict (numérico)
    report_dict = classification_report(
        y_test, y_test_pred,
        labels=labels,
        digits=4,
        output_dict=True,
        zero_division=0
    )

    # ---- Prints opcionales ----
    if verbose:
        print(f"[{title}] Accuracy train: {train_accuracy:.4f} | test: {test_accuracy:.4f}")
        print(f"[{title}] Balanced Acc train: {train_bal_acc:.4f} | test: {test_bal_acc:.4f}")
        print(f"[{title}] Precision(1): {precision_test:.4f} | Recall(1): {recall_test:.4f} | F1(1): {f1_test:.4f}")
        print(f"[{title}] Confusion matrix (test):\n{conf_matrix}")

    # ---- Plot opcional ----
    if show_plot:
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=list(labels))
        disp.plot(cmap="viridis")
        plt.title(f"Matriz de confusión - {title}")
        plt.show()

    return {
        # métricas simples (perfectas para DataFrame / sort)
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "train_balanced_accuracy": train_bal_acc,
        "test_balanced_accuracy": test_bal_acc,
        "precision_test": precision_test,
        "recall_test": recall_test,
        "f1_test": f1_test,

        # componentes CM por si quieres análisis FP/FN
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),

        # extra por si quieres imprimir o auditar
        "conf_matrix": conf_matrix,
        "classification_report": report_dict
    }


def generate_auc_roc_pr_auc(y_train, y_test, y_train_proba, y_test_proba):
    auc_roc_train = roc_auc_score(y_train, y_train_proba)
    auc_roc_test = roc_auc_score(y_test, y_test_proba)

    print(f"AUC-ROC (train): {auc_roc_train:.4f}")
    print(f"AUC-ROC (test) : {auc_roc_test:.4f}")

    pr_auc_train = average_precision_score(y_train, y_train_proba)
    pr_auc_test = average_precision_score(y_test, y_test_proba)

    print(f"PR-AUC  (train): {pr_auc_train:.4f}")
    print(f"PR-AUC  (test) : {pr_auc_test:.4f}")

    return {"auc-roc-train":auc_roc_train,
            "auc-roc-test":auc_roc_test,
            "pr-auc-train":pr_auc_train,
            "pr-auc-test":pr_auc_test}


def threshold_max_recall_given_precision(y_true, y_proba, min_precision=0.20, fallback=0.5):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    prec_t, rec_t = prec[:-1], rec[:-1]  # alineados con thr

    mask = prec_t >= min_precision
    if not np.any(mask):
        return fallback, None, None

    valid_idx = np.where(mask)[0]
    best_abs = valid_idx[np.argmax(rec_t[mask])]
    return thr[best_abs], prec_t[best_abs], rec_t[best_abs]


def best_threshold_max_fbeta(y_true, y_proba, beta=2.0):
    prec, rec, thr = precision_recall_curve(y_true, y_proba)
    prec_t, rec_t = prec[:-1], rec[:-1]
    b2 = beta**2
    fbeta = (1+b2) * prec_t * rec_t / (b2*prec_t + rec_t + 1e-12)
    i = np.argmax(fbeta)
    return float(thr[i]), float(prec_t[i]), float(rec_t[i]), float(fbeta[i])


def run_model_and_evaluate_reg_log(
    X_train, y_train, X_test, y_test, solver='lbfgs', sampler=None, sampling_strategy=None,
        balanced=None):

    oversamplers = {
        "smote": over_sm,
        "adasyn": over_adasyn,
        "b_smote": over_bsm,
        "svm_smote": over_svmsm,
        "ro": over_random
    }

    if balanced is None:
        model = LogisticRegression(max_iter=3000,
                                   solver=solver,
                                   random_state=42)
    else:
        model = LogisticRegression(class_weight="balanced",
                                   solver=solver,
                                   max_iter=3000,
                                   random_state=42)

    # 1) Split train/val (val sin oversampling)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # 2) Oversampling solo en sub-train
    if sampler is not None:
        X_tr_fit, y_tr_fit = oversamplers[sampler](X_tr, y_tr, sampling_strategy)
    else:
        X_tr_fit, y_tr_fit = X_tr, y_tr

    # 3) Fit para elegir umbral
    model.fit(X_tr_fit, y_tr_fit)

    proba_val = model.predict_proba(X_val)[:, 1]
    threshold, p_val, r_val, fbeta_val = best_threshold_max_fbeta(y_val, proba_val)

    # 4) Refit final (opcional, como haces tú)
    if sampler is not None:
        X_train_fit, y_train_fit = oversamplers[sampler](X_train, y_train, sampling_strategy)
    else:
        X_train_fit, y_train_fit = X_train, y_train

    model.fit(X_train_fit, y_train_fit)

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba  = model.predict_proba(X_test)[:, 1]

    y_train_pred = (y_train_proba >= threshold).astype(int)
    y_test_pred  = (y_test_proba  >= threshold).astype(int)

    rf_metrics = model_evaluation_matrix(y_train, y_test, y_train_pred, y_test_pred, "LogisticRegression", (0, 1))
    roc_auc_dict = generate_auc_roc_pr_auc(y_train, y_test, y_train_proba, y_test_proba)

    fp_id = np.where((y_test == 0) & (y_test_pred == 1))[0]
    fn_id = np.where((y_test == 1) & (y_test_pred == 0))[0]

    extra = {
        "threshold": threshold,
        "val_precision_at_threshold": p_val,
        "val_recall_at_threshold": r_val,
        "val_fbeta_at_threshold": fbeta_val,
        "sampler": sampler
    }

    return rf_metrics, roc_auc_dict, fp_id, fn_id, extra


def run_model_and_evaluate(
    X_train, y_train, X_test, y_test, name, model, sampler=None, sampling_strategy=None):

    oversamplers = {
        "smote": over_sm,
        "adasyn": over_adasyn,
        "b_smote": over_bsm,
        "svm_smote": over_svmsm,
        "ro": over_random
    }


    # 1) Split train/val (val sin oversampling)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )

    # 2) Oversampling solo en sub-train
    if sampler is not None:
        X_tr_fit, y_tr_fit = oversamplers[sampler](X_tr, y_tr, sampling_strategy)
    else:
        X_tr_fit, y_tr_fit = X_tr, y_tr

    # 3) Fit para elegir umbral
    model.fit(X_tr_fit, y_tr_fit)

    proba_val = model.predict_proba(X_val)[:, 1]
    threshold, p_val, r_val, fbeta_val = best_threshold_max_fbeta(y_val, proba_val)

    # 4) Refit final (opcional, como haces tú)
    if sampler is not None:
        X_train_fit, y_train_fit = oversamplers[sampler](X_train, y_train, sampling_strategy)
    else:
        X_train_fit, y_train_fit = X_train, y_train

    model.fit(X_train_fit, y_train_fit)

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba  = model.predict_proba(X_test)[:, 1]

    y_train_pred = (y_train_proba >= threshold).astype(int)
    y_test_pred  = (y_test_proba  >= threshold).astype(int)

    rf_metrics = model_evaluation_matrix(y_train, y_test, y_train_pred, y_test_pred, name, (0, 1))
    roc_auc_dict = generate_auc_roc_pr_auc(y_train, y_test, y_train_proba, y_test_proba)

    fp_id = np.where((y_test == 0) & (y_test_pred == 1))[0]
    fn_id = np.where((y_test == 1) & (y_test_pred == 0))[0]

    extra = {
        "threshold": threshold,
        "val_precision_at_threshold": p_val,
        "val_recall_at_threshold": r_val,
        "val_fbeta_at_threshold": fbeta_val,
        "sampler": sampler,
        "ratio": sampling_strategy,
        "model": name
    }

    return rf_metrics, roc_auc_dict, fp_id, fn_id, extra


def run_model_and_evaluate_xgb(
        X_train, y_train, X_test, y_test,
        params,
        sampler=None,
        sampling_strategy=None  # <- recomendado para reproducibilidad
):
    """
    - calibration set SOLO para umbral
    - early stopping con split train_es / val_es (sin oversampling en val_es)
    - refit final con train_es + val_es usando best_iteration
    - oversampling SOLO en el sub-train de cada entrenamiento (nunca en val/calib/test)
    """

    oversamplers = {
        "smote": over_sm,
        "adasyn": over_adasyn,
        "b_smote": over_bsm,
        "svm_smote": over_svmsm,
        "ro": over_random
    }

    # 0) Split calibration (no se toca)
    X_dev, X_cal, y_dev, y_cal = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
        random_state=42
    )

    # 1) Split para early stopping: train_es / val_es
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_dev, y_dev,
        test_size=0.2,
        stratify=y_dev,
        random_state=42
    )

    # 2) Oversampling SOLO en train_es
    if sampler is not None:
        X_tr_fit, y_tr_fit = oversamplers[sampler](X_tr, y_tr, sampling_strategy)
    else:
        X_tr_fit, y_tr_fit = X_tr, y_tr

    # 3) Fit con early stopping (guarda best_iteration)
    early_stop = xgboost.callback.EarlyStopping(
        rounds=150,
        metric_name="aucpr",
        data_name="validation_0",
        save_best=True,
        maximize=True
    )

    xgb_es = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        callbacks=[early_stop],
        **params
    )

    xgb_es.fit(
        X_tr_fit, y_tr_fit,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    best_iter = int(xgb_es.best_iteration)
    n_estimators_used = best_iter + 1
    print("Best iteration: ", best_iter)

    # 4) Refit final con (train_es + val_es) y n_estimators = best_iter+1
    #    (oversampling SOLO en el conjunto de refit)
    X_refit = np.vstack([X_tr, X_val]) if hasattr(X_tr, "shape") else X_tr.append(X_val)
    y_refit = np.concatenate([y_tr, y_val])

    if sampler is not None:
        X_refit_fit, y_refit_fit = oversamplers[sampler](X_refit, y_refit, sampling_strategy)
    else:
        X_refit_fit, y_refit_fit = X_refit, y_refit

    params2 = params.copy()
    params2["n_estimators"] = n_estimators_used

    xgb_final = XGBClassifier(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        **params2
    )

    xgb_final.fit(X_refit_fit, y_refit_fit, verbose=False)

    # 5) Threshold en calibration (modelo final; calib nunca visto)
    proba_cal = xgb_final.predict_proba(X_cal)[:, 1]
    threshold, p_cal, r_cal, fbeta_cal = best_threshold_max_fbeta(y_cal, proba_cal)

    # 6) Probabilidades y predicción final
    y_train_proba = xgb_final.predict_proba(X_train)[:, 1]
    y_test_proba = xgb_final.predict_proba(X_test)[:, 1]

    y_train_pred = (y_train_proba >= threshold).astype(int)
    y_test_pred = (y_test_proba >= threshold).astype(int)

    rf_metrics = model_evaluation_matrix(y_train, y_test, y_train_pred, y_test_pred, "XGBoost", (0, 1))
    roc_auc_dict = generate_auc_roc_pr_auc(y_train, y_test, y_train_proba, y_test_proba)

    fp_id = np.where((y_test == 0) & (y_test_pred == 1))[0]
    fn_id = np.where((y_test == 1) & (y_test_pred == 0))[0]

    extra = {
        "threshold": float(threshold),
        "cal_precision_at_threshold": float(p_cal),
        "cal_recall_at_threshold": float(r_cal),
        "cal_fbeta_at_threshold": float(fbeta_cal),
        "sampler": sampler,
        "ratio": sampling_strategy,
        "model": "XGBoost",
        "best_iteration": best_iter,
        "n_estimators_used": n_estimators_used
    }
    print(extra)

    return rf_metrics, roc_auc_dict, fp_id, fn_id, extra


def shap_test(name, model, X_train, X_test, fp_id, fn_id):
# Calculamos shap para el modelo
    if name in ["Logistic_Regression"]:
        explainer = shap.Explainer(model, X_train)
        # SHAP values para test
        shap_values = explainer(X_test)
        expected_value = explainer.expected_value

        # Importancia global de variables
        shap.plots.bar(shap_values, max_display=20)

        # Falsos positivos y falsos negativos
        X_test_fp = X_test.iloc[fp_id]
        X_test_fn= X_test.iloc[fn_id]
        shap_fp = shap_values[fp_id, :]
        shap_fn = shap_values[fn_id, :]

        #Importancia para falsos positivos
        shap.plots.bar(shap_fp, max_display=20)

        # Importancia para falsos negativos
        shap.plots.bar(shap_fn, max_display=20)

    else:
        X_train_shap = X_train.copy()

        # Fuerza todo a float (los bool pasan a 0.0 / 1.0)
        X_train_shap = X_train_shap.astype(float)
        X_test_shap = X_test.astype(float)
        cols = list(X_train_shap.columns)

        explainer = shap.TreeExplainer(model, X_train_shap)
        shap_values = explainer(X_test_shap)
        # expected_value_impago = explainer.expected_value[1]

        shap_values.feature_names = cols
        shap.plots.bar(shap_values, max_display=20)

        # Falsos positivos y falsos negativos
        X_test_fp = X_test.iloc[fp_id]
        X_test_fn = X_test.iloc[fn_id]
        shap_fp = shap_values[fp_id]
        shap_fn = shap_values[fn_id]

        # Importancia para falsos positivos
        shap.plots.bar(shap_fp, max_display=20)

        # Importancia para falsos negativos
        shap.plots.bar(shap_fn, max_display=20)

    # umbral = 0.45
    # print(f"Evaluamos el modelo modificando el umbral a {umbral}")
    # y_pred_custom = (y_test_proba > umbral).astype(int)
    # bsm_metrics_proba = model_evaluation_matrix(y_train, y_test, y_train_pred, y_pred_custom, name, (0, 1))

    return explainer, fp_id, fn_id, X_test_fp, X_test_fn


