import shap
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report, \
    roc_auc_score, average_precision_score, precision_recall_curve
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from src.model_utils.oversampling import over_bsm, over_sm, over_adasyn, over_svmsm, over_random


def model_evaluation_matrix(y_train, y_test, y_train_pred, y_test_pred, title, labels=None):
    # Accuracies
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    print(f"Accuracy (train): {train_accuracy:.4f}")
    print(f"Accuracy (test) : {test_accuracy:.4f}")

    # Matriz de confusión (sobre test)
    conf_matrix = confusion_matrix(y_test, y_test_pred, labels=labels)

    if labels is None:
        labels_to_show = np.unique(y_test)
    else:
        labels_to_show = labels

    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                                  display_labels=labels_to_show)
    print(f"\nGenerando matriz de confusión para {title}:\n")
    disp.plot(cmap="viridis")
    plt.title(f"Matriz de confusión - {title}")
    plt.show()

    # Classification report
    print(f"\nGenerando classification report para {title}:\n")
    print(classification_report(y_test, y_test_pred, digits=4))
    class_report_dict = classification_report(y_test, y_test_pred, digits=4)

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
        "conf_matrix": conf_matrix,
        "classification_report": class_report_dict
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


def run_model_and_evaluate_reg_log(X_train, y_train, X_test, y_test, solver='lbfgs' ,balanced=None):

    if balanced is None:
        model = LogisticRegression(max_iter=3000,
                                   solver=solver,
                                   random_state=42)
    else:
        model = LogisticRegression(class_weight="balanced",
                                   solver=solver,
                                   max_iter=3000,
                                   random_state=42)

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    rf_metrics = model_evaluation_matrix(y_train, y_test, y_train_pred, y_test_pred, "LogisticRegresion", (0, 1))

    y_train_proba = model.predict_proba(X_train)[:, 1]
    y_test_proba = model.predict_proba(X_test)[:, 1]

    roc_auc_dict = generate_auc_roc_pr_auc(y_train, y_test, y_train_proba, y_test_proba)

    # Devolvemos indices de falsos positivos y falsos negativos
    fp_id = np.where((y_test == 0) & (y_test_pred == 1))[0]
    fn_id = np.where((y_test == 1) & (y_test_pred == 0))[0]

    return rf_metrics, roc_auc_dict, fp_id, fn_id


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


def run_model_and_evaluate_reg_log2(
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
        X_tr_fit, y_tr_fit = oversamplers[sampler](X_tr, y_tr)
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
        "sampler": sampler
    }

    return rf_metrics, roc_auc_dict, fp_id, fn_id, extra


def run_model_and_evaluate(
    name, model, X_train, y_train, X_test, y_test,
    sampler=None, min_precision=0.20, threshold_fallback=0.5
):
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
        X_tr_fit, y_tr_fit = oversamplers[sampler](X_tr, y_tr)
    else:
        X_tr_fit, y_tr_fit = X_tr, y_tr

    # 3) Fit para elegir umbral
    model.fit(X_tr_fit, y_tr_fit)

    proba_val = model.predict_proba(X_val)[:, 1]
    threshold, p_val, r_val = threshold_max_recall_given_precision(
        y_val, proba_val, min_precision=min_precision, fallback=threshold_fallback
    )

    # 4) Refit final (opcional, como haces tú)
    if sampler is not None:
        X_train_fit, y_train_fit = oversamplers[sampler](X_train, y_train)
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
        "sampler": sampler,
        "min_precision": min_precision
    }

    return rf_metrics, roc_auc_dict, fp_id, fn_id, extra


# def run_model_and_evaluate(name, model, X_train, y_train, X_test, y_test, sampler=None,
#     min_precision=0.20):
#     oversamplers = {
#         "smote": over_sm,
#         "adasyn": over_adasyn,
#         "b_smote": over_bsm,
#         "svm_smote": over_svmsm,
#         "ro": over_random
#     }
#
#     # 1) Split train/val (val sin SMOTE)
#     X_tr, X_val, y_tr, y_val = train_test_split(
#         X_train, y_train,
#         test_size=0.2,
#         stratify=y_train,
#         random_state=42
#     )
#
#     # 2) SMOTE solo en sub-train
#     if sampler is not None:
#         X_tr_fit, y_tr_fit = oversamplers[sampler](X_tr, y_tr)
#     else:
#         X_tr_fit, y_tr_fit = X_tr, y_tr
#
#     model.fit(X_tr_fit, y_tr_fit)
#
#     # 4) Elegir umbral en validación (real)
#     proba_val = model.predict_proba(X_val)[:, 1]
#     threshold = threshold_max_recall_given_precision(
#         y_val, proba_val, min_precision=min_precision
#     )
#
#     if sampler is not None:
#         X_train_fit, y_train_fit = oversamplers[sampler](X_train, y_train)
#     else:
#         X_train_fit, y_train_fit = X_train, y_train
#
#     model.fit(X_train_fit, y_train_fit)
#
#     y_train_proba = model.predict_proba(X_train)[:, 1]
#     y_test_proba = model.predict_proba(X_test)[:, 1]
#
#     y_train_pred = (y_train_proba >= threshold).astype(int)
#     y_test_pred = (y_test_proba >= threshold).astype(int)
#
#     rf_metrics = model_evaluation_matrix(y_train, y_test, y_train_pred, y_test_pred, name, (0, 1))
#
#     roc_auc_dict = generate_auc_roc_pr_auc(y_train, y_test, y_train_proba, y_test_proba)
#
#     # Devolvemos indices de falsos positivos y falsos negativos
#     fp_id = np.where((y_test == 0) & (y_test_pred == 1))[0]
#     fn_id = np.where((y_test == 1) & (y_test_pred == 0))[0]
#
#     return rf_metrics, roc_auc_dict, fp_id, fn_id


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





# def threshold_max_recall(y_true, proba, min_precision=0.20):
#     """
#     Devuelve el umbral que maximiza recall sujeto a precision >= min_precision.
#     Si no se puede alcanzar min_precision, devuelve 0.5.
#     """
#     prec, rec, thr = precision_recall_curve(y_true, proba)
#     prec_t = prec[:-1]
#     rec_t  = rec[:-1]
#
#     ok = np.where(prec_t >= min_precision)[0]
#     if len(ok) == 0:
#         return 0.5
#
#     best = ok[np.argmax(rec_t[ok])]
#     print(f"Mejor umbral elegido ha sido: {thr[best]}")
#     return thr[best]