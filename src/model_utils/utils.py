import numpy as np
import sklearn.metrics as metrics
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
import plotly.express as px
from sklearn.metrics import make_scorer, f1_score
from sklearn.model_selection import RandomizedSearchCV


def genereate_conf_matrix_and_metrics(X_test, y_test, logreg):

    for umbral in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45]:

        y_proba = logreg.predict_proba(X_test)[:, 1]
        y_pred_thr = (y_proba > umbral).astype(int)
        print(f"\nUmbral = {umbral}")
        print(metrics.confusion_matrix(y_test, y_pred_thr))
        print(metrics.classification_report(y_test, y_pred_thr, digits=4))



def genearate_plot_preplexity_tsne(X_scaled):
    perplexity = np.arange(45, 65, 5)
    divergence = []

    for i in perplexity:
        model = TSNE(n_components=3, init="pca", perplexity=i, max_iter=300)
        tsne_results = model.fit_transform(X_scaled)
        model.fit_transform(X_scaled)
        divergence.append(model.kl_divergence_)
    fig = px.line(x=perplexity, y=divergence, markers=True)
    fig.update_layout(xaxis_title="Perplexity Values", yaxis_title="Divergence")
    fig.update_traces(line_color="red", line_width=1)
    fig.show()


def boruta_selected_vars(X_train, y_train):
    rf = RandomForestClassifier(
        n_estimators=500,
        n_jobs=-1,
        random_state=42
    )
    boruta = BorutaPy(
        estimator=rf,
        n_estimators='auto',
        verbose=2,
        random_state=42,
        alpha=0.1
    )
    boruta.fit(X_train.values, y_train.values)

    # Máscaras de selección
    selected_features = X_train.columns[boruta.support_]
    print(f"Las variables seleccionadas con Boruta son: {selected_features}")
    tentative_features = X_train.columns[boruta.support_weak_]
    print(f"Las variables tentativas con Boruta son: {tentative_features}")

    return selected_features, tentative_features


def hyperparameter_tuning(name, model, params, X_train, y_train):
    f1_positive_scorer = make_scorer(f1_score, pos_label=1)

    random_search = RandomizedSearchCV(
        model,
        param_distributions=params[name],
        n_iter=20,
        scoring="average_precision",
        cv=5,
        random_state=42,
        n_jobs=-1
    )

    # Ajustar el modelo a los datos de entrenamiento
    random_search.fit(X_train, y_train)

    # Mostrar los mejores parámetros encontrados
    print(f"\n\nMejores parámetros para {name}: {random_search.best_params_}")
    best_params = random_search.best_params_

    # Ajustar el modelo con los mejores parámetros encontrados
    best_model = random_search.best_estimator_

    return best_model, random_search, best_params


