"""Experimento 1: reproducción fiel del protocolo de Alawaji y Aloraini (2025).

Igual que el artículo: TF-IDF de unigramas, eliminación de palabras vacías
(NLTK), hiperparámetros por defecto, CV estratificada de 5 pliegues, seis
modelos con y sin ponderación de clases, voto duro sobre DT + LR + RF.

Diferencias deliberadas (correcciones, no cambios de método):
- TF-IDF dentro del Pipeline (se ajusta en cada pliegue de entrenamiento).
- Se agregan las referencias que el artículo no reporta: clasificador de
  mayoría y métricas por clase sobre el conjunto de oro.

Salida: reports/exp1_reproduccion.csv
"""
import warnings

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from comun import REPORTS, SCORERS, SEED, cargar, pesos_balanceados, pipeline, resumir, splits

warnings.filterwarnings("ignore", category=UserWarning)


def modelos(cw):
    dt = DecisionTreeClassifier(class_weight=cw, random_state=SEED)
    lr = LogisticRegression(class_weight=cw, max_iter=2000)
    rf = RandomForestClassifier(class_weight=cw, random_state=SEED, n_jobs=1)
    return {
        "Árbol de decisión": dt,
        "Regresión logística": lr,
        "Random Forest": rf,
        "SVM (lineal)": LinearSVC(class_weight=cw, random_state=SEED),
        "XGBoost": XGBClassifier(random_state=SEED, n_jobs=1, tree_method="hist"),
        "Voting duro (DT+LR+RF)": VotingClassifier(
            [("dt", dt), ("lr", lr), ("rf", rf)], voting="hard"),
    }


def main():
    _, X, y = cargar()
    sp = splits(X, y)
    filas = []
    base = pipeline(DummyClassifier(strategy="most_frequent"), rep="articulo")
    filas.append(resumir("Clasificador de mayoría",
                         cross_validate(base, X, y, cv=sp, scoring=SCORERS, n_jobs=4)) |
                 {"ponderacion": "—"})
    for cw, etiqueta in [(None, "no"), ("balanced", "sí")]:
        for nombre, clf in modelos(cw).items():
            params = {}
            if nombre == "XGBoost" and cw:
                params = {"clf__sample_weight": pesos_balanceados(y)}
            res = cross_validate(pipeline(clf, rep="articulo"), X, y, cv=sp,
                                 scoring=SCORERS, n_jobs=4, params=params)
            filas.append(resumir(nombre, res) | {"ponderacion": etiqueta})
            print(f"{nombre:25s} pond={etiqueta:2s} F1m_oro={filas[-1]['f1_macro_oro']:.3f} "
                  f"acc={filas[-1]['accuracy_todo']:.3f} ({filas[-1]['seg_ajuste']:.0f}s)",
                  flush=True)
    out = pd.DataFrame(filas)
    out.to_csv(REPORTS / "exp1_reproduccion.csv", index=False)


if __name__ == "__main__":
    main()
