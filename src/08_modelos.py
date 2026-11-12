"""Experimento 3: comparación de modelos con validación cruzada anidada.

- Bucle externo: 5 pliegues estratificados (clase x fuente). Estima el
  desempeño con datos que no participaron en la selección de hiperparámetros.
- Bucle interno: 3 pliegues sobre el entrenamiento externo, GridSearchCV con
  el F1 macro sobre filas de oro como criterio.
- Todo (TF-IDF, chi2, modelo) está dentro del Pipeline: cada ajuste ve solo
  su pliegue de entrenamiento.

Uso: python src/08_modelos.py [modelo ...]   (sin argumentos corre todos)
Cada modelo se guarda en reports/nested/<modelo>.json al terminar, y no se
recalcula si ya existe (borrar el archivo para repetirlo).
"""
import json
import sys
import time
import warnings

import numpy as np
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from comun import (REPORTS, SCORERS, SEED, cargar, f1_macro_oro, pesos_balanceados, pipeline,
                   splits)

warnings.filterwarnings("ignore")
OUT = REPORTS / "nested"
K = 3000   # términos retenidos por chi2 para los modelos de árboles


def lr(C=1.0, cw="balanced"):
    return LogisticRegression(C=C, class_weight=cw, max_iter=3000)


def svm_cal(C=0.1):
    # LinearSVC no da probabilidades: se calibra (sigmoide) para el voto suave.
    return CalibratedClassifierCV(LinearSVC(C=C, class_weight="balanced", random_state=SEED),
                                  method="sigmoid", cv=3)


def lgbm():
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31,
                          class_weight="balanced", subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.5, random_state=SEED, n_jobs=1, verbose=-1)


def miembros():
    """Integrantes de los ensambles, cada uno con su propia representación."""
    return [("lr_char", pipeline(lr(C=1.0), rep="char")),
            ("svm_pal", pipeline(svm_cal(), rep="uni_bi")),
            ("lgbm", pipeline(lgbm(), rep="uni_bi", k_chi2=K))]


# nombre -> (pipeline, rejilla, usa_pesos_muestrales, n_jobs_busqueda)
MODELOS = {
    "mayoria": (pipeline(DummyClassifier(strategy="most_frequent")), {}, False, 1),
    "lr_defecto": (pipeline(lr(cw=None)), {}, False, 1),
    "arbol": (pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
                       k_chi2=K),
              {"clf__max_depth": [20, None], "clf__min_samples_leaf": [1, 5]}, False, 4),
    # Representación fija en caracteres: ganó en el Experimento 2 y la búsqueda
    # anidada completa la eligió en 3 de 3 pliegues; rejillas de C alrededor del óptimo.
    "lr": (pipeline(lr(), rep="char"), {"clf__C": [0.1, 0.3, 1]}, False, 3),
    "svm": (pipeline(LinearSVC(class_weight="balanced", random_state=SEED), rep="char"),
            {"clf__C": [0.01, 0.03, 0.1, 0.3]}, False, 4),
    "random_forest": (pipeline(RandomForestClassifier(
        n_estimators=300, class_weight="balanced_subsample", random_state=SEED, n_jobs=1),
        k_chi2=K),
        {"clf__min_samples_leaf": [1, 3]}, False, 4),
    "xgboost": (pipeline(XGBClassifier(
        n_estimators=300, learning_rate=0.1, subsample=0.8, colsample_bytree=0.5,
        tree_method="hist", random_state=SEED, n_jobs=1), k_chi2=K),
        {"clf__max_depth": [4, 6]}, True, 4),
    "lightgbm": (pipeline(lgbm(), k_chi2=K),
                 {"clf__num_leaves": [15, 31]}, False, 4),
    "catboost": (pipeline(CatBoostClassifier(
        iterations=300, depth=6, learning_rate=0.1, auto_class_weights="Balanced",
        random_seed=SEED, verbose=0, thread_count=4), k_chi2=2000), {}, False, 1),
    "voting_suave": (VotingClassifier(miembros(), voting="soft", n_jobs=3), {}, False, 1),
    "stacking": (StackingClassifier(miembros(), final_estimator=lr(C=1.0), cv=3,
                                    stack_method="predict_proba", n_jobs=3), {}, False, 1),
}


def correr(nombre):
    ruta = OUT / f"{nombre}.json"
    if ruta.exists():
        print(f"{nombre}: ya calculado")
        return
    pipe, grid, pesos, nj = MODELOS[nombre]
    _, X, y = cargar()
    folds, t0 = [], time.time()
    for k, (tr, te) in enumerate(splits(X, y, 5)):
        Xtr, ytr, Xte, yte = X.iloc[tr], y[tr], X.iloc[te], y[te]
        params = {"clf__sample_weight": pesos_balanceados(ytr)} if pesos else {}
        if grid:
            gs = GridSearchCV(pipe, grid, cv=splits(Xtr, ytr, 3, seed=SEED + k),
                              scoring=f1_macro_oro, n_jobs=nj, refit=True)
            gs.fit(Xtr, ytr, **params)
            est, mejor = gs.best_estimator_, {p: str(v)[:60] for p, v in gs.best_params_.items()}
        else:
            est, mejor = pipe.fit(Xtr, ytr, **params), {}
        m_oro = (Xte.fuente == "oro").to_numpy()
        fila = {s: float(f(est, Xte, yte)) for s, f in SCORERS.items()}
        fila["mejores_params"] = mejor
        fila["confusion_oro"] = confusion_matrix(yte[m_oro], est.predict(Xte[m_oro]),
                                                 labels=[0, 1, 2]).tolist()
        folds.append(fila)
        print(f"  {nombre} pliegue {k + 1}: F1m_oro={fila['f1_macro_oro']:.3f} {mejor}",
              flush=True)
    res = {"modelo": nombre, "segundos": time.time() - t0, "pliegues": folds}
    OUT.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    v = np.array([f["f1_macro_oro"] for f in folds])
    print(f"{nombre}: F1 macro oro = {v.mean():.3f} ± {v.std(ddof=1):.3f} "
          f"({res['segundos'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    for n in (sys.argv[1:] or MODELOS):
        correr(n)
