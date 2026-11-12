"""Experimentos 5 y 6: validación y prueba temporal, y modelo final.

Regla fijada ANTES de mirar la prueba (y escrita aquí para que quede constancia):

- La muestra temporal (600 reseñas publicadas después del corpus, preetiquetadas
  sin ver estrellas ni predicciones) se parte por fecha en su mediana.
- VALIDACIÓN TEMPORAL (del 1 al 29 de octubre): responde la única pregunta que
  la Entrega 2 dejó abierta: con qué datos entrenar la regresión logística
  elegida en la CV anidada (solo oro, oro + débil, oro + débil con peso 0,3).
  Gana el mayor F1 macro; el C de cada variante se elige con CV interna.
- PRUEBA TEMPORAL (del 30 de octubre al 11 de noviembre): se usa una sola vez,
  para reportar el modelo final junto a las líneas base, la reproducción del
  artículo, los modelos que le siguieron en la CV anidada y la versión sin
  enmascarar entidades. Nada de lo que se mida aquí cambia la elección.
- El modelo final se reentrena con la variante ganadora sobre el conjunto de
  modelado (nunca con datos temporales) y se guarda con la normalización
  DENTRO del Pipeline, para que la app reciba texto crudo.

Salidas: reports/exp5_validacion_temporal.csv, reports/exp6_prueba_temporal.csv,
         reports/prueba_por_entidad.csv, reports/predicciones_prueba.csv,
         reports/prueba_final.json, models/modelo_final.joblib, models/metadata.json
"""
from datetime import date
from importlib import import_module
import json
from pathlib import Path
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.preprocessing import FunctionTransformer

from comun import (CLASES, REPORTS, ROOT, SEED, cargar, f1_macro_oro, normalizar,
                   normalizar_lista, pesos_balanceados, pipeline, splits, tfidf)

warnings.filterwarnings("ignore")
ANOT = ROOT / "data" / "anotacion"
MODELS = ROOT / "models"
REJILLA_C = [0.1, 0.3, 1, 3]
N_BOOT = 2000


# ------------------------------------------------------------------ datos
def temporal(enmascarar=True):
    """Muestra temporal etiquetada, sin irrelevantes, partida por fecha."""
    m = pd.read_csv(ANOT / "muestra_temporal.csv", parse_dates=["fecha"])
    hoja = pd.read_excel(ANOT / "hoja_revision_temporal.xlsx")
    m = m.merge(hoja[["fila", "etiqueta"]], on="fila")
    m = m[m.etiqueta != "irrelevante"].reset_index(drop=True)
    corte = m.fecha.median()
    m["parte"] = np.where(m.fecha < corte, "validacion", "prueba")
    X = pd.DataFrame({"texto": m.texto.map(lambda t: normalizar(t, enmascarar)),
                      "fuente": "oro"})
    y = m.etiqueta.map({c: i for i, c in enumerate(CLASES)}).to_numpy()
    return m, X, y, corte


def lr(C=1.0):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=3000)


def ajustar_lr(X, y, peso=None):
    """LR sobre caracteres con C elegido por CV interna (F1 macro sobre oro)."""
    params = {} if peso is None else {"clf__sample_weight": peso}
    gs = GridSearchCV(pipeline(lr(), rep="char"), {"clf__C": REJILLA_C},
                      cv=splits(X, y, 3), scoring=f1_macro_oro, n_jobs=4)
    gs.fit(X, y, **params)
    return gs.best_estimator_, gs.best_params_["clf__C"]


def variantes(X, y):
    oro = (X.fuente == "oro").to_numpy()
    return {"solo oro": (X[oro], y[oro], None),
            "oro + débil": (X, y, None),
            "oro + débil (peso 0,3)": (X, y, np.where(oro, 1.0, 0.3))}


# ------------------------------------------------------------------ métricas
def metricas(y, p):
    return {"f1_macro": f1_score(y, p, average="macro", labels=[0, 1, 2], zero_division=0),
            "f1_negativa": f1_score(y, p, labels=[0], average="macro", zero_division=0),
            "f1_positiva": f1_score(y, p, labels=[1], average="macro", zero_division=0),
            "f1_conflicto": f1_score(y, p, labels=[2], average="macro", zero_division=0),
            "prec_conflicto": precision_score(y, p, labels=[2], average="macro",
                                              zero_division=0),
            "recall_conflicto": recall_score(y, p, labels=[2], average="macro",
                                             zero_division=0),
            "exactitud": float((p == y).mean())}


def bootstrap(y, p, ref=None, n=N_BOOT, seed=SEED):
    """IC del 95 % del F1 macro y, si hay `ref`, de la diferencia p - ref (pareada)."""
    rng = np.random.default_rng(seed)
    f, d = [], []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        a = f1_score(y[i], p[i], average="macro", labels=[0, 1, 2], zero_division=0)
        f.append(a)
        if ref is not None:
            d.append(a - f1_score(y[i], ref[i], average="macro", labels=[0, 1, 2],
                                  zero_division=0))
    out = {"f1_macro_ic_inf": np.percentile(f, 2.5), "f1_macro_ic_sup": np.percentile(f, 97.5)}
    if ref is not None:
        out |= {"dif_vs_final_inf": np.percentile(d, 2.5),
                "dif_vs_final_sup": np.percentile(d, 97.5)}
    return out


# ------------------------------------------------------------------ experimentos
def validacion(X, y, Xv, yv):
    filas, ajustados = [], {}
    for nombre, (Xtr, ytr, peso) in variantes(X, y).items():
        t0 = time.time()
        est, C = ajustar_lr(Xtr, ytr, peso)
        p = est.predict(Xv)
        filas.append({"entrenamiento": nombre, "n": len(ytr), "C": C} | metricas(yv, p) |
                     bootstrap(yv, p))
        ajustados[nombre] = (est, C)
        print(f"  {nombre:24s} C={C:<4} F1m={filas[-1]['f1_macro']:.3f} "
              f"F1conf={filas[-1]['f1_conflicto']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    d = pd.DataFrame(filas)
    d.round(4).to_csv(REPORTS / "exp5_validacion_temporal.csv", index=False)
    ganadora = d.sort_values("f1_macro", ascending=False).entrenamiento.iloc[0]
    print(f"Variante elegida en validación temporal: {ganadora}")
    return ganadora, ajustados[ganadora]


def comparados(X, y, variante, C_final):
    """Modelos a contrastar en la prueba (todos se entrenan con el conjunto de modelado)."""
    m08 = import_module("08_modelos")
    m05 = import_module("05_reproduccion")
    Xtr, ytr, peso = variantes(X, y)[variante]
    final = pipeline(lr(C_final), rep="char")
    out = {
        "Regresión logística (final)": (final, Xtr, ytr, peso),
        "Stacking": (m08.MODELOS["stacking"][0], X, y, None),
        "CatBoost": (m08.MODELOS["catboost"][0], X, y, None),
        "Protocolo del artículo (voto duro)": (
            pipeline(m05.modelos("balanced")["Voting duro (DT+LR+RF)"], rep="articulo"),
            X, y, None),
        "Reg. logística por defecto (línea base)": (
            pipeline(LogisticRegression(max_iter=3000)), X, y, None),
        "Mayoría (línea base)": (pipeline(DummyClassifier(strategy="most_frequent")),
                                 X, y, None),
    }
    return out


def prueba(m, X, y, Xt, yt, variante, C_final):
    filas, preds = [], {}
    for nombre, (est, Xtr, ytr, peso) in comparados(X, y, variante, C_final).items():
        t0 = time.time()
        params = {} if peso is None else {"clf__sample_weight": peso}
        est = clone(est).fit(Xtr, ytr, **params)
        preds[nombre] = np.ravel(est.predict(Xt))
        print(f"  {nombre:40s} ({time.time() - t0:.0f}s)", flush=True)
        if nombre.endswith("(final)"):
            modelo_final = est

    # Ablación del enmascaramiento: el mismo modelo final, sin máscara.
    _, Xs, ys = cargar_sin_mascara()
    Xtr_s, ytr_s, peso_s = variantes(Xs, ys)[variante]
    params = {} if peso_s is None else {"clf__sample_weight": peso_s}
    sin = pipeline(lr(C_final), rep="char").fit(Xtr_s, ytr_s, **params)
    _, Xt_s, _, _ = temporal(enmascarar=False)
    Xt_s = Xt_s[(m.parte == "prueba").to_numpy()]
    preds["Regresión logística sin enmascarar entidades"] = sin.predict(Xt_s)

    ref = preds["Regresión logística (final)"]
    for nombre, p in preds.items():
        filas.append({"modelo": nombre} | metricas(yt, p) |
                     bootstrap(yt, p, None if p is ref else ref))
    d = pd.DataFrame(filas)
    d.round(4).to_csv(REPORTS / "exp6_prueba_temporal.csv", index=False)
    print(d[["modelo", "f1_macro", "f1_macro_ic_inf", "f1_macro_ic_sup", "f1_conflicto",
             "exactitud"]].round(3).to_string(index=False))
    return modelo_final, preds


def cargar_sin_mascara():
    df, X, y = cargar()
    X = X.assign(texto=df.texto.map(lambda t: normalizar(t, enmascarar=False)))
    return df, X, y


def por_entidad(mt, yt, preds):
    filas = []
    for nombre in ["Regresión logística (final)", "Regresión logística sin enmascarar entidades"]:
        for ent, g in mt.assign(i=range(len(mt))).groupby("entidad"):
            i = g.i.to_numpy()
            filas.append({"modelo": nombre, "entidad": ent, "n": len(i)} |
                         metricas(yt[i], preds[nombre][i]))
    d = pd.DataFrame(filas)
    d.round(4).to_csv(REPORTS / "prueba_por_entidad.csv", index=False)
    print(d[["modelo", "entidad", "n", "f1_macro", "exactitud"]].round(3).to_string(index=False))


def despliegue(est_cv, X, y, variante, C_final, mt, Xt, yt):
    """Pipeline de despliegue: recibe texto crudo y normaliza dentro."""
    Xtr, ytr, peso = variantes(X, y)[variante]
    pasos = [("normalizar", FunctionTransformer(normalizar_lista, validate=False)),
             ("tfidf", tfidf("char")), ("clf", lr(C_final))]
    dep = SkPipeline(pasos)
    df, _, _ = cargar()
    crudo = df.texto.loc[Xtr.index].tolist()
    dep.fit(crudo, ytr, **({} if peso is None else {"clf__sample_weight": peso}))
    # Coherencia entrenamiento-uso: mismas predicciones que el modelo evaluado.
    p_dep = dep.predict(mt.texto.tolist())
    p_cv = est_cv.predict(Xt)
    assert (p_dep == p_cv).all(), "el pipeline de despliegue no coincide con el evaluado"
    MODELS.mkdir(exist_ok=True)
    joblib.dump(dep, MODELS / "modelo_final.joblib")
    meta = {"modelo": "Regresión logística multinomial, class_weight='balanced'",
            "representacion": "TF-IDF de n-gramas de caracteres (char_wb, 2 a 5), "
                              "sublinear_tf, min_df=2, max_df=0.95, máx. 60 000",
            "C": C_final, "entrenamiento": variante, "n_entrenamiento": int(len(ytr)),
            "clases": CLASES, "enmascara_entidades": True, "semilla": SEED,
            "fecha_entrenamiento": str(date.today()), "sklearn": sklearn.__version__,
            "prueba_temporal": metricas(yt, p_dep) | {"n": int(len(yt))}}
    (MODELS / "metadata.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print("Modelo de despliegue guardado; coincide con el evaluado en las", len(yt),
          "reseñas de prueba.")
    return dep


def main():
    df, X, y = cargar()
    m, Xtmp, ytmp, corte = temporal()
    val = (m.parte == "validacion").to_numpy()
    print(f"Corte por fecha: {corte}. Validación={val.sum()}, prueba={(~val).sum()}")
    print(pd.crosstab(m.parte, m.etiqueta))

    print("\nExperimento 5: validación temporal")
    variante, (_, C_final) = validacion(X, y, Xtmp[val], ytmp[val])

    print("\nExperimento 6: prueba temporal (una sola vez)")
    mt, Xt, yt = m[~val].reset_index(drop=True), Xtmp[~val], ytmp[~val]
    est, preds = prueba(m, X, y, Xt, yt, variante, C_final)
    por_entidad(mt, yt, preds)

    p = preds["Regresión logística (final)"]
    cm = confusion_matrix(yt, p, labels=[0, 1, 2])
    (REPORTS / "prueba_final.json").write_text(json.dumps(
        {"variante": variante, "C": C_final, "corte": str(corte), "n_prueba": int(len(yt)),
         "confusion": cm.tolist(), "clases": CLASES}, indent=1, ensure_ascii=False))
    mt.assign(real=[CLASES[i] for i in yt], pred=[CLASES[i] for i in p])[
        ["fila", "entidad", "estrellas", "fecha", "real", "pred", "texto"]].to_csv(
        REPORTS / "predicciones_prueba.csv", index=False)
    print(cm)
    despliegue(est, X, y, variante, C_final, mt, Xt, yt)


if __name__ == "__main__":
    main()
