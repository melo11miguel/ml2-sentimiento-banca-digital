"""Piezas compartidas del flujo: carga, preprocesamiento, Pipeline, CV y métricas.

Principio anti-fuga: todo lo que aprende de los datos (vocabulario e IDF del
TF-IDF, selección chi2, SMOTE, el clasificador) vive DENTRO del Pipeline y se
ajusta solo con el pliegue de entrenamiento. La limpieza de texto de
`normalizar` es una función fija (no aprende nada), así que aplicarla antes
no filtra información.

X es un DataFrame con dos columnas: `texto` (el único predictor) y `fuente`
(oro/débil). `fuente` NO entra al modelo: el primer paso del Pipeline
selecciona solo `texto`. La usamos para puntuar únicamente sobre las filas de
oro, cuyas etiquetas son confiables.
"""
import os
from pathlib import Path
import re
import unicodedata

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer

SEED = 42
ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data" / "processed" / "dataset.csv"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"
CLASES = ["negativa", "positiva", "conflicto"]


# ---------------------------------------------------------------- datos
# Enmascaramiento de entidades (entrega final): el nombre de la app no debe ser
# predictor del sentimiento. Cubre variantes ("david plata", "nequii", "nubank")
# y deja "david" cuando es nombre de persona. ML2_SIN_MASCARA=1 lo desactiva
# (solo para la ablación que mide su efecto).
ENMASCARAR = os.environ.get("ML2_SIN_MASCARA") != "1"
RE_ENTIDAD = re.compile(r"\b(?:david ?p?lata|david ?plta|davi(?!d\b)\w*|nequi\w*|nu ?bank|nuu?)\b")


def normalizar(t: str, enmascarar: bool = None) -> str:
    """Limpieza determinista (no aprende de los datos)."""
    if enmascarar is None:
        enmascarar = ENMASCARAR
    t = t.lower()
    t = unicodedata.normalize("NFKD", t)
    t = "".join(c for c in t if not unicodedata.combining(c) or c == "̃")
    t = unicodedata.normalize("NFC", t)          # conserva la ñ, quita tildes
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"<(correo|numero|nombre)>", r" \1tok ", t)
    if enmascarar:
        t = RE_ENTIDAD.sub(" entidadtok ", t)
    t = re.sub(r"\d+", " num ", t)
    t = re.sub(r"(.)\1{2,}", r"\1\1", t)         # "caidaaaa" -> "caidaa"
    t = re.sub(r"[^\wñ\s]", " ", t)              # emojis y puntuación fuera
    return re.sub(r"\s+", " ", t).strip()


def normalizar_lista(textos):
    """Versión vectorizada para el Pipeline de despliegue (recibe texto crudo)."""
    return [normalizar(str(t)) for t in textos]


# Lista de NLTK pasada por la misma normalización ("más" -> "mas").
STOP_NLTK = sorted({normalizar(w) for w in
                    (ROOT / "data" / "stopwords_es_nltk.txt").read_text().split("\n") if w})


def cargar(solo_oro: bool = False):
    df = pd.read_csv(DATASET)
    if solo_oro:
        df = df[df.fuente == "oro"]
    df = df.reset_index(drop=True)
    X = pd.DataFrame({"texto": df.texto.map(normalizar), "fuente": df.fuente})
    y = df.etiqueta.map({c: i for i, c in enumerate(CLASES)}).to_numpy()
    return df, X, y


def estratos(X, y):
    """Estratificar por clase y fuente a la vez, para que cada pliegue tenga oro."""
    return np.char.add(X.fuente.to_numpy().astype(str), y.astype(str))


def cv(n=5, seed=SEED):
    return StratifiedKFold(n_splits=n, shuffle=True, random_state=seed)


def splits(X, y, n=5, seed=SEED):
    return list(cv(n, seed).split(X, estratos(X, y)))


# ---------------------------------------------------------------- métricas
def f1_macro_oro(estimator, X, y):
    """Scorer: F1 macro solo sobre filas de oro (etiqueta manual)."""
    m = (X.fuente == "oro").to_numpy()
    return f1_score(y[m], estimator.predict(X[m]), average="macro", labels=[0, 1, 2],
                    zero_division=0)


def f1_macro_todo(estimator, X, y):
    return f1_score(y, estimator.predict(X), average="macro", labels=[0, 1, 2],
                    zero_division=0)


# ---------------------------------------------------------------- representación
def _columna_texto(X):
    return X["texto"].to_numpy()


solo_texto = FunctionTransformer(_columna_texto, validate=False)


def tfidf(rep: str):
    """Representaciones comparadas. 'articulo' replica el artículo base."""
    comunes = dict(sublinear_tf=True, min_df=2, max_df=0.95)
    if rep == "articulo":   # unigramas + palabras vacías NLTK, como Alawaji y Aloraini
        return TfidfVectorizer(ngram_range=(1, 1), stop_words=STOP_NLTK, **comunes)
    if rep == "uni":
        return TfidfVectorizer(ngram_range=(1, 1), **comunes)
    if rep == "uni_bi":
        return TfidfVectorizer(ngram_range=(1, 2), **comunes)
    if rep == "char":
        return TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=60_000,
                               **comunes)
    if rep == "palabra_char":
        return FeatureUnion([("w", TfidfVectorizer(ngram_range=(1, 2), **comunes)),
                             ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                                   max_features=60_000, **comunes))])
    raise ValueError(rep)


def pipeline(clf, rep="uni_bi", k_chi2=None, smote=False):
    pasos = [("texto", solo_texto), ("tfidf", tfidf(rep))]
    if k_chi2:
        pasos.append(("chi2", SelectKBest(chi2, k=k_chi2)))
    if smote:
        pasos.append(("smote", SMOTE(random_state=SEED)))
    pasos.append(("clf", clf))
    return Pipeline(pasos)


# ---------------------------------------------------------------- evaluación
def _m_oro(X):
    return (X.fuente == "oro").to_numpy()




def f1_conflicto_oro(est, X, y):
    m = _m_oro(X)
    return f1_score(y[m], est.predict(X[m]), labels=[2], average="macro", zero_division=0)


def recall_conflicto_oro(est, X, y):
    from sklearn.metrics import recall_score
    m = _m_oro(X)
    return recall_score(y[m], est.predict(X[m]), labels=[2], average="macro", zero_division=0)


def accuracy_todo(est, X, y):
    # ravel: CatBoost devuelve predicciones con forma (n, 1)
    return float((np.ravel(est.predict(X)) == y).mean())


def f1_weighted_todo(est, X, y):
    return f1_score(y, est.predict(X), average="weighted", zero_division=0)


SCORERS = {
    "f1_macro_oro": f1_macro_oro,          # métrica principal
    "f1_conflicto_oro": f1_conflicto_oro,
    "recall_conflicto_oro": recall_conflicto_oro,
    "f1_macro_todo": f1_macro_todo,
    "accuracy_todo": accuracy_todo,        # comparables con el artículo
    "f1_weighted_todo": f1_weighted_todo,
}


def resumir(nombre, res):
    """Media y desviación estándar entre pliegues de un cross_validate."""
    fila = {"modelo": nombre}
    for k in SCORERS:
        v = res[f"test_{k}"]
        fila[k] = v.mean()
        fila[k + "_sd"] = v.std(ddof=1)
    fila["seg_ajuste"] = res["fit_time"].mean()
    return fila


def pesos_balanceados(y):
    from sklearn.utils.class_weight import compute_sample_weight
    return compute_sample_weight("balanced", y)
