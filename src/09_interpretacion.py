"""Experimento 4 (ablación de la supervisión débil) e interpretación preliminar.

Usa la regresión logística con la configuración más elegida en el bucle
interno de la CV anidada (moda de los 5 pliegues externos).

1. Ablación, con los mismos pliegues externos y prueba solo en oro:
   a) entrenar solo con oro, b) oro + débil, c) oro + débil con peso 0.3.
2. Términos con mayor coeficiente por clase (modelo ajustado con todo).
3. Estabilidad: solapamiento (Jaccard) de los 25 términos principales por
   clase entre los modelos de los 5 pliegues.
4. Errores: predicciones fuera de pliegue sobre oro y ejemplos por tipo.

Salidas: reports/exp4_ablacion.csv, reports/terminos_por_clase.csv,
         reports/estabilidad_terminos.csv, reports/errores_ejemplos.csv,
         reports/figures/fig6_terminos.png
"""
from collections import Counter
import json
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

from comun import (CLASES, FIGS, REPORTS, SCORERS, cargar, pipeline, splits)

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})
COL = {"negativa": "#2a78d6", "positiva": "#eb6834", "conflicto": "#1baf7a"}


def config_lr():
    r = json.loads((REPORTS / "nested" / "lr.json").read_text())
    elegidas = Counter(json.dumps(p["mejores_params"], sort_keys=True) for p in r["pliegues"])
    mejor = json.loads(elegidas.most_common(1)[0][0])
    rep = "char" if "char_wb" in mejor.get("tfidf", "char_wb") else "uni_bi"
    C = float(mejor["clf__C"])
    print(f"Configuración modal: rep={rep}, C={C}  ({elegidas.most_common(1)[0][1]}/5 pliegues)")
    return rep, C


def modelo(rep, C):
    return pipeline(LogisticRegression(C=C, class_weight="balanced", max_iter=3000), rep=rep)


def ablacion(X, y, rep, C):
    oro = (X.fuente == "oro").to_numpy()
    filas = []
    for k, (tr, te) in enumerate(splits(X, y, 5)):
        Xte, yte = X.iloc[te], y[te]
        for nombre, mask, peso_debil in [("solo oro", oro[tr], None),
                                         ("oro + débil", np.ones(len(tr), bool), 1.0),
                                         ("oro + débil (peso 0,3)", np.ones(len(tr), bool), 0.3)]:
            idx = tr[mask]
            m = modelo(rep, C)
            params = {}
            if peso_debil is not None:
                params = {"clf__sample_weight": np.where(oro[idx], 1.0, peso_debil)}
            m.fit(X.iloc[idx], y[idx], **params)
            filas.append({"pliegue": k, "entrenamiento": nombre, "n_entrenamiento": len(idx)} |
                         {s: f(m, Xte, yte) for s, f in SCORERS.items()})
    d = pd.DataFrame(filas)
    res = d.groupby("entrenamiento").agg(["mean", "std"]).drop(columns="pliegue")
    res.columns = [f"{a}_{b}" for a, b in res.columns]
    res.round(4).to_csv(REPORTS / "exp4_ablacion.csv")
    print(res[["n_entrenamiento_mean", "f1_macro_oro_mean", "f1_macro_oro_std",
               "f1_conflicto_oro_mean"]].round(3))


def top_terminos(m, n=25):
    nombres = m.named_steps["tfidf"].get_feature_names_out()
    coef = m.named_steps["clf"].coef_
    return {c: [(nombres[i], coef[k, i]) for i in np.argsort(coef[k])[::-1][:n]]
            for k, c in enumerate(CLASES)}


def interpretar(X, y, rep, C):
    # Para leer términos se usa siempre la representación por palabras:
    # los n-gramas de caracteres no son legibles para el negocio.
    rep_lect = "uni_bi"
    m = modelo(rep_lect, C).fit(X, y)
    top = top_terminos(m)
    filas = [{"clase": c, "rank": i + 1, "termino": t, "coef": round(v, 3)}
             for c, lst in top.items() for i, (t, v) in enumerate(lst)]
    pd.DataFrame(filas).to_csv(REPORTS / "terminos_por_clase.csv", index=False)

    # Estabilidad entre pliegues
    conj = {c: [] for c in CLASES}
    for tr, _ in splits(X, y, 5):
        mk = clone(modelo(rep_lect, C)).fit(X.iloc[tr], y[tr])
        for c, lst in top_terminos(mk).items():
            conj[c].append({t for t, _ in lst})
    est = []
    for c in CLASES:
        js = [len(a & b) / len(a | b) for i, a in enumerate(conj[c]) for b in conj[c][i + 1:]]
        est.append({"clase": c, "jaccard_medio": np.mean(js), "jaccard_min": np.min(js)})
    pd.DataFrame(est).round(3).to_csv(REPORTS / "estabilidad_terminos.csv", index=False)
    print(pd.DataFrame(est).round(3))

    fig, axs = plt.subplots(1, 3, figsize=(6.6, 3.4))
    for ax, c in zip(axs, CLASES):
        t = top[c][:12][::-1]
        ax.barh([a for a, _ in t], [b for _, b in t], color=COL[c], height=0.7)
        ax.set_title(c, fontsize=9)
        ax.tick_params(axis="y", labelsize=7)
        ax.tick_params(axis="x", labelsize=7)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
    axs[1].set_xlabel("Coeficiente de la regresión logística (uni+bigramas)", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig6_terminos.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def errores(df, X, y, rep, C):
    pred = np.empty(len(y), dtype=int)
    for tr, te in splits(X, y, 5):
        pred[te] = modelo(rep, C).fit(X.iloc[tr], y[tr]).predict(X.iloc[te])
    oro = (X.fuente == "oro").to_numpy()
    e = df[oro].assign(real=[CLASES[i] for i in y[oro]], pred=[CLASES[i] for i in pred[oro]])
    e = e[e.real != e.pred]
    print(e.groupby(["real", "pred"]).size())
    (e.groupby(["real", "pred"], group_keys=False).apply(lambda g: g.head(6))
     [["real", "pred", "entidad", "estrellas", "texto"]]
     .to_csv(REPORTS / "errores_ejemplos.csv", index=False))
    # ¿El error depende de la entidad? (sesgo entre apps)
    por_ent = (df[oro].assign(ok=(pred[oro] == y[oro])).groupby("entidad").ok.mean())
    print("Exactitud en oro por entidad:\n", por_ent.round(3))


def main():
    df, X, y = cargar()
    rep, C = config_lr()
    ablacion(X, y, rep, C)
    interpretar(X, y, rep, C)
    errores(df, X, y, rep, C)


if __name__ == "__main__":
    main()
