"""Figuras del informe (PNG a 200 ppp en reports/figures).

Paleta: tres primeros tonos categóricos de la paleta de referencia validada
(seguros para daltonismo en todos los pares); rampa secuencial de un solo tono
para los mapas de calor. Cada figura se puede generar por separado.
Uso: python src/10_figuras.py [nombre ...]
"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from comun import CLASES, FIGS, REPORTS

COL = {"negativa": "#2a78d6", "positiva": "#eb6834", "conflicto": "#1baf7a"}
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GRIS = "#a3a29c"
AZUL = LinearSegmentedColormap.from_list("azul", ["#eef4fc", "#2a78d6", "#0d3a73"])

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.edgecolor": GRIS, "axes.labelcolor": TXT2,
    "xtick.color": TXT2, "ytick.color": TXT2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": False, "figure.dpi": 200,
    "savefig.bbox": "tight", "legend.frameon": False,
})

NOMBRES = {
    "mayoria": "Mayoría (línea base)", "lr_defecto": "Reg. logística por defecto (línea base)",
    "arbol": "Árbol de decisión", "lr": "Regresión logística", "svm": "SVM lineal",
    "random_forest": "Random Forest", "xgboost": "XGBoost", "lightgbm": "LightGBM",
    "catboost": "CatBoost", "voting_suave": "Voting suave", "stacking": "Stacking",
}


def guardar(fig, nombre):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{nombre}.png")
    plt.close(fig)
    print("ok", nombre)


def ruido():
    """Composición de la etiqueta manual por nivel de estrellas (muestra anotada)."""
    from comun import ROOT
    d = ROOT / "data" / "anotacion"
    m = pd.read_csv(d / "muestra_maestra.csv")
    fin = d / "etiquetas_finales.csv"
    if fin.exists():
        m = m.merge(pd.read_csv(fin), on="id")
    else:
        m["etiqueta"] = pd.read_csv(d / "preetiquetas.csv").sort_values("fila").preetiqueta.values
    m = m[m.etiqueta != "irrelevante"]
    tab = pd.crosstab(m.estrellas, m.etiqueta, normalize="index")[CLASES]
    n = m.estrellas.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.2, 2.6))
    izq = np.zeros(len(tab))
    for c in CLASES:
        ax.barh(tab.index.astype(str), tab[c], left=izq, color=COL[c], label=c,
                edgecolor="white", linewidth=1.5, height=0.65)
        for i, (v, l) in enumerate(zip(tab[c], izq)):
            if v >= 0.08:
                ax.text(l + v / 2, i, f"{v:.0%}", ha="center", va="center", color="white",
                        fontsize=8, fontweight="bold")
        izq += tab[c].to_numpy()
    for i, e in enumerate(tab.index):
        ax.text(1.01, i, f"n = {n[e]}", va="center", color=TXT2, fontsize=8)
    ax.set_yticklabels([f"{e} estrella" + ("s" if e > 1 else "") for e in tab.index])
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("Proporción de cada etiqueta en la muestra anotada")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    guardar(fig, "fig1_estrellas_vs_etiqueta")


def reproduccion():
    """Exactitud (como la reporta el artículo) frente a F1 macro en oro."""
    d = pd.read_csv(REPORTS / "exp1_reproduccion.csv")
    d = d[(d.ponderacion == "sí") | (d.modelo == "Clasificador de mayoría")]
    d = d.iloc[::-1]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    h = 0.36
    ax.barh(y + h / 2, d.accuracy_todo, h, color="#2a78d6", label="Exactitud (todo el corpus)")
    ax.barh(y - h / 2, d.f1_macro_oro, h, xerr=d.f1_macro_oro_sd, color="#eb6834",
            error_kw={"ecolor": TXT2, "elinewidth": 0.8, "capsize": 2},
            label="F1 macro (muestra anotada)")
    for yi, a, f in zip(y, d.accuracy_todo, d.f1_macro_oro):
        ax.text(a + 0.01, yi + h / 2, f"{a:.2f}", va="center", fontsize=7, color=TXT2)
        ax.text(f + 0.04, yi - h / 2, f"{f:.2f}", va="center", fontsize=7, color=TXT2)
    ax.axvline(0.9024, color=TXT2, lw=0.8, ls="--")
    ax.text(0.9024, len(d) - 0.3, "artículo: 0,90", ha="center", fontsize=7, color=TXT2)
    ax.set_yticks(y, d.modelo)
    ax.set_xlim(0, 1.05)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.45, 1.02))
    guardar(fig, "fig2_reproduccion")


def representacion():
    d = pd.read_csv(REPORTS / "exp2_representacion.csv")
    etq_r = {"articulo": "Artículo", "uni": "Uni", "uni_bi": "Uni+bi",
             "char": "Caract.", "palabra_char": "Pal.+car."}
    etq_b = {"ninguno": "Sin tratamiento", "ponderacion": "Ponderación", "smote": "SMOTE"}
    fig, axs = plt.subplots(1, 2, figsize=(6.4, 2.5), sharey=True)
    for ax, mod in zip(axs, ["LR", "SVM"]):
        t = d[d.modelo == mod].pivot(index="balance", columns="representacion",
                                     values="f1_macro_oro").loc[list(etq_b), list(etq_r)]
        ax.imshow(t.values, cmap=AZUL, vmin=0.4, vmax=0.75, aspect="auto")
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                v = t.values[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 0.6 else TXT)
        ax.set_xticks(range(t.shape[1]), [etq_r[c] for c in t.columns], fontsize=7.5)
        ax.set_yticks(range(t.shape[0]), [etq_b[c] for c in t.index])
        ax.set_title("Regresión logística" if mod == "LR" else "SVM lineal", fontsize=9,
                     color=TXT)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    guardar(fig, "fig3_representacion")


def _nested():
    filas = []
    for f in sorted((REPORTS / "nested").glob("*.json")):
        r = json.loads(f.read_text())
        v = np.array([p["f1_macro_oro"] for p in r["pliegues"]])
        c = np.array([p["f1_conflicto_oro"] for p in r["pliegues"]])
        a = np.array([p["accuracy_todo"] for p in r["pliegues"]])
        filas.append({"clave": r["modelo"], "modelo": NOMBRES[r["modelo"]], "f1": v.mean(),
                      "sd": v.std(ddof=1), "f1_conf": c.mean(), "f1_conf_sd": c.std(ddof=1),
                      "acc": a.mean(), "min": r["segundos"] / 60,
                      "conf": np.sum([p["confusion_oro"] for p in r["pliegues"]], axis=0)})
    return pd.DataFrame(filas).sort_values("f1")


def modelos():
    d = _nested()
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    y = np.arange(len(d))
    base = d.clave.isin(["mayoria", "lr_defecto"]).to_numpy()
    for marca, col, et in [(~base, "#2a78d6", "Modelos del curso"), (base, GRIS, "Líneas base")]:
        ax.errorbar(d.f1[marca], y[marca], xerr=d.sd[marca], fmt="o", ms=5, color=col,
                    ecolor=col, elinewidth=1.2, capsize=2.5, label=et)
    ax.scatter(d.f1_conf, y, marker="D", s=14, color="#1baf7a", zorder=3,
               label="F1 de la clase conflicto")
    for yi, v, s in zip(y, d.f1, d.sd):
        ax.text(v + s + 0.012, yi, f"{v:.3f}", va="center", fontsize=7, color=TXT2)
    ax.set_yticks(y, d.modelo)
    ax.set_xlim(0, 0.9)
    ax.set_xlabel("F1 macro sobre la muestra anotada (media ± DE, 5 pliegues externos)")
    ax.xaxis.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.4, 1.0), fontsize=7.5)
    guardar(fig, "fig4_modelos_anidada")
    d.drop(columns="conf").round(4).to_csv(REPORTS / "exp3_resumen_anidada.csv", index=False)


def confusion(clave=None):
    d = _nested()
    fila = d[d.clave == clave].iloc[0] if clave else d.iloc[-1]
    cm = np.array(fila.conf)
    cmn = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    ax.imshow(cmn, cmap=AZUL, vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{cmn[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=7.5, color="white" if cmn[i, j] > 0.55 else TXT)
    ax.set_xticks(range(3), CLASES)
    ax.set_yticks(range(3), CLASES)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta anotada")
    ax.set_title(fila.modelo, fontsize=9, color=TXT)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    guardar(fig, "fig5_confusion")


TODAS = {"ruido": ruido, "reproduccion": reproduccion, "representacion": representacion,
         "modelos": modelos, "confusion": confusion}

if __name__ == "__main__":
    for n in (sys.argv[1:] or TODAS):
        TODAS[n]()
