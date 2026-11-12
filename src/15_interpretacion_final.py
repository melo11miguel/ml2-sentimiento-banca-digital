"""Interpretación del modelo final y figuras de la entrega final.

1. SHAP por palabra sobre las reseñas de la prueba temporal (texto crudo, el
   mismo que recibe la app). Agregamos por palabra normalizada: aporte medio a
   cada clase cuando la palabra aparece (mínimo MIN_FREC apariciones).
2. Ejemplos locales: un conflicto detectado, un conflicto perdido y una queja
   leída como positiva.
3. Figuras: comparación en la prueba temporal (F1 macro con IC del 95 %),
   matriz de confusión de la prueba, SHAP global y SHAP local.

Salidas: reports/shap_global.csv, reports/shap_ejemplos.csv,
         reports/figures/fig7_prueba_temporal.png, fig8_shap_global.png, fig9_shap_local.png
"""
from collections import defaultdict
import json
import time
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd

from comun import CLASES, FIGS, REPORTS, ROOT, normalizar
from explicar import explicador, palabras

warnings.filterwarnings("ignore")
plt.rcParams.update({"font.family": "serif", "axes.spines.top": False,
                     "axes.spines.right": False})
COL = {"negativa": "#2a78d6", "positiva": "#eb6834", "conflicto": "#1baf7a"}
MIN_FREC = 5


def shap_prueba(modelo, pred):
    t0 = time.time()
    sv = explicador(modelo)(pred.texto.tolist())
    print(f"SHAP de {len(pred)} reseñas en {time.time() - t0:.0f}s")
    return sv


def global_(sv, n):
    acum, frec = defaultdict(lambda: np.zeros(3)), defaultdict(int)
    for i in range(n):
        vistos = set()
        for tok, v in palabras(sv, i):
            w = normalizar(tok)
            if not w:
                continue
            acum[w] += v
            if w not in vistos:
                frec[w] += 1
                vistos.add(w)
    filas = [{"palabra": w, "frecuencia": frec[w]} |
             {c: acum[w][k] / frec[w] for k, c in enumerate(CLASES)}
             for w in acum if frec[w] >= MIN_FREC]
    d = pd.DataFrame(filas)
    d.round(4).sort_values("frecuencia", ascending=False).to_csv(
        REPORTS / "shap_global.csv", index=False)
    return d


def fig_global(d):
    fig, axs = plt.subplots(1, 3, figsize=(6.6, 3.3))
    for ax, c in zip(axs, CLASES):
        t = d.nlargest(10, c)[::-1]
        ax.barh(t.palabra, t[c], color=COL[c], height=0.7)
        ax.set_title(c, fontsize=9)
        ax.tick_params(labelsize=7)
    axs[1].set_xlabel("Aporte SHAP medio a la probabilidad de la clase", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig8_shap_global.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def elegir_ejemplos(pred):
    """Ejemplos cortos y legibles de tres situaciones."""
    corta = pred.texto.str.split().str.len().between(8, 22)
    casos = [("Conflicto detectado", (pred.real == "conflicto") & (pred.pred == "conflicto")),
             ("Conflicto perdido", (pred.real == "conflicto") & (pred.pred != "conflicto")),
             ("Queja leída como positiva", (pred.real == "negativa") & (pred.pred == "positiva"))]
    out = []
    for titulo, m in casos:
        cand = pred[m & corta]
        if len(cand) == 0:
            cand = pred[m]
        if len(cand):
            out.append((titulo, cand.index[0]))
    return out


def fig_local(sv, pred, ejemplos, modelo):
    filas = []
    fig, axs = plt.subplots(len(ejemplos), 1, figsize=(6.8, 1.05 * len(ejemplos) + 0.4))
    axs = np.atleast_1d(axs)
    norm = TwoSlopeNorm(vcenter=0, vmin=-0.3, vmax=0.3)
    cmap = plt.get_cmap("PiYG")
    k_conf = CLASES.index("conflicto")
    for ax, (titulo, i) in zip(axs, ejemplos):
        r = pred.loc[i]
        prob = modelo.predict_proba([r.texto])[0]
        ax.axis("off")
        ax.set_title(f"{titulo}: anotada {r.real}, predicha {r.pred} "
                     f"(P(conflicto) = {prob[k_conf]:.2f})", fontsize=8, loc="left")
        x, yv = 0.0, 0.62
        rend = fig.canvas.get_renderer()
        for tok, v in palabras(sv, i):
            t = ax.text(x, yv, tok, fontsize=8, transform=ax.transAxes, va="center",
                        bbox=dict(boxstyle="round,pad=0.15", lw=0,
                                  fc=cmap(norm(v[k_conf]))))
            bb = t.get_window_extent(renderer=rend).transformed(ax.transAxes.inverted())
            x += bb.width + 0.012
            if x > 0.93:
                x, yv = 0.0, yv - 0.42
            filas.append({"ejemplo": titulo, "palabra": tok} |
                         {c: round(float(v[k]), 4) for k, c in enumerate(CLASES)})
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, ax=axs, orientation="horizontal", fraction=0.04, pad=0.02,
                      aspect=50)
    cb.set_label("Aporte SHAP a la probabilidad de conflicto", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    fig.savefig(FIGS / "fig9_shap_local.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    pd.DataFrame(filas).to_csv(REPORTS / "shap_ejemplos.csv", index=False)


def fig_prueba():
    d = pd.read_csv(REPORTS / "exp6_prueba_temporal.csv")
    info = json.loads((REPORTS / "prueba_final.json").read_text())
    d = d.sort_values("f1_macro")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.9), width_ratios=[1.7, 1])
    yy = np.arange(len(d))
    base = d.modelo.str.contains("línea base")
    col = np.where(d.modelo.str.contains("final"), "#0B5A3C",
                   np.where(base, "#999999", "#2a78d6"))
    a1.errorbar(d.f1_macro, yy, xerr=[d.f1_macro - d.f1_macro_ic_inf,
                                      d.f1_macro_ic_sup - d.f1_macro],
                fmt="none", ecolor="#bbbbbb", capsize=2, lw=1)
    a1.scatter(d.f1_macro, yy, c=col, s=22, zorder=3)
    a1.scatter(d.f1_conflicto, yy, marker="D", s=14, c="#1baf7a", zorder=3,
               label="F1 de conflicto")
    for yi, v in zip(yy, d.f1_macro):
        a1.text(v, yi + 0.28, f"{v:.2f}", fontsize=6, ha="center")
    a1.set_yticks(yy, d.modelo.str.replace(" (línea base)", " (base)"), fontsize=6.5)
    a1.set_xlabel("F1 macro en la prueba temporal (IC 95 % bootstrap)", fontsize=7)
    a1.tick_params(axis="x", labelsize=7)
    a1.set_xlim(0, 1)
    a1.legend(fontsize=6, loc="lower right", frameon=False)

    cm = np.array(info["confusion"])
    pct = cm / cm.sum(1, keepdims=True)
    a2.imshow(pct, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            a2.text(j, i, f"{pct[i, j]:.0%}\n({cm[i, j]})", ha="center", va="center",
                    fontsize=7, color="white" if pct[i, j] > 0.5 else "black")
    a2.set_xticks(range(3), CLASES, fontsize=7)
    a2.set_yticks(range(3), CLASES, fontsize=7)
    a2.set_xlabel("Predicción", fontsize=7)
    a2.set_ylabel("Etiqueta anotada", fontsize=7)
    a2.set_title("Modelo final", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "fig7_prueba_temporal.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    modelo = joblib.load(ROOT / "models" / "modelo_final.joblib")
    pred = pd.read_csv(REPORTS / "predicciones_prueba.csv")
    fig_prueba()
    sv = shap_prueba(modelo, pred)
    d = global_(sv, len(pred))
    for c in CLASES:
        print(c, d.nlargest(10, c)[["palabra", "frecuencia", c]].round(3).values.tolist())
    fig_global(d)
    fig_local(sv, pred, elegir_ejemplos(pred), modelo)


if __name__ == "__main__":
    main()
