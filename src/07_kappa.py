"""Acuerdo entre anotadores y etiquetas finales.

1. Lee las dos hojas ciegas y calcula el kappa de Cohen entre anotadores, y el
   de cada anotador frente a la preetiqueta asistida (para medir su calidad).
2. Escribe desacuerdos.xlsx con los casos en que difieren (columna `consenso`
   vacía). Si ese archivo ya existe y `consenso` está completo, lo usa.
3. Con la hoja de revisión corregida, genera etiquetas_finales.csv, que
   04_dataset.py usa en lugar de las preetiquetas.

Con --simuladas lee las hojas _SIMULADA (respuestas ficticias de
simular_anotacion.py) y escribe todas las salidas con sufijo _SIMULADO, para
ensayar el flujo sin tocar ni mezclar las hojas reales.

Salida: reports/acuerdo_anotadores.csv y data/anotacion/etiquetas_finales.csv
"""
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import cohen_kappa_score

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data" / "anotacion"
ETIQ = {"positiva", "negativa", "conflicto", "irrelevante"}

SIM = "--simuladas" in sys.argv
SUF = "_SIMULADA" if SIM else ""
SUF_OUT = "_SIMULADO" if SIM else ""


def leer(nombre):
    h = pd.read_excel(D / f"hoja_ciega_{nombre}{SUF}.xlsx")
    h["etiqueta"] = h.etiqueta.astype(str).str.strip().str.lower()
    faltan = (~h.etiqueta.isin(ETIQ)).sum()
    if faltan:
        sys.exit(f"hoja_ciega_{nombre}{SUF}.xlsx tiene {faltan} filas sin etiqueta válida")
    return h.set_index("id").etiqueta


def main():
    if SIM:
        print("=" * 72)
        print("ENSAYO CON RESPUESTAS FICTICIAS (simular_anotacion.py).")
        print("Estas cifras NO miden el acuerdo entre Miguel y Nicolás.")
        print("=" * 72)
    a, b = leer("Miguel"), leer("Nicolas")
    b = b.loc[a.index]
    pre = (pd.read_csv(D / "muestra_maestra.csv")
           .assign(pre=pd.read_csv(D / "preetiquetas.csv").sort_values("fila").preetiqueta.values)
           .set_index("id").pre.loc[a.index])
    acuerdo = pd.DataFrame([
        ("Miguel vs Nicolás", cohen_kappa_score(a, b), (a == b).mean()),
        ("Miguel vs preetiqueta", cohen_kappa_score(a, pre), (a == pre).mean()),
        ("Nicolás vs preetiqueta", cohen_kappa_score(b, pre), (b == pre).mean()),
    ], columns=["par", "kappa_cohen", "acuerdo_simple"])
    acuerdo.round(3).to_csv(ROOT / "reports" / f"acuerdo_anotadores{SUF_OUT}.csv", index=False)
    print(acuerdo.round(3).to_string(index=False))
    print(pd.crosstab(a.rename("Miguel"), b.rename("Nicolás")))

    ruta_des = D / f"desacuerdos{SUF_OUT}.xlsx"
    dif = a.index[a != b]
    if not ruta_des.exists():
        textos = pd.read_csv(D / "muestra_maestra.csv").set_index("id").texto
        pd.DataFrame({"id": dif, "texto": textos.loc[dif].values, "Miguel": a.loc[dif].values,
                      "Nicolas": b.loc[dif].values, "consenso": ""}).to_excel(ruta_des, index=False)
        sys.exit(f"{len(dif)} desacuerdos escritos en {ruta_des.name}: llenen `consenso` y "
                 "vuelvan a ejecutar.")
    des = pd.read_excel(ruta_des).set_index("id")
    des["consenso"] = des.consenso.astype(str).str.strip().str.lower()
    if (~des.consenso.isin(ETIQ)).any():
        sys.exit(f"{ruta_des.name} tiene filas sin `consenso` válido.")

    ciega = a.copy()
    ciega.loc[des.index] = des.consenso
    rev = pd.read_excel(D / "hoja_revision.xlsx")
    rev["etiqueta"] = rev.etiqueta.astype(str).str.strip().str.lower()
    if (~rev.etiqueta.isin(ETIQ)).any():
        sys.exit("hoja_revision.xlsx tiene filas sin etiqueta válida.")
    print(f"Revisión: se corrigió el {(rev.etiqueta != rev.preetiqueta).mean():.1%} "
          "de las preetiquetas.")
    final = pd.concat([ciega.rename("etiqueta").reset_index(),
                       rev[["id", "etiqueta"]]], ignore_index=True)
    final.to_csv(D / f"etiquetas_finales{SUF_OUT}.csv", index=False)
    print(f"etiquetas_finales{SUF_OUT}.csv: {len(final)} filas")


if __name__ == "__main__":
    main()
