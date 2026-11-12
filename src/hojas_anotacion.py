"""Genera las hojas de anotación en Excel (con lista desplegable de etiquetas).

- hoja_ciega_Miguel.xlsx / hoja_ciega_Nicolas.xlsx: subconjunto ciego, sin
  estrellas ni preetiqueta, para el kappa de Cohen.
- hoja_revision.xlsx: resto de la muestra con la preetiqueta para corregir.
No sobrescribe hojas que ya existan (para no borrar trabajo hecho).
"""
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font
from openpyxl.worksheet.datavalidation import DataValidation

D = Path(__file__).resolve().parents[1] / "data" / "anotacion"
ETIQ = ["positiva", "negativa", "conflicto", "irrelevante"]
# Tamaño de la hoja ciega. 120 reseñas bastan para estimar el kappa y son unos
# 15 minutos de trabajo por anotador.
N_HOJA_CIEGA = 120


def escribir(df, ruta, columnas):
    if ruta.exists():
        print(f"ya existe, no se toca: {ruta.name}")
        return
    wb = Workbook()
    ws = wb.active
    ws.title = "anotacion"
    ws.append(columnas)
    for c in ws[1]:
        c.font = Font(bold=True)
    for fila in df[columnas].itertuples(index=False):
        ws.append(list(fila))
    col_et = columnas.index("etiqueta") + 1
    letra = ws.cell(1, col_et).column_letter
    dv = DataValidation(type="list", formula1='"' + ",".join(ETIQ) + '"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{letra}2:{letra}{len(df) + 1}")
    ws.column_dimensions["A"].width = 38
    ws.column_dimensions["B"].width = 90
    ws.column_dimensions[letra].width = 14
    for (c,) in ws.iter_rows(min_row=2, min_col=2, max_col=2):
        c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"
    wb.save(ruta)
    print(f"{ruta.name}: {len(df)} filas")


def main():
    m = pd.read_csv(D / "muestra_maestra.csv")
    m["preetiqueta"] = pd.read_csv(D / "preetiquetas.csv").sort_values("fila").preetiqueta.values
    m["duda"] = ""
    ciega = (m[m.ciega].groupby("estrellas", group_keys=False)
             .sample(frac=N_HOJA_CIEGA / m.ciega.sum(), random_state=7)
             .sample(frac=1, random_state=7).assign(etiqueta=""))
    for nombre in ["Miguel", "Nicolas"]:
        escribir(ciega, D / f"hoja_ciega_{nombre}.xlsx", ["id", "texto", "etiqueta", "duda"])
    # Todo lo que no está en la hoja ciega va a la de revisión, para que entre
    # las dos cubran la muestra completa y no se pierda ninguna reseña.
    rev = m[~m.id.isin(ciega.id)].assign(etiqueta=lambda d: d.preetiqueta)
    escribir(rev, D / "hoja_revision.xlsx", ["id", "texto", "etiqueta", "duda", "preetiqueta"])


if __name__ == "__main__":
    main()
